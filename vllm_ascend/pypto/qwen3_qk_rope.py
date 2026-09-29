#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""PyPTO-backed Qwen3-14B Q/K RMSNorm and RoPE executor."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

_BATCH_PAD = 16
_Q_SIZE = 5120
_KV_SIZE = 1024
_HEAD_DIM = 128
_HALF_DIM = 64


class Qwen3QKNormRoPEExecutor:
    """Run per-head Q/K normalization and NeoX-style RoPE on one NPU."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "qk_norm_rope_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "qk_norm_rope_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import configure_qwen3_call_config, get_shared_worker

        kernel = importlib.import_module("qk_norm_rope_vllm").qwen3_qk_norm_rope_decode
        compiled = kernel.compile(
            config=RunConfig(
                execution_mode=ExecutionMode.ONBOARD,
                platform="a2a3",
                device_id=device_id,
                save_kernels=True,
                save_kernels_dir=build_dir,
            )
        )
        self._worker, self._lock = get_shared_worker(device_id)
        self._handle = self._worker.register_callable(compiled.chip_callable)
        self._call_config = CallConfig()
        configure_qwen3_call_config(self._call_config)
        self._device_id = device_id
        self._closed = False
        atexit.register(self.close)

    @property
    def device_id(self) -> int:
        return self._device_id

    @staticmethod
    def _chip_tensor(tensor: torch.Tensor):
        from simpler.task_interface import ChipTensor, DataType

        return ChipTensor.make(
            tensor.data_ptr(),
            tuple(tensor.shape),
            DataType.BFLOAT16,
            child_memory=True,
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from simpler.task_interface import ChipStorageTaskArgs

        batch = q.shape[0]
        if q.ndim != 2 or tuple(q.shape) != (batch, _Q_SIZE):
            raise ValueError(f"Unexpected Q shape: {tuple(q.shape)}")
        if tuple(k.shape) != (batch, _KV_SIZE) or not 1 <= batch <= _BATCH_PAD:
            raise ValueError(f"Unexpected K shape: {tuple(k.shape)}")
        if tuple(q_norm_weight.shape) not in ((_HEAD_DIM,), (1, _HEAD_DIM)):
            raise ValueError("Unexpected Q norm weight shape")
        if tuple(k_norm_weight.shape) not in ((_HEAD_DIM,), (1, _HEAD_DIM)):
            raise ValueError("Unexpected K norm weight shape")
        if tuple(rope_cos.shape) != (batch, _HALF_DIM):
            raise ValueError("Unexpected RoPE cosine shape")
        if tuple(rope_sin.shape) != (batch, _HALF_DIM):
            raise ValueError("Unexpected RoPE sine shape")
        q = q.contiguous()
        k = k.contiguous()

        tensors = (q, k, q_norm_weight, k_norm_weight, rope_cos, rope_sin)
        for tensor in tensors:
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Qwen3 PyPTO QK-RoPE requires contiguous BF16 tensors")
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Qwen3 PyPTO QK-RoPE tensors must use selected NPU")

        with self._lock:
            q_pad = torch.zeros((_BATCH_PAD, _Q_SIZE), dtype=q.dtype, device=q.device)
            k_pad = torch.zeros((_BATCH_PAD, _KV_SIZE), dtype=k.dtype, device=k.device)
            cos_pad = torch.zeros((_BATCH_PAD, _HALF_DIM), dtype=q.dtype, device=q.device)
            sin_pad = torch.zeros_like(cos_pad)
            q_pad[:batch].copy_(q)
            k_pad[:batch].copy_(k)
            cos_pad[:batch].copy_(rope_cos)
            sin_pad[:batch].copy_(rope_sin)
            q_out = torch.empty_like(q_pad)
            k_out = torch.empty_like(k_pad)
            torch.npu.synchronize(self._device_id)

            args = ChipStorageTaskArgs()
            for tensor in (
                q_pad,
                k_pad,
                q_norm_weight.reshape(1, _HEAD_DIM),
                k_norm_weight.reshape(1, _HEAD_DIM),
                cos_pad,
                sin_pad,
                q_out,
                k_out,
            ):
                args.add_tensor(self._chip_tensor(tensor))
            self._worker.run(self._handle, args, self._call_config)
            torch.npu.synchronize(self._device_id)
            return q_out[:batch], k_out[:batch]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._handle)
