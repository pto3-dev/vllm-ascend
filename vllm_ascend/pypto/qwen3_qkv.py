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

"""PyPTO-backed Qwen3-14B packed QKV projection for pure Decode."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

_BATCH_PAD = 16
_HIDDEN = 5120
_QKV_SIZE = 7168


class Qwen3QKVExecutor:
    """Run a compiled packed QKV projection on the selected Ascend NPU."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "qkv_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "qkv_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import get_shared_worker

        kernel = importlib.import_module("qkv_vllm").qwen3_qkv_decode
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
        self._call_config.runtime_env.ring_task_window = 1024
        self._call_config.runtime_env.ring_heap = 8 * 1024 * 1024
        self._call_config.runtime_env.ring_dep_pool = 32768
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

    def run(self, x: torch.Tensor, qkv_weight: torch.Tensor) -> torch.Tensor:
        from simpler.task_interface import ChipStorageTaskArgs

        if x.ndim != 2 or x.shape[1] != _HIDDEN or not (1 <= x.shape[0] <= _BATCH_PAD):
            raise ValueError(f"Unsupported Qwen3 QKV input shape: {tuple(x.shape)}")
        if tuple(qkv_weight.shape) != (_QKV_SIZE, _HIDDEN):
            raise ValueError(f"Unexpected QKV weight shape: {tuple(qkv_weight.shape)}")
        for tensor in (x, qkv_weight):
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Qwen3 PyPTO QKV requires contiguous BF16 tensors")
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Qwen3 PyPTO QKV tensors must reside on the selected NPU")

        with self._lock:
            batch = x.shape[0]
            padded = torch.zeros((_BATCH_PAD, _HIDDEN), dtype=torch.bfloat16, device=x.device)
            padded[:batch].copy_(x)
            output = torch.empty((_BATCH_PAD, _QKV_SIZE), dtype=torch.bfloat16, device=x.device)
            torch.npu.synchronize(self._device_id)

            args = ChipStorageTaskArgs()
            for tensor in (padded, qkv_weight, output):
                args.add_tensor(self._chip_tensor(tensor))
            self._worker.run(self._handle, args, self._call_config)
            torch.npu.synchronize(self._device_id)
            return output[:batch]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._handle)
