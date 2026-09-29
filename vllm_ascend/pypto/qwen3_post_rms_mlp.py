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

"""One-call Qwen3-14B Decode post-attention RMSNorm and MLP executor."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

from vllm_ascend import envs

_BATCH_PAD = 16
_HIDDEN = 5120
_INTERMEDIATE = 17408


class Qwen3PostRMSMLPExecutor:
    """Run the fused post-attention norm and MLP callable for any decoder layer."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "post_rms_mlp_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "post_rms_mlp_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import configure_qwen3_call_config, get_shared_worker

        kernel = importlib.import_module("post_rms_mlp_vllm").qwen3_post_rms_mlp_decode
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
        self._cached_input: torch.Tensor | None = None
        self._cached_residual: torch.Tensor | None = None
        self._closed = False
        atexit.register(self.close)

    @property
    def device_id(self) -> int:
        return self._device_id

    @staticmethod
    def _chip_tensor(tensor: torch.Tensor):
        from simpler.task_interface import ChipTensor, DataType

        return ChipTensor.make(tensor.data_ptr(), tuple(tensor.shape), DataType.BFLOAT16, child_memory=True)

    def run(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from simpler.task_interface import ChipStorageTaskArgs

        if x.ndim != 2 or x.shape[1] != _HIDDEN or not (1 <= x.shape[0] <= _BATCH_PAD):
            raise ValueError(f"Unsupported Qwen3 post-RMS+MLP input shape: {tuple(x.shape)}")
        if residual.shape != x.shape:
            raise ValueError("Post-attention residual must match hidden states")
        if tuple(norm_weight.shape) != (_HIDDEN,):
            raise ValueError(f"Unexpected post-attention RMSNorm weight shape: {tuple(norm_weight.shape)}")
        if tuple(gate_up_weight.shape) != (2 * _INTERMEDIATE, _HIDDEN):
            raise ValueError(f"Unexpected gate/up weight shape: {tuple(gate_up_weight.shape)}")
        if tuple(down_weight.shape) != (_HIDDEN, _INTERMEDIATE):
            raise ValueError(f"Unexpected down weight shape: {tuple(down_weight.shape)}")
        for tensor in (x, residual, norm_weight, gate_up_weight, down_weight):
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Qwen3 PyPTO post-RMS+MLP requires contiguous BF16 tensors")
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Qwen3 PyPTO post-RMS+MLP tensors must reside on the selected NPU")

        with self._lock:
            from vllm_ascend.pypto.qwen3_runtime import pad_qwen3_input

            batch = x.shape[0]
            padded_x = pad_qwen3_input(x, self._cached_input, _BATCH_PAD)
            padded_residual = pad_qwen3_input(residual, self._cached_residual, _BATCH_PAD)
            if envs.VLLM_ASCEND_PYPTO_QWEN3_REUSE_INPUT_BUFFERS:
                self._cached_input = padded_x
                self._cached_residual = padded_residual
            output = torch.empty_like(padded_x)
            residual_output = torch.empty_like(padded_x)
            norm_weight_view = norm_weight.reshape(1, _HIDDEN)
            args = ChipStorageTaskArgs()
            for tensor in (
                padded_x,
                padded_residual,
                norm_weight_view,
                gate_up_weight,
                down_weight,
                output,
                residual_output,
            ):
                args.add_tensor(self._chip_tensor(tensor))
            torch.npu.synchronize(self._device_id)
            self._worker.run(self._handle, args, self._call_config)
            if not envs.VLLM_ASCEND_PYPTO_QWEN3_SKIP_POST_SYNC:
                torch.npu.synchronize(self._device_id)
            return output[:batch], residual_output[:batch]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._handle)
