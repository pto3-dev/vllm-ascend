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

"""PyPTO-backed Qwen3-14B Decode RMSNorm and fused-add RMSNorm."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

from vllm_ascend import envs

_BATCH_PAD = 16
_HIDDEN = 5120


class Qwen3RMSNormExecutor:
    """Compile both RMSNorm forms and reuse their callables across layers."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "rmsnorm_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "rmsnorm_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import configure_qwen3_call_config, get_shared_worker

        module = importlib.import_module("rmsnorm_vllm")

        def compile_kernel(kernel, name: str):
            return kernel.compile(
                config=RunConfig(
                    execution_mode=ExecutionMode.ONBOARD,
                    platform="a2a3",
                    device_id=device_id,
                    save_kernels=True,
                    save_kernels_dir=str(Path(build_dir) / name),
                )
            )

        plain = compile_kernel(module.qwen3_rmsnorm_decode, "plain")
        fused = compile_kernel(module.qwen3_add_rmsnorm_decode, "fused")
        self._worker, self._lock = get_shared_worker(device_id)
        self._plain_handle = self._worker.register_callable(plain.chip_callable)
        self._fused_handle = self._worker.register_callable(fused.chip_callable)
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

        return ChipTensor.make(
            tensor.data_ptr(),
            tuple(tensor.shape),
            DataType.BFLOAT16,
            child_memory=True,
        )

    def run(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from simpler.task_interface import ChipStorageTaskArgs

        if x.ndim != 2 or x.shape[1] != _HIDDEN or not (1 <= x.shape[0] <= _BATCH_PAD):
            raise ValueError(f"Unsupported Qwen3 RMSNorm input shape: {tuple(x.shape)}")
        if tuple(weight.shape) != (_HIDDEN,):
            raise ValueError(f"Unexpected RMSNorm weight shape: {tuple(weight.shape)}")
        tensors = (x, weight) if residual is None else (x, weight, residual)
        for tensor in tensors:
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Qwen3 PyPTO RMSNorm requires contiguous BF16 tensors")
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Qwen3 PyPTO RMSNorm tensors must reside on the selected NPU")
        if residual is not None and residual.shape != x.shape:
            raise ValueError("RMSNorm residual must match the hidden-state shape")

        with self._lock:
            from vllm_ascend.pypto.qwen3_runtime import pad_qwen3_input

            batch = x.shape[0]
            padded_x = pad_qwen3_input(x, self._cached_input, _BATCH_PAD)
            if envs.VLLM_ASCEND_PYPTO_QWEN3_REUSE_INPUT_BUFFERS:
                self._cached_input = padded_x
            output = torch.empty_like(padded_x)
            weight_view = weight.reshape(1, _HIDDEN)
            args = ChipStorageTaskArgs()
            if residual is None:
                for tensor in (padded_x, weight_view, output):
                    args.add_tensor(self._chip_tensor(tensor))
                handle = self._plain_handle
                residual_output = x
            else:
                padded_residual = pad_qwen3_input(residual, self._cached_residual, _BATCH_PAD)
                if envs.VLLM_ASCEND_PYPTO_QWEN3_REUSE_INPUT_BUFFERS:
                    self._cached_residual = padded_residual
                padded_residual_output = torch.empty_like(padded_x)
                for tensor in (
                    padded_x,
                    padded_residual,
                    weight_view,
                    output,
                    padded_residual_output,
                ):
                    args.add_tensor(self._chip_tensor(tensor))
                handle = self._fused_handle
                residual_output = padded_residual_output[:batch]
            torch.npu.synchronize(self._device_id)
            self._worker.run(handle, args, self._call_config)
            if not envs.VLLM_ASCEND_PYPTO_QWEN3_SKIP_POST_SYNC:
                torch.npu.synchronize(self._device_id)
            return output[:batch], residual_output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._fused_handle)
        self._worker.unregister_callable(self._plain_handle)
