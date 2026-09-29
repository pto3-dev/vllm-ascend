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
"""PyPTO fused Q/K norm, RoPE, KV append, and paged attention for Qwen3-14B."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

from vllm_ascend import envs

_BATCH_MAX = 16
_NUM_Q_HEADS = 40
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 128
_Q_SIZE = _NUM_Q_HEADS * _HEAD_DIM
_KV_SIZE = _NUM_KV_HEADS * _HEAD_DIM
_QKV_SIZE = _Q_SIZE + 2 * _KV_SIZE


class Qwen3FusedAttentionExecutor:
    """Run the vLLM-specific fused PyPTO attention entry on one NPU."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "paged_attention_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "paged_attention_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import configure_qwen3_call_config, get_shared_worker

        kernel = importlib.import_module("paged_attention_vllm").qwen3_vllm_fused_attention
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

        dtype = {
            torch.bfloat16: DataType.BFLOAT16,
            torch.float32: DataType.FLOAT32,
            torch.int32: DataType.INT32,
        }.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported PyPTO tensor dtype: {tensor.dtype}")
        return ChipTensor.make(
            tensor.data_ptr(),
            tuple(tensor.shape),
            dtype,
            child_memory=True,
        )

    def run(
        self,
        qkv: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        q_norm_w: torch.Tensor,
        k_norm_w: torch.Tensor,
    ) -> torch.Tensor:
        from simpler.task_interface import ChipStorageTaskArgs

        batch = qkv.shape[0]
        if tuple(qkv.shape) != (batch, _QKV_SIZE) or not 1 <= batch <= _BATCH_MAX:
            raise ValueError(f"Unexpected QKV shape: {tuple(qkv.shape)}")
        cache_tail = (_BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM)
        if tuple(key_cache.shape[1:]) != cache_tail or value_cache.shape != key_cache.shape:
            raise ValueError("Unexpected vLLM KV-cache shape")
        if block_tables.ndim != 2 or block_tables.shape[0] < batch:
            raise ValueError(f"Unexpected block-table shape: {tuple(block_tables.shape)}")
        if seq_lens.numel() < batch or slot_mapping.numel() < batch:
            raise ValueError("Decode metadata is shorter than the QKV batch")
        if rope_cos.ndim != 2 or tuple(rope_cos.shape) != tuple(rope_sin.shape):
            raise ValueError("RoPE tables must have matching rank-2 shapes")
        if rope_cos.shape[1] != _HEAD_DIM:
            raise ValueError("RoPE table head dimension must be 128")
        for name, weight in (("q_norm", q_norm_w), ("k_norm", k_norm_w)):
            if tuple(weight.shape) != (1, _HEAD_DIM) or weight.dtype != torch.float32:
                raise ValueError(f"{name} weight must be FP32 [1, 128]")

        device_tensors = (
            qkv,
            key_cache,
            value_cache,
            block_tables,
            rope_cos,
            rope_sin,
            q_norm_w,
            k_norm_w,
        )
        for tensor in device_tensors:
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Fused Qwen3 tensors must use the selected NPU")
            if not tensor.is_contiguous():
                raise ValueError("Fused Qwen3 tensors must be contiguous")
        if qkv.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
            raise ValueError("QKV and KV cache must be BF16")
        if value_cache.dtype != torch.bfloat16:
            raise ValueError("Value cache must be BF16")
        if block_tables.dtype != torch.int32:
            raise ValueError("Block tables must be INT32")
        if rope_cos.dtype != torch.float32 or rope_sin.dtype != torch.float32:
            raise ValueError("RoPE tables must be FP32")

        with self._lock:
            block_flat = block_tables[:batch].contiguous().view(-1)
            seq_device = seq_lens[:batch].to(device=qkv.device, dtype=torch.int32, non_blocking=True).contiguous()
            slot_device = slot_mapping[:batch].to(device=qkv.device, dtype=torch.int32, non_blocking=True).contiguous()
            key_2d = key_cache.view(-1, _KV_SIZE)
            value_2d = value_cache.view(-1, _KV_SIZE)
            output = torch.empty((batch, _Q_SIZE), dtype=torch.bfloat16, device=qkv.device)
            torch.npu.synchronize(self._device_id)

            args = ChipStorageTaskArgs()
            for tensor in (
                qkv,
                key_2d,
                value_2d,
                block_flat,
                seq_device,
                slot_device,
                rope_cos,
                rope_sin,
                q_norm_w,
                k_norm_w,
                output,
            ):
                args.add_tensor(self._chip_tensor(tensor))
            self._worker.run(self._handle, args, self._call_config)
            if not envs.VLLM_ASCEND_PYPTO_QWEN3_SKIP_POST_SYNC:
                torch.npu.synchronize(self._device_id)
            return output.view(batch, _NUM_Q_HEADS, _HEAD_DIM)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._handle)
