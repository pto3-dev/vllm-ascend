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

"""PyPTO-backed Qwen3-14B one-page paged-attention executor."""

from __future__ import annotations

import atexit
import importlib
import sys
from pathlib import Path

import torch

_BATCH_PAD = 16
_NUM_Q_HEADS = 40
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 128
_Q_SIZE = _NUM_Q_HEADS * _HEAD_DIM
_KV_SIZE = _NUM_KV_HEADS * _HEAD_DIM


class Qwen3PagedAttentionExecutor:
    """Run one-page GQA attention from vLLM's device-resident cache metadata."""

    def __init__(self, pypto_lib_root: str, build_dir: str, device_id: int) -> None:
        model_dir = Path(pypto_lib_root).resolve() / "models" / "qwen3_14b"
        if not (model_dir / "paged_attention_vllm.py").is_file():
            raise FileNotFoundError(model_dir / "paged_attention_vllm.py")
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from pypto.runtime import ExecutionMode, RunConfig
        from simpler.task_interface import CallConfig

        from vllm_ascend.pypto.qwen3_runtime import get_shared_worker

        kernel = importlib.import_module("paged_attention_vllm").qwen3_paged_attention_one_page_direct
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
        self._call_config.runtime_env.ring_heap = 16 * 1024 * 1024
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

        dtype = {
            torch.bfloat16: DataType.BFLOAT16,
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
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        from simpler.task_interface import ChipStorageTaskArgs

        batch = query.shape[0]
        if tuple(query.shape[1:]) != (_NUM_Q_HEADS, _HEAD_DIM):
            raise ValueError(f"Unexpected query shape: {tuple(query.shape)}")
        if not 1 <= batch <= _BATCH_PAD:
            raise ValueError(f"Unexpected Decode batch: {batch}")
        expected_cache_tail = (_BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM)
        if tuple(key_cache.shape[1:]) != expected_cache_tail:
            raise ValueError(f"Unexpected key-cache shape: {tuple(key_cache.shape)}")
        if tuple(value_cache.shape) != tuple(key_cache.shape):
            raise ValueError("Key/value cache shapes differ")
        if block_tables.ndim != 2 or block_tables.shape[0] < batch:
            raise ValueError(f"Unexpected block-table shape: {tuple(block_tables.shape)}")
        if tuple(seq_lens.shape) != (batch,):
            raise ValueError(f"Unexpected seq_lens shape: {tuple(seq_lens.shape)}")

        tensors = (query, key_cache, value_cache)
        for tensor in tensors:
            if tensor.dtype != torch.bfloat16:
                raise ValueError("Qwen3 PyPTO paged attention requires BF16 tensors")
            if tensor.device.type != "npu" or tensor.device.index != self._device_id:
                raise ValueError("Qwen3 PyPTO tensors must use the selected NPU")
        if block_tables.dtype != torch.int32:
            raise ValueError("Block tables must be INT32")
        if block_tables.device.type != "npu" or block_tables.device.index != self._device_id:
            raise ValueError("Block tables must be device-resident")
        if seq_lens.dtype != torch.int32:
            raise ValueError("seq_lens must be INT32")
        if seq_lens.device.type not in ("cpu", "npu"):
            raise ValueError("seq_lens must be a CPU or NPU tensor")

        if bool(torch.any(seq_lens < 1)) or bool(torch.any(seq_lens > _BLOCK_SIZE)):
            raise ValueError("One-page PyPTO attention requires 1 <= seq_len <= 128")

        with self._lock:
            seq_lens_device = seq_lens.to(device=query.device, non_blocking=True)
            query_pad = torch.zeros((_BATCH_PAD, _Q_SIZE), dtype=query.dtype, device=query.device)
            physical_blocks_pad = torch.zeros((_BATCH_PAD,), dtype=torch.int32, device=query.device)
            seq_pad = torch.ones((_BATCH_PAD,), dtype=torch.int32, device=query.device)
            physical_blocks = block_tables[:batch, 0].contiguous()
            query_pad[:batch].copy_(query.reshape(batch, _Q_SIZE))
            physical_blocks_pad[:batch].copy_(physical_blocks)
            seq_pad[:batch].copy_(seq_lens_device)
            output = torch.empty_like(query_pad)
            torch.npu.synchronize(self._device_id)

            args = ChipStorageTaskArgs()
            for tensor in (
                query_pad,
                key_cache,
                value_cache,
                physical_blocks_pad,
                seq_pad,
                output,
            ):
                args.add_tensor(self._chip_tensor(tensor))
            self._worker.run(self._handle, args, self._call_config)
            torch.npu.synchronize(self._device_id)
            return output[:batch].view(batch, _NUM_Q_HEADS, _HEAD_DIM)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.unregister_callable(self._handle)
