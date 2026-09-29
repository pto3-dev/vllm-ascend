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

"""Validate or replace native Qwen3-14B one-page paged attention with PyPTO."""

from __future__ import annotations

import logging
import math
import threading

import torch

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
)
from vllm_ascend.pypto.qwen3_paged_attention import Qwen3PagedAttentionExecutor
from vllm_ascend.pypto.qwen3_runtime import get_pypto_paths

_LOG = logging.getLogger(__name__)
_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_PA_MODE
if _MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_PA_MODE: {_MODE}")

_ORIGINAL_FORWARD_PAGED_ATTENTION = AscendAttentionBackendImpl.forward_paged_attention
_ORIGINAL_FORWARD_FUSED_ATTENTION = AscendAttentionBackendImpl.forward_fused_infer_attention
_EXECUTOR: Qwen3PagedAttentionExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_DISPATCH_COUNT = 0
_MAX_ABS = 0.0
_SUM_SQUARE = 0.0
_REF_SUM_SQUARE = 0.0
_NUMEL = 0
_WORST_DISPATCH = 0
_WORST_LAYER = 0


def _get_executor(device_id: int) -> Qwen3PagedAttentionExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            root, build_dir = get_pypto_paths("paged-attention")
            _EXECUTOR = Qwen3PagedAttentionExecutor(root, build_dir, device_id)
        elif _EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 paged-attention executor cannot change NPU")
        return _EXECUTOR


def _is_qwen3_14b_one_page(
    backend: AscendAttentionBackendImpl,
    query: torch.Tensor,
) -> bool:
    return (
        backend.num_heads == 40
        and backend.num_kv_heads == 8
        and backend.head_size == 128
        and backend.sliding_window is None
        and query.ndim == 3
        and 1 <= query.shape[0] <= 16
        and tuple(query.shape[1:]) == (40, 128)
        and query.dtype == torch.bfloat16
        and backend.key_cache is not None
        and backend.value_cache is not None
    )


def _cpu_attention_references(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build CPU references for the one-page GQA attention rounding variants."""
    batch = query.shape[0]
    physical_blocks = block_tables[:batch, 0].contiguous()
    selected_key = key_cache.index_select(0, physical_blocks)
    selected_value = value_cache.index_select(0, physical_blocks)

    query_cpu = query.detach().to(device="cpu", dtype=torch.float32).reshape(batch, 8, 5, 128)
    key_cpu = selected_key.detach().to(device="cpu", dtype=torch.float32).permute(0, 2, 1, 3)
    value_cpu = selected_value.detach().to(device="cpu", dtype=torch.float32).permute(0, 2, 1, 3)
    seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.int64)

    scores = torch.einsum("bghd,bgsd->bghs", query_cpu, key_cpu) / math.sqrt(128)
    positions = torch.arange(128).view(1, 1, 1, 128)
    valid = positions < seq_lens_cpu.view(batch, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))
    exp_scores = torch.exp(scores - scores.amax(dim=-1, keepdim=True))

    fp32_probabilities = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    fp32_context = torch.einsum("bghs,bgsd->bghd", fp32_probabilities, value_cpu)

    bf16_exp = exp_scores.to(torch.bfloat16).float()
    bf16_exp_context = torch.einsum("bghs,bgsd->bghd", bf16_exp, value_cpu)
    bf16_exp_context /= bf16_exp.sum(dim=-1, keepdim=True)

    normalized_bf16 = fp32_probabilities.to(torch.bfloat16).float()
    normalized_bf16_context = torch.einsum("bghs,bgsd->bghd", normalized_bf16, value_cpu)

    return {
        "fp32": fp32_context.reshape(batch, 40, 128).to(torch.bfloat16),
        "bf16_exp_before_sum": bf16_exp_context.reshape(batch, 40, 128).to(torch.bfloat16),
        "normalized_bf16": normalized_bf16_context.reshape(batch, 40, 128).to(torch.bfloat16),
    }


def _log_cpu_reference_diagnostics(
    dispatch: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    native_output: torch.Tensor,
    pypto_output: torch.Tensor,
) -> None:
    layer = (dispatch - 1) % 40
    if not envs.VLLM_ASCEND_PYPTO_QWEN3_PA_DIAGNOSTIC:
        return

    references = _cpu_attention_references(query, key_cache, value_cache, block_tables, seq_lens)
    native_cpu = native_output[: query.shape[0]].detach().cpu().float()
    pypto_cpu = pypto_output.detach().cpu().float()
    native_pypto_delta = (pypto_cpu - native_cpu).abs()
    worst_flat = native_pypto_delta.argmax().item()
    worst_batch = worst_flat // (40 * 128)
    worst_head = (worst_flat // 128) % 40
    worst_dim = worst_flat % 128
    _LOG.warning(
        "PyPTO Qwen3 PA worst dispatch=%d layer=%d coord=(%d,%d,%d) max_abs=%.6g native=%.6g pypto=%.6g",
        dispatch,
        layer,
        worst_batch,
        worst_head,
        worst_dim,
        native_pypto_delta.flatten()[worst_flat].item(),
        native_cpu[worst_batch, worst_head, worst_dim].item(),
        pypto_cpu[worst_batch, worst_head, worst_dim].item(),
    )
    for name, reference in references.items():
        reference = reference.float()
        native_delta = native_cpu - reference
        pypto_delta = pypto_cpu - reference
        _LOG.warning(
            "PyPTO Qwen3 PA diagnostic dispatch=%d layer=%d reference=%s "
            "reference_value=%.6g "
            "native_max_abs=%.6g native_rmse=%.6g "
            "pypto_max_abs=%.6g pypto_rmse=%.6g",
            dispatch,
            layer,
            name,
            reference[worst_batch, worst_head, worst_dim].item(),
            native_delta.abs().max().item(),
            native_delta.square().mean().sqrt().item(),
            pypto_delta.abs().max().item(),
            pypto_delta.square().mean().sqrt().item(),
        )


def _forward_paged_attention_shadow(
    self: AscendAttentionBackendImpl,
    query: torch.Tensor,
    attn_metadata,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    native_output = _ORIGINAL_FORWARD_PAGED_ATTENTION(
        self,
        query,
        attn_metadata,
        output,
    )
    if not _is_qwen3_14b_one_page(self, query):
        return native_output

    pypto_output = _get_executor(query.device.index).run(
        query,
        self.key_cache,
        self.value_cache,
        attn_metadata.block_tables,
        attn_metadata.seq_lens,
    )
    delta = pypto_output.float() - native_output[: query.shape[0]].float()

    global _DISPATCH_COUNT, _MAX_ABS, _SUM_SQUARE, _REF_SUM_SQUARE, _NUMEL, _WORST_DISPATCH, _WORST_LAYER
    call_max = delta.abs().max().item()
    next_dispatch = _DISPATCH_COUNT + 1
    new_worst = call_max > _MAX_ABS
    if new_worst:
        _MAX_ABS = call_max
        _WORST_DISPATCH = next_dispatch
        _WORST_LAYER = (next_dispatch - 1) % 40
    _DISPATCH_COUNT = next_dispatch
    _SUM_SQUARE += delta.square().sum().item()
    _REF_SUM_SQUARE += native_output[: query.shape[0]].float().square().sum().item()
    _NUMEL += delta.numel()
    if new_worst and call_max >= 0.03125:
        _log_cpu_reference_diagnostics(
            _DISPATCH_COUNT,
            query,
            self.key_cache,
            self.value_cache,
            attn_metadata.block_tables,
            attn_metadata.seq_lens,
            native_output,
            pypto_output,
        )
    if _DISPATCH_COUNT == 1:
        _LOG.warning(
            "PyPTO Qwen3 PA compare query=%s key_cache=%s block_tables=%s seq_lens=%s",
            tuple(query.shape),
            tuple(self.key_cache.shape),
            tuple(attn_metadata.block_tables.shape),
            tuple(attn_metadata.seq_lens.shape),
        )
    if _DISPATCH_COUNT % 40 == 0:
        _LOG.warning(
            "PyPTO Qwen3 PA compare dispatched=%d max_abs=%.6g rmse=%.6g ref_rms=%.6g worst_dispatch=%d worst_layer=%d",
            _DISPATCH_COUNT,
            _MAX_ABS,
            math.sqrt(_SUM_SQUARE / _NUMEL),
            math.sqrt(_REF_SUM_SQUARE / _NUMEL),
            _WORST_DISPATCH,
            _WORST_LAYER,
        )
    if _MODE == "replace":
        return pypto_output
    return native_output


def _forward_fused_attention_shadow(
    self: AscendAttentionBackendImpl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    native_output = _ORIGINAL_FORWARD_FUSED_ATTENTION(
        self,
        query,
        key,
        value,
        attn_metadata,
        output,
    )
    if attn_metadata.attn_state != AscendAttentionState.DecodeOnly or not _is_qwen3_14b_one_page(self, query):
        return native_output

    pypto_output = _get_executor(query.device.index).run(
        query,
        self.key_cache,
        self.value_cache,
        attn_metadata.block_tables,
        attn_metadata.seq_lens,
    )
    delta = pypto_output.float() - native_output[: query.shape[0]].float()

    global _DISPATCH_COUNT, _MAX_ABS, _SUM_SQUARE, _REF_SUM_SQUARE, _NUMEL, _WORST_DISPATCH, _WORST_LAYER
    call_max = delta.abs().max().item()
    next_dispatch = _DISPATCH_COUNT + 1
    new_worst = call_max > _MAX_ABS
    if new_worst:
        _MAX_ABS = call_max
        _WORST_DISPATCH = next_dispatch
        _WORST_LAYER = (next_dispatch - 1) % 40
    _DISPATCH_COUNT = next_dispatch
    _SUM_SQUARE += delta.square().sum().item()
    _REF_SUM_SQUARE += native_output[: query.shape[0]].float().square().sum().item()
    _NUMEL += delta.numel()
    if new_worst and call_max >= 0.03125:
        _log_cpu_reference_diagnostics(
            _DISPATCH_COUNT,
            query,
            self.key_cache,
            self.value_cache,
            attn_metadata.block_tables,
            attn_metadata.seq_lens,
            native_output,
            pypto_output,
        )
    if _DISPATCH_COUNT == 1:
        _LOG.warning(
            "PyPTO Qwen3 PA compare native=fused-infer query=%s key_cache=%s block_tables=%s seq_lens=%s",
            tuple(query.shape),
            tuple(self.key_cache.shape),
            tuple(attn_metadata.block_tables.shape),
            tuple(attn_metadata.seq_lens.shape),
        )
    if _DISPATCH_COUNT % 40 == 0:
        _LOG.warning(
            "PyPTO Qwen3 PA compare dispatched=%d max_abs=%.6g rmse=%.6g ref_rms=%.6g worst_dispatch=%d worst_layer=%d",
            _DISPATCH_COUNT,
            _MAX_ABS,
            math.sqrt(_SUM_SQUARE / _NUMEL),
            math.sqrt(_REF_SUM_SQUARE / _NUMEL),
            _WORST_DISPATCH,
            _WORST_LAYER,
        )
    if _MODE == "replace":
        return pypto_output
    return native_output


if _MODE in ("shadow", "replace"):
    AscendAttentionBackendImpl.forward_paged_attention = _forward_paged_attention_shadow
    AscendAttentionBackendImpl.forward_fused_infer_attention = _forward_fused_attention_shadow
    _LOG.warning("Enabled PyPTO Qwen3-14B one-page paged-attention mode=%s", _MODE)
