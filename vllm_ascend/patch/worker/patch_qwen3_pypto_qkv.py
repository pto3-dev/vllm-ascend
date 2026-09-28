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

"""Opt-in QKV and RoPE replacements for Qwen3-14B pure Decode."""

from __future__ import annotations

import logging
import threading

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3 import Qwen3Attention

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.patch.worker.patch_qwen3_pypto_mlp import _pure_decode
from vllm_ascend.pypto.qwen3_fused_attention import Qwen3FusedAttentionExecutor
from vllm_ascend.pypto.qwen3_qk_rope import Qwen3QKNormRoPEExecutor
from vllm_ascend.pypto.qwen3_qkv import Qwen3QKVExecutor
from vllm_ascend.pypto.qwen3_rope import Qwen3RoPEExecutor
from vllm_ascend.pypto.qwen3_runtime import get_pypto_paths

_LOG = logging.getLogger(__name__)
_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_QKV_MODE
_FUSED_PA_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_FUSED_PA_MODE
_PA_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_PA_MODE
_QK_ROPE_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_QK_ROPE_MODE
_ROPE_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_ROPE_MODE
if _FUSED_PA_MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_FUSED_PA_MODE: {_FUSED_PA_MODE}")
if _QK_ROPE_MODE not in ("off", "shadow"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_QK_ROPE_MODE: {_QK_ROPE_MODE}")
if _ROPE_MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_ROPE_MODE: {_ROPE_MODE}")
if _MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_QKV_MODE: {_MODE}")
if _FUSED_PA_MODE != "off" and (_PA_MODE != "off" or _QK_ROPE_MODE != "off" or _ROPE_MODE != "off"):
    raise ValueError("Fused Qwen3 attention requires PA, QK-RoPE, and RoPE modes to be off")
_ORIGINAL_FORWARD = Qwen3Attention.forward
_EXECUTOR: Qwen3QKVExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_DISPATCH_COUNT = 0
_FUSED_PA_EXECUTOR: Qwen3FusedAttentionExecutor | None = None
_FUSED_PA_EXECUTOR_LOCK = threading.Lock()
_FUSED_PA_DISPATCH_COUNT = 0
_FUSED_PA_MAX_ABS = 0.0
_QK_EXECUTOR: Qwen3QKNormRoPEExecutor | None = None
_QK_EXECUTOR_LOCK = threading.Lock()
_QK_DISPATCH_COUNT = 0

_ROPE_EXECUTOR: Qwen3RoPEExecutor | None = None
_ROPE_EXECUTOR_LOCK = threading.Lock()
_ROPE_DISPATCH_COUNT = 0


def _get_executor(device_id: int) -> Qwen3QKVExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            root, build_dir = get_pypto_paths("qkv")
            _EXECUTOR = Qwen3QKVExecutor(root, build_dir, device_id)
        elif _EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 QKV executor cannot change NPU")
        return _EXECUTOR


def _get_fused_pa_executor(device_id: int) -> Qwen3FusedAttentionExecutor:
    global _FUSED_PA_EXECUTOR
    with _FUSED_PA_EXECUTOR_LOCK:
        if _FUSED_PA_EXECUTOR is None:
            root, build_dir = get_pypto_paths("fused-pa")
            _FUSED_PA_EXECUTOR = Qwen3FusedAttentionExecutor(root, build_dir, device_id)
        elif _FUSED_PA_EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 fused-attention executor cannot change NPU")
        return _FUSED_PA_EXECUTOR


def _get_qk_executor(device_id: int) -> Qwen3QKNormRoPEExecutor:
    global _QK_EXECUTOR
    with _QK_EXECUTOR_LOCK:
        if _QK_EXECUTOR is None:
            root, build_dir = get_pypto_paths("qk-rope")
            _QK_EXECUTOR = Qwen3QKNormRoPEExecutor(root, build_dir, device_id)
        elif _QK_EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 QK-RoPE executor cannot change NPU")
        return _QK_EXECUTOR


def _get_rope_executor(device_id: int) -> Qwen3RoPEExecutor:
    global _ROPE_EXECUTOR
    with _ROPE_EXECUTOR_LOCK:
        if _ROPE_EXECUTOR is None:
            root, build_dir = get_pypto_paths("rope")
            _ROPE_EXECUTOR = Qwen3RoPEExecutor(root, build_dir, device_id)
        elif _ROPE_EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 RoPE executor cannot change NPU")
        return _ROPE_EXECUTOR


def _native_qk_rope(
    self: Qwen3Attention,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_shape = q.shape
    k_shape = k.shape
    q = self.q_norm(q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim))
    k = self.k_norm(k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim))
    return self.rotary_emb(positions, q.view(q_shape), k.view(k_shape))


def _run_qk_rope(
    self: Qwen3Attention,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _QK_ROPE_MODE == "off":
        return _native_qk_rope(self, positions, q, k)
    if (
        self.head_dim != 128
        or not getattr(self.rotary_emb, "is_neox_style", False)
        or getattr(self.rotary_emb, "rotary_dim", None) != 128
        or self.q_norm.weight.dtype != torch.bfloat16
        or self.k_norm.weight.dtype != torch.bfloat16
        or not self.q_norm.weight.is_contiguous()
        or not self.k_norm.weight.is_contiguous()
    ):
        raise RuntimeError("PyPTO Qwen3 QK-RoPE requires BF16 NeoX RoPE with head_dim=128")

    cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
    cos_sin = cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    pypto_q, pypto_k = _get_qk_executor(q.device.index).run(
        q,
        k,
        self.q_norm.weight,
        self.k_norm.weight,
        cos.contiguous(),
        sin.contiguous(),
    )

    global _QK_DISPATCH_COUNT
    _QK_DISPATCH_COUNT += 1
    if _QK_DISPATCH_COUNT % 40 == 0:
        _LOG.warning("PyPTO Qwen3 QK-RoPE dispatched %d layers", _QK_DISPATCH_COUNT)

    native_q, native_k = _native_qk_rope(self, positions, q, k)
    for name, actual, expected in (
        ("q", pypto_q, native_q),
        ("k", pypto_k, native_k),
    ):
        delta = actual.float() - expected.float()
        _LOG.warning(
            "PyPTO Qwen3 QK-RoPE shadow %s max_abs=%.6g rmse=%.6g ref_rms=%.6g",
            name,
            delta.abs().max().item(),
            delta.square().mean().sqrt().item(),
            expected.float().square().mean().sqrt().item(),
        )
    return native_q, native_k


def _run_rope_path(
    self: Qwen3Attention,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _QK_ROPE_MODE == "shadow":
        return _run_qk_rope(self, positions, q, k)

    q_shape = q.shape
    k_shape = k.shape
    q = self.q_norm(q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)).view(q_shape)
    k = self.k_norm(k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)).view(k_shape)
    if _ROPE_MODE == "off":
        return self.rotary_emb(positions, q, k)
    if (
        self.head_dim != 128
        or not getattr(self.rotary_emb, "is_neox_style", False)
        or getattr(self.rotary_emb, "rotary_dim", None) != 128
    ):
        raise RuntimeError("PyPTO Qwen3 RoPE requires NeoX RoPE with head_dim=128")

    cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
    cos_sin = cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    pypto_q, pypto_k = _get_rope_executor(q.device.index).run(
        q,
        k,
        cos.contiguous(),
        sin.contiguous(),
    )

    global _ROPE_DISPATCH_COUNT
    _ROPE_DISPATCH_COUNT += 1
    if _ROPE_DISPATCH_COUNT % 40 == 0:
        _LOG.warning("PyPTO Qwen3 RoPE dispatched %d layers", _ROPE_DISPATCH_COUNT)
    if _ROPE_MODE == "replace":
        return pypto_q, pypto_k
    native_q, native_k = self.rotary_emb(positions, q, k)

    for name, actual, expected in (
        ("q", pypto_q, native_q),
        ("k", pypto_k, native_k),
    ):
        delta = actual.float() - expected.float()
        _LOG.warning(
            "PyPTO Qwen3 RoPE shadow %s max_abs=%.6g rmse=%.6g ref_rms=%.6g",
            name,
            delta.abs().max().item(),
            delta.square().mean().sqrt().item(),
            expected.float().square().mean().sqrt().item(),
        )
    return native_q, native_k


def _run_fused_pa(self: Qwen3Attention, qkv: torch.Tensor) -> torch.Tensor:
    forward_context = get_forward_context()
    if isinstance(forward_context.attn_metadata, list):
        raise RuntimeError("PyPTO Qwen3 fused attention does not support DBO")
    metadata = forward_context.attn_metadata[self.attn.layer_name]
    backend = self.attn.impl
    if (
        metadata.attn_state != AscendAttentionState.DecodeOnly
        or backend.key_cache is None
        or backend.value_cache is None
        or backend.sliding_window is not None
    ):
        raise RuntimeError("PyPTO Qwen3 fused attention requires initialized pure-Decode paged KV cache")

    batch = qkv.shape[0]
    if metadata.seq_lens.numel() < batch or metadata.slot_mapping.numel() < batch:
        raise RuntimeError("PyPTO Qwen3 fused attention received incomplete Decode metadata")
    if not getattr(self.rotary_emb, "is_neox_style", False) or getattr(self.rotary_emb, "rotary_dim", None) != 128:
        raise RuntimeError("PyPTO Qwen3 fused attention requires 128-dim NeoX RoPE")
    cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
    rope_cos_half, rope_sin_half = cache.chunk(2, dim=-1)
    rope_cos = torch.cat((rope_cos_half, rope_cos_half), dim=-1)
    rope_sin = torch.cat((rope_sin_half, rope_sin_half), dim=-1)
    return _get_fused_pa_executor(qkv.device.index).run(
        qkv,
        backend.key_cache,
        backend.value_cache,
        metadata.block_tables,
        metadata.seq_lens,
        metadata.slot_mapping,
        rope_cos.float().contiguous(),
        rope_sin.float().contiguous(),
        self.q_norm.weight.float().view(1, 128).contiguous(),
        self.k_norm.weight.float().view(1, 128).contiguous(),
    )


def _qwen3_attention_forward(
    self: Qwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not _pure_decode():
        return _ORIGINAL_FORWARD(self, positions, hidden_states)

    weight = self.qkv_proj.weight
    supported = (
        hidden_states.ndim == 2
        and hidden_states.shape[1] == 5120
        and 1 <= hidden_states.shape[0] <= 16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and self.q_size == 5120
        and self.kv_size == 1024
        and tuple(weight.shape) == (7168, 5120)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and getattr(self.qkv_proj, "bias", None) is None
    )
    if not supported:
        raise RuntimeError("PyPTO Qwen3-14B Decode QKV requires TP=1, BF16, bias-free ND weights, <=16 tokens")

    if _MODE == "off":
        qkv, _ = self.qkv_proj(hidden_states)
    else:
        global _DISPATCH_COUNT
        pypto_qkv = _get_executor(hidden_states.device.index).run(hidden_states, weight)
        _DISPATCH_COUNT += 1
        if _DISPATCH_COUNT % 40 == 0:
            _LOG.warning("PyPTO Qwen3 QKV dispatched %d layers", _DISPATCH_COUNT)
        if _MODE == "shadow":
            native_qkv, _ = self.qkv_proj(hidden_states)
            delta = pypto_qkv.float() - native_qkv.float()
            max_abs = delta.abs().max().item()
            ref_rms = native_qkv.float().square().mean().sqrt().item()
            rmse = delta.square().mean().sqrt().item()
            _LOG.warning(
                "PyPTO Qwen3 QKV shadow max_abs=%.6g rmse=%.6g ref_rms=%.6g",
                max_abs,
                rmse,
                ref_rms,
            )
            qkv = native_qkv
        else:
            qkv = pypto_qkv

    if _FUSED_PA_MODE == "off":
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = _run_rope_path(self, positions, q, k)
        attn_output = self.attn(q, k, v)
    else:
        pypto_attn = _run_fused_pa(self, qkv).view(qkv.shape[0], self.q_size)
        global _FUSED_PA_DISPATCH_COUNT, _FUSED_PA_MAX_ABS
        _FUSED_PA_DISPATCH_COUNT += 1
        if _FUSED_PA_MODE == "shadow":
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = _run_rope_path(self, positions, q, k)
            native_attn = self.attn(q, k, v)
            delta = pypto_attn.float() - native_attn.float()
            _FUSED_PA_MAX_ABS = max(_FUSED_PA_MAX_ABS, delta.abs().max().item())
            if _FUSED_PA_DISPATCH_COUNT % 40 == 0:
                _LOG.warning(
                    "PyPTO Qwen3 fused PA dispatched=%d max_abs=%.6g rmse=%.6g ref_rms=%.6g",
                    _FUSED_PA_DISPATCH_COUNT,
                    _FUSED_PA_MAX_ABS,
                    delta.square().mean().sqrt().item(),
                    native_attn.float().square().mean().sqrt().item(),
                )
            attn_output = native_attn
        else:
            if _FUSED_PA_DISPATCH_COUNT % 40 == 0:
                _LOG.warning(
                    "PyPTO Qwen3 fused PA replaced %d layers",
                    _FUSED_PA_DISPATCH_COUNT,
                )
            attn_output = pypto_attn
    output, _ = self.o_proj(attn_output)
    return output


if _MODE != "off" or _QK_ROPE_MODE != "off" or _ROPE_MODE != "off" or _FUSED_PA_MODE != "off":
    Qwen3Attention.forward = _qwen3_attention_forward
    _LOG.warning(
        "Enabled PyPTO Qwen3-14B Decode QKV=%s QK-RoPE=%s RoPE=%s fused-PA=%s",
        _MODE,
        _QK_ROPE_MODE,
        _ROPE_MODE,
        _FUSED_PA_MODE,
    )
