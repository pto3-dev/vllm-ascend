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

"""Opt-in Qwen3-14B Decode RMSNorm and MLP replacement.

Only supported pure Decode calls use PyPTO; Prefill keeps the original forward.
"""

from __future__ import annotations

import logging
import threading

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.patch.worker.patch_qwen3_pypto_rmsnorm import (
    MODE as _RMS_MODE,
)
from vllm_ascend.patch.worker.patch_qwen3_pypto_rmsnorm import (
    rms_forward,
)
from vllm_ascend.pypto.qwen3_input_rms_qkv import Qwen3InputRMSQKVExecutor
from vllm_ascend.pypto.qwen3_mlp import Qwen3MLPExecutor
from vllm_ascend.pypto.qwen3_post_rms_mlp import Qwen3PostRMSMLPExecutor
from vllm_ascend.pypto.qwen3_runtime import get_pypto_paths

_LOG = logging.getLogger(__name__)
_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE
_FUSED_POST_RMS_MLP = envs.VLLM_ASCEND_PYPTO_QWEN3_FUSED_POST_RMS_MLP
_FUSED_INPUT_RMS_QKV = envs.VLLM_ASCEND_PYPTO_QWEN3_FUSED_INPUT_RMS_QKV
if _MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE: {_MODE}")
if _FUSED_POST_RMS_MLP and (_MODE != "replace" or _RMS_MODE != "replace"):
    raise ValueError("Fused post-RMS+MLP requires RMSNorm and MLP replace modes")
if _FUSED_INPUT_RMS_QKV and (_RMS_MODE != "replace" or envs.VLLM_ASCEND_PYPTO_QWEN3_QKV_MODE != "replace"):
    raise ValueError("Fused input-RMS+QKV requires RMSNorm and QKV replace modes")
_ORIGINAL_FORWARD = Qwen3DecoderLayer.forward
_EXECUTOR: Qwen3MLPExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_FUSED_EXECUTOR: Qwen3PostRMSMLPExecutor | None = None
_FUSED_EXECUTOR_LOCK = threading.Lock()
_FUSED_INPUT_EXECUTOR: Qwen3InputRMSQKVExecutor | None = None
_FUSED_INPUT_EXECUTOR_LOCK = threading.Lock()
_DISPATCH_COUNT = 0
_FUSED_DISPATCH_COUNT = 0
_FUSED_INPUT_DISPATCH_COUNT = 0


def _pure_decode() -> bool:
    context = get_forward_context()
    metadata = None if context is None else context.attn_metadata
    return (
        isinstance(metadata, dict)
        and bool(metadata)
        and all(getattr(item, "attn_state", None) is AscendAttentionState.DecodeOnly for item in metadata.values())
    )


def _get_executor(device_id: int) -> Qwen3MLPExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            root, build_dir = get_pypto_paths("mlp")
            _EXECUTOR = Qwen3MLPExecutor(root, build_dir, device_id)
        elif _EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 MLP executor cannot change NPU")
        return _EXECUTOR


def _get_fused_executor(device_id: int) -> Qwen3PostRMSMLPExecutor:
    global _FUSED_EXECUTOR
    with _FUSED_EXECUTOR_LOCK:
        if _FUSED_EXECUTOR is None:
            root, build_dir = get_pypto_paths("post_rms_mlp")
            _FUSED_EXECUTOR = Qwen3PostRMSMLPExecutor(root, build_dir, device_id)
        elif _FUSED_EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 fused post-RMS+MLP executor cannot change NPU")
        return _FUSED_EXECUTOR


def _get_fused_input_executor(device_id: int) -> Qwen3InputRMSQKVExecutor:
    global _FUSED_INPUT_EXECUTOR
    with _FUSED_INPUT_EXECUTOR_LOCK:
        if _FUSED_INPUT_EXECUTOR is None:
            root, build_dir = get_pypto_paths("input_rms_qkv")
            _FUSED_INPUT_EXECUTOR = Qwen3InputRMSQKVExecutor(root, build_dir, device_id)
        elif _FUSED_INPUT_EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 fused input-RMS+QKV executor cannot change NPU")
        return _FUSED_INPUT_EXECUTOR


def _qwen3_decode_forward(
    self: Qwen3DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _pure_decode():
        return _ORIGINAL_FORWARD(self, positions, hidden_states, residual)

    if _FUSED_INPUT_RMS_QKV:
        global _FUSED_INPUT_DISPATCH_COUNT
        from vllm_ascend.patch.worker.patch_qwen3_pypto_qkv import run_qwen3_attention_from_qkv

        if getattr(self.self_attn.qkv_proj, "bias", None) is not None:
            raise RuntimeError("Fused input-RMS+QKV requires a bias-free QKV projection")
        qkv, residual = _get_fused_input_executor(hidden_states.device.index).run(
            hidden_states,
            residual,
            self.input_layernorm.weight,
            self.self_attn.qkv_proj.weight,
        )
        _FUSED_INPUT_DISPATCH_COUNT += 1
        if _FUSED_INPUT_DISPATCH_COUNT % 40 == 0:
            _LOG.warning("PyPTO Qwen3 fused input-RMS+QKV dispatched %d layers", _FUSED_INPUT_DISPATCH_COUNT)
        hidden_states = run_qwen3_attention_from_qkv(self.self_attn, positions, qkv)
    else:
        if _RMS_MODE == "off":
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = rms_forward(self.input_layernorm, hidden_states, residual, "input")
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    if _FUSED_POST_RMS_MLP:
        global _FUSED_DISPATCH_COUNT
        if residual is None:
            raise RuntimeError("Fused post-RMS+MLP requires a residual tensor")
        output = _get_fused_executor(hidden_states.device.index).run(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.mlp.gate_up_proj.weight,
            self.mlp.down_proj.weight,
        )
        _FUSED_DISPATCH_COUNT += 1
        if _FUSED_DISPATCH_COUNT % 40 == 0:
            _LOG.warning("PyPTO Qwen3 fused post-RMS+MLP dispatched %d layers", _FUSED_DISPATCH_COUNT)
        return output
    if _RMS_MODE == "off":
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    else:
        hidden_states, residual = rms_forward(self.post_attention_layernorm, hidden_states, residual, "post_attention")
    if _MODE == "off":
        return self.mlp(hidden_states), residual

    gate_up_weight = self.mlp.gate_up_proj.weight
    down_weight = self.mlp.down_proj.weight
    supported = (
        hidden_states.ndim == 2
        and hidden_states.shape[1] == 5120
        and 1 <= hidden_states.shape[0] <= 16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and tuple(gate_up_weight.shape) == (34816, 5120)
        and tuple(down_weight.shape) == (5120, 17408)
        and gate_up_weight.dtype == down_weight.dtype == torch.bfloat16
        and gate_up_weight.is_contiguous()
        and down_weight.is_contiguous()
    )
    if not supported:
        raise RuntimeError("PyPTO Qwen3-14B Decode MLP requires TP=1, BF16, ND weights, <=16 tokens")

    global _DISPATCH_COUNT
    executor = _get_executor(hidden_states.device.index)
    pypto_output = executor.run(hidden_states, gate_up_weight, down_weight)
    _DISPATCH_COUNT += 1
    if _DISPATCH_COUNT % 40 == 0:
        _LOG.warning("PyPTO Qwen3 MLP dispatched %d layers", _DISPATCH_COUNT)
    if _MODE == "shadow":
        native_output = self.mlp(hidden_states)
        delta = pypto_output.float() - native_output.float()
        max_abs = delta.abs().max().item()
        ref_rms = native_output.float().square().mean().sqrt().item()
        rmse = delta.square().mean().sqrt().item()
        _LOG.warning(
            "PyPTO Qwen3 MLP shadow max_abs=%.6g rmse=%.6g ref_rms=%.6g",
            max_abs,
            rmse,
            ref_rms,
        )
        return native_output, residual
    return pypto_output, residual


if _MODE != "off" or _RMS_MODE != "off":
    Qwen3DecoderLayer.forward = _qwen3_decode_forward
    _LOG.warning(
        "Enabled PyPTO Qwen3-14B Decode MLP mode=%s RMSNorm mode=%s",
        _MODE,
        _RMS_MODE,
    )
