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
from vllm_ascend.pypto.qwen3_mlp import Qwen3MLPExecutor
from vllm_ascend.pypto.qwen3_runtime import get_pypto_paths

_LOG = logging.getLogger(__name__)
_MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE
if _MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE: {_MODE}")
_ORIGINAL_FORWARD = Qwen3DecoderLayer.forward
_EXECUTOR: Qwen3MLPExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_DISPATCH_COUNT = 0


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


def _qwen3_decode_forward(
    self: Qwen3DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _pure_decode():
        return _ORIGINAL_FORWARD(self, positions, hidden_states, residual)

    if _RMS_MODE == "off":
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
    else:
        hidden_states, residual = rms_forward(self.input_layernorm, hidden_states, residual, "input")
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
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
