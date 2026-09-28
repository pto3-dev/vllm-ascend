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

"""Opt-in PyPTO input/post-attention RMSNorm for Qwen3-14B pure Decode."""

from __future__ import annotations

import logging
import threading

import torch

from vllm_ascend import envs
from vllm_ascend.pypto.qwen3_rmsnorm import Qwen3RMSNormExecutor
from vllm_ascend.pypto.qwen3_runtime import get_pypto_paths

MODE = envs.VLLM_ASCEND_PYPTO_QWEN3_RMS_MODE
if MODE not in ("off", "shadow", "replace"):
    raise ValueError(f"Invalid VLLM_ASCEND_PYPTO_QWEN3_RMS_MODE: {MODE}")

_LOG = logging.getLogger(__name__)
_EXECUTOR: Qwen3RMSNormExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_DISPATCH_COUNT = 0


def _get_executor(device_id: int) -> Qwen3RMSNormExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            root, build_dir = get_pypto_paths("rmsnorm")
            _EXECUTOR = Qwen3RMSNormExecutor(root, build_dir, device_id)
        elif _EXECUTOR.device_id != device_id:
            raise RuntimeError("PyPTO Qwen3 RMSNorm executor cannot change NPU")
        return _EXECUTOR


def rms_forward(
    norm,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    stage: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = norm.weight
    supported = (
        hidden_states.ndim == 2
        and hidden_states.shape[1] == 5120
        and 1 <= hidden_states.shape[0] <= 16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and tuple(weight.shape) == (5120,)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and norm.hidden_size == 5120
        and norm.variance_epsilon == 1e-6
        and getattr(norm, "bias", None) is None
    )
    if not supported:
        raise RuntimeError("PyPTO Qwen3-14B Decode RMSNorm requires TP=1, BF16, no bias, <=16 tokens")

    global _DISPATCH_COUNT
    executor = _get_executor(hidden_states.device.index)
    pypto_hidden, pypto_residual = executor.run(hidden_states, weight, residual)
    _DISPATCH_COUNT += 1
    if MODE == "shadow":
        native = norm(hidden_states, residual) if residual is not None else norm(hidden_states)
        if residual is None:
            native_hidden, native_residual = native, hidden_states
        else:
            native_hidden, native_residual = native
        hidden_diff = (pypto_hidden.float() - native_hidden.float()).abs().max().item()
        residual_diff = (pypto_residual.float() - native_residual.float()).abs().max().item()
        _LOG.warning(
            "PyPTO Qwen3 RMSNorm shadow stage=%s hidden_max_abs=%.6g residual_max_abs=%.6g",
            stage,
            hidden_diff,
            residual_diff,
        )
        return native_hidden, native_residual
    if _DISPATCH_COUNT % 80 == 0:
        _LOG.warning("PyPTO Qwen3 RMSNorm dispatched %d layer norms", _DISPATCH_COUNT)
    return pypto_hidden, pypto_residual
