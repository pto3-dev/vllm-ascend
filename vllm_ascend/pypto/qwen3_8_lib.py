# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resolve the optional source checkout before importing Qwen PTO kernels."""

import sys
from pathlib import Path

from vllm_ascend import envs


def ensure_model_root():
    configured = envs.VLLM_ASCEND_PYPTO_LIB_ROOT
    if not configured:
        raise ValueError("Set VLLM_ASCEND_PYPTO_LIB_ROOT to the PyPTO-Lib source checkout.")
    root = Path(configured).expanduser().resolve()
    model = root / "models" / "qwen3_8_27b" / "model_program.py"
    if not model.is_file():
        raise ValueError("PyPTO-Lib checkout is missing models/qwen3_8_27b/model_program.py.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
