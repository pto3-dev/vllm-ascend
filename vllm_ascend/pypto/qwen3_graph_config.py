# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source and artifact paths for the original Qwen3-14B graph adapter."""

from pathlib import Path
from tempfile import gettempdir

from vllm_ascend import envs


def get_graph_paths() -> tuple[str, str]:
    """Resolve the library checkout and the graph callable's artifact directory.

    Raises:
        RuntimeError: The checkout is unset or lacks the original Decode source.
    """
    root = envs.VLLM_ASCEND_PYPTO_LIB_ROOT
    if not root:
        raise RuntimeError("VLLM_ASCEND_PYPTO_LIB_ROOT (or PYPTO_LIB_ROOT) is required for the Qwen3 graph")
    source_root = Path(root).expanduser().resolve()
    if not (source_root / "models" / "qwen3_14b" / "decode_fwd.py").is_file():
        raise RuntimeError(f"Original Qwen3 Decode source not found in {source_root}")
    build_root = Path(envs.VLLM_ASCEND_PYPTO_QWEN3_BUILD_ROOT or Path(gettempdir()) / "pypto-qwen3-14b")
    return str(source_root), str(build_root.expanduser().resolve() / "original_decode40_graph")
