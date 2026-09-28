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

"""Share one Simpler runtime across Qwen3 PyPTO Decode callables per NPU."""

from __future__ import annotations

import atexit
import os
import tempfile
import threading

from simpler.task_interface import ChipWorker
from simpler_setup.runtime_builder import RuntimeBuilder

from vllm_ascend import envs

_workers: dict[int, tuple[ChipWorker, threading.Lock]] = {}
_registry_lock = threading.Lock()


def get_pypto_paths(component: str) -> tuple[str, str]:
    """Resolve the PyPTO-Lib checkout and one component's artifact directory."""
    root = envs.VLLM_ASCEND_PYPTO_LIB_ROOT
    if not root:
        raise RuntimeError(
            "VLLM_ASCEND_PYPTO_LIB_ROOT (or legacy PYPTO_LIB_ROOT) is required for the PyPTO Qwen3 bridge"
        )
    build_root = envs.VLLM_ASCEND_PYPTO_QWEN3_BUILD_ROOT or os.path.join(tempfile.gettempdir(), "pypto-qwen3-14b")
    return root, os.path.join(build_root, component)


def get_shared_worker(device_id: int) -> tuple[ChipWorker, threading.Lock]:
    """Initialize the device once; callables may register after initialization."""
    with _registry_lock:
        existing = _workers.get(device_id)
        if existing is not None:
            return existing
        worker = ChipWorker()
        binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=False)
        worker.init(device_id=device_id, bins=binaries)
        shared = (worker, threading.Lock())
        _workers[device_id] = shared
        return shared


def _finalize_workers() -> None:
    with _registry_lock:
        for worker, _ in _workers.values():
            worker.finalize()
        _workers.clear()


atexit.register(_finalize_workers)
