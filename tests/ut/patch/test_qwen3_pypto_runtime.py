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

"""The Qwen3 PyPTO executors must share one Simpler runtime per device."""

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.pypto import qwen3_runtime


@pytest.mark.parametrize("reuse", [False, True])
def test_pad_qwen3_input_reuses_only_private_storage(monkeypatch, reuse):
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_PYPTO_QWEN3_REUSE_INPUT_BUFFERS", lambda: reuse)
    first = qwen3_runtime.pad_qwen3_input(torch.ones(1, 3), None, 4)
    second = qwen3_runtime.pad_qwen3_input(torch.full((1, 3), 2.0), first, 4)
    assert (first.data_ptr() == second.data_ptr()) is reuse
    assert torch.equal(second[0], torch.full((3,), 2.0))
    assert torch.equal(second[1:], torch.zeros(3, 3))


def test_qwen3_callables_share_arena_cache_sizing():
    config = SimpleNamespace(runtime_env=SimpleNamespace())
    qwen3_runtime.configure_qwen3_call_config(config)
    assert config.runtime_env.ring_task_window == 1024
    assert config.runtime_env.ring_heap == 32 * 1024 * 1024
    assert config.runtime_env.ring_dep_pool == 32768


def test_shared_worker_reuses_runtime_per_device(monkeypatch):
    instances = []

    class FakeWorker:
        def __init__(self):
            self.inits = []
            self.finalized = False
            instances.append(self)

        def init(self, *, device_id, bins):
            self.inits.append((device_id, bins))

        def finalize(self):
            self.finalized = True

    class FakeBuilder:
        def __init__(self, *, platform):
            assert platform == "a2a3"

        def get_binaries(self, runtime, *, build):
            assert runtime == "tensormap_and_ringbuffer"
            assert build is False
            return SimpleNamespace()

    monkeypatch.setattr(qwen3_runtime, "ChipWorker", FakeWorker)
    monkeypatch.setattr(qwen3_runtime, "RuntimeBuilder", FakeBuilder)
    monkeypatch.setattr(qwen3_runtime, "_workers", {})

    worker_a, lock_a = qwen3_runtime.get_shared_worker(5)
    worker_b, lock_b = qwen3_runtime.get_shared_worker(5)
    worker_c, _ = qwen3_runtime.get_shared_worker(6)

    assert worker_a is worker_b
    assert lock_a is lock_b
    assert worker_c is not worker_a
    assert len(instances) == 2
    assert len(worker_a.inits) == len(worker_c.inits) == 1

    qwen3_runtime._finalize_workers()
    assert all(worker.finalized for worker in instances)


@pytest.mark.parametrize("debug, expected_level", [(False, logging.WARNING), (True, logging.DEBUG)])
def test_simpler_log_level_is_configured_before_worker_init(monkeypatch, debug, expected_level):
    previous_level = logging.getLogger("simpler").level

    class FakeWorker:
        def init(self, *, device_id, bins):
            assert logging.getLogger("simpler").level == expected_level

        def finalize(self):
            pass

    class FakeBuilder:
        def __init__(self, *, platform):
            pass

        def get_binaries(self, runtime, *, build):
            return SimpleNamespace()

    monkeypatch.setattr(qwen3_runtime, "ChipWorker", FakeWorker)
    monkeypatch.setattr(qwen3_runtime, "RuntimeBuilder", FakeBuilder)
    monkeypatch.setattr(qwen3_runtime, "_workers", {})
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_PYPTO_QWEN3_DEBUG", lambda: debug)
    try:
        qwen3_runtime.get_shared_worker(5)
        qwen3_runtime._finalize_workers()
    finally:
        logging.getLogger("simpler").setLevel(previous_level)
