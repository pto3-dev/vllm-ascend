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

from types import SimpleNamespace

from vllm_ascend.pypto import qwen3_runtime


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
