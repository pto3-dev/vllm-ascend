# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Imported allocation aliases retain canonical ownership and strict bounds."""

import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

from pypto.runtime.runtime_base import Worker as RuntimeBase
from simpler.buffer import wrap_device_malloc

from vllm_ascend.pypto.qwen38.borrowed_npu_storage import BorrowedBufferView, bind_imported_tensor


class BorrowContracts(unittest.TestCase):
    def make_runtime(self):
        worker = NS(_child_prov_lock=threading.Lock(), _child_alloc={})
        worker._record_device_alloc = lambda handle: worker._child_alloc.setdefault(handle.identity, replace(handle))
        return NS(_w=worker, _device_buffers={}, _require_ready=lambda label: None)

    def records(self):
        tables = []
        for rank in (0, 1):
            item = dict(
                pointer=4096 + rank * 8192 + 512,
                storage_base=4096 + rank * 8192,
                storage_nbytes=8192,
                offset=512,
                shape=[1, 256],
                dtype="bfloat16",
                ipc_identity="a" * 64,
                owner_instance_id=(rank + 1).to_bytes(8, "little").hex(),
                worker_id=rank,
            )
            tables.append({"cache": item, "alias": dict(item)})
        return tables

    def test_same_storage_alias_reuses_registered_handle(self):
        runtime = self.make_runtime()
        tables = self.records()
        first = bind_imported_tensor(runtime, tables, "cache")
        second = bind_imported_tensor(runtime, tables, "alias")
        for rank in (0, 1):
            self.assertIs(first.shards[rank].buffer, second.shards[rank].buffer)
            self.assertEqual(len(runtime._w._child_alloc), 2)
            RuntimeBase._require_owned_resident_tensor(runtime, second.shards[rank], "alias", worker_id=rank)

    def test_wrong_target_worker_is_rejected(self):
        tables = self.records()
        tables[1]["cache"]["worker_id"] = 0
        with self.assertRaisesRegex(ValueError, "different target worker"):
            bind_imported_tensor(self.make_runtime(), tables, "cache")

    def test_expired_resident_registration_is_rejected(self):
        runtime = self.make_runtime()
        tensor = bind_imported_tensor(runtime, self.records(), "cache").shards[0]
        runtime._device_buffers.clear()
        with self.assertRaisesRegex(ValueError, "not a live allocation"):
            RuntimeBase._require_owned_resident_tensor(runtime, tensor, "expired", worker_id=0)

    def test_pointer_collision_with_another_storage_is_rejected(self):
        runtime = self.make_runtime()
        tables = self.records()
        bind_imported_tensor(runtime, tables, "cache")
        tables[0]["alias"]["ipc_identity"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different live runtime allocation"):
            bind_imported_tensor(runtime, tables, "alias")

    def test_native_storage_view_bounds_are_enforced(self):
        backing = wrap_device_malloc(4096, 512, (1).to_bytes(8, "little"), 7)
        for offset, nbytes in [(-1, 1), (512, 1), (16, 513)]:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                BorrowedBufferView(backing, offset, nbytes)


if __name__ == "__main__":
    unittest.main()
