# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Borrow caller-owned NPU allocations in the model's two Simpler L2 workers.

The CPU-only L3 helper scopes a bootstrap adapter to its own process. It never
initializes Torch-NPU on the L3. Each L2 imports native allocator IPC handles
before initializing its ChipWorker, retaining imports until its task loop ends.
The descriptor names the imported mapping and does not transfer tensor contents.
"""

import contextlib
import hashlib
import inspect


class BorrowedBufferView:
    """Expose a tensor subrange while preserving its backing's canonical ID."""

    def __init__(self, backing, offset, nbytes):
        self.backing = backing
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.base = int(backing.base) + self.offset
        self.owner_worker_id = backing.owner_worker_id
        if self.offset < 0 or self.offset + self.nbytes > backing.nbytes:
            raise ValueError("Borrowed view exceeds its IPC storage.")

    def tensor(self, shapes, dtype, **kwargs):
        return self.backing.tensor(shapes, dtype, byte_offset=self.offset, **kwargs)

    def to_descriptor(self):
        return self.backing.to_descriptor()


def reconstruct_imports(manifest, owner_instance_id, worker_id, native_connection=None):
    import torch
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    torch.npu.set_device(worker_id)
    tensors = {}
    records = {}
    for name, args in manifest.items():
        value = rebuild_npu_tensor(*args)
        if value.device.type != "npu" or value.device.index != worker_id:
            raise ValueError(f"Borrowed tensor {name} belongs to a different logical NPU.")
        if not value.is_contiguous():
            raise ValueError(f"Borrowed tensor {name} must be a contiguous native view.")
        owned = native_connection is not None and (
            name == "lm_head" or (name.startswith("weights") and name[7:].isdigit())
        )
        if owned:
            from torch_npu.multiprocessing.reductions import _npu_reduce_tensor

            # A bounded one-bank temporary, before capture. The native side
            # rebinds its parameters and releases its previous storage on ack.
            previous = value
            value = torch.empty_like(previous, memory_format=torch.contiguous_format)
            # Use synchronous DMA for this one-time relocation. No native
            # AICore copy task must precede Simpler's controller reservation.
            import ctypes

            acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/lib64/libascendcl.so")
            acl.aclrtMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            acl.aclrtMemcpy.restype = ctypes.c_int
            count = previous.numel() * previous.element_size()
            rc = acl.aclrtMemcpy(value.data_ptr(), count, previous.data_ptr(), count, 3)
            if rc:
                raise RuntimeError(f"Weight ownership DMA failed for {name}: {rc}")
            torch.npu.synchronize()
            args = _npu_reduce_tensor(value)[1]
            native_connection.send(dict(ok=True, command="adopt_l2_weight", name=name, ipc=args))
            if not native_connection.poll(180):
                raise TimeoutError(f"Native weight adoption timed out: {name}")
            reply = native_connection.recv()
            if reply != dict(command="adopted_l2_weight", name=name):
                raise RuntimeError(f"Native weight adoption failed: {reply}")
            previous = None
            import gc

            gc.collect()
            torch.npu.ipc_collect()
            native_connection.send(dict(ok=True, command="old_weight_released", name=name))
            if not native_connection.poll(180) or native_connection.recv() != dict(
                command="old_weight_retired", name=name
            ):
                raise RuntimeError(f"Native old weight retirement failed: {name}")
            torch.npu.synchronize()
        storage = value.untyped_storage()
        tensors[name] = value
        handle = args[7]
        records[name] = dict(
            allocation_owner="L2" if owned else "native",
            owner_allocated_bytes=int(torch.npu.memory_allocated()) if owned else None,
            owner_reserved_bytes=int(torch.npu.memory_reserved()) if owned else None,
            pointer=value.data_ptr(),
            storage_base=storage.data_ptr(),
            storage_nbytes=storage.nbytes(),
            shape=list(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            offset=value.data_ptr() - storage.data_ptr(),
            ipc_identity=hashlib.sha256(bytes(handle) + int(args[9]).to_bytes(8, "little")).hexdigest(),
            allocation_offset_bytes=int(args[9]),
            owner_instance_id=owner_instance_id.hex(),
            worker_id=worker_id,
        )
    return tensors, records


@contextlib.contextmanager
def import_in_chip_bootstrap(manifests, reply_connections, graph_connections=None, native_connections=None):
    import simpler.worker as worker_module
    import torch

    if torch.npu.is_initialized():
        raise RuntimeError("The L3 helper must remain NPU-uninitialized before forking L2 workers.")
    original = worker_module._chip_process_loop
    signature = inspect.signature(original)
    required = {"device_id", "owner_instance_id"}
    if not required.issubset(signature.parameters):
        raise RuntimeError("The pinned Simpler chip bootstrap ABI has changed.")

    def bootstrap(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        rank = int(bound.arguments["device_id"])
        try:
            retained, records = reconstruct_imports(
                manifests[rank],
                bound.arguments["owner_instance_id"],
                rank,
                None if native_connections is None else native_connections[rank],
            )
            reply_connections[rank].send(dict(ok=True, records=records))
        except Exception as exc:
            reply_connections[rank].send(dict(ok=False, error=f"{type(exc).__name__}: {exc}"))
            raise
        # IPC refcounts must retire while Torch's NPU context is still alive.
        # ChipWorker.finalize tears that context down after its task loop drains.
        # This hook is scoped to this forked importer child, not the installed SDK.
        original_finalize = worker_module.ChipWorker.finalize
        released = False

        def release_imports():
            nonlocal released
            if released:
                return
            import gc

            torch.npu.synchronize()
            retained.clear()
            gc.collect()
            torch.npu.ipc_collect()
            released = True

        def finalize_with_import_release(chip):
            release_imports()
            return original_finalize(chip)

        worker_module.ChipWorker.finalize = finalize_with_import_release
        try:
            scope = contextlib.nullcontext()
            if graph_connections is not None:
                from vllm_ascend.pypto.qwen38.mpmd_graph_session import install_in_importer

                scope = install_in_importer(
                    retained,
                    graph_connections[rank],
                    rank,
                    owned_weight_names=tuple(
                        name for name, item in records.items() if item["allocation_owner"] == "L2"
                    ),
                )
            with scope:
                return original(*args, **kwargs)
        finally:
            worker_module.ChipWorker.finalize = original_finalize
            if not released:
                release_imports()

    worker_module._chip_process_loop = bootstrap
    try:
        yield
    finally:
        worker_module._chip_process_loop = original


def bind_imported_tensor(runtime, records, name):
    """Create address-free wire views; ownership/free remains with native caller."""
    import torch
    from pypto.runtime import DeviceTensor, StackedDeviceTensor
    from simpler.buffer import wrap_device_malloc

    shards = []
    canonical = getattr(runtime, "_borrowed_backings", None)
    if canonical is None:
        canonical = runtime._borrowed_backings = {}
    for rank, table in enumerate(records):
        item = table[name]
        if item["worker_id"] != rank:
            raise ValueError("Imported view names a different target worker.")
        key = (rank, item["ipc_identity"], item["storage_base"])
        backing = canonical.get(key)
        if backing is None:
            # IDs are scoped to this binding adapter and separate from ordinary
            # runtime allocations. Imported aliases are never passed to free().
            identity = int(item["ipc_identity"][:12], 16) | (1 << 63)
            backing = wrap_device_malloc(
                item["storage_base"],
                item["storage_nbytes"],
                bytes.fromhex(item["owner_instance_id"]),
                identity,
                owner_worker_id=rank,
            )
            canonical[key] = backing
            with runtime._w._child_prov_lock:
                if backing.identity in runtime._w._child_alloc:
                    raise ValueError("Imported storage identity collides with a live allocation.")
                runtime._w._record_device_alloc(backing)
        view = BorrowedBufferView(backing, item["offset"], item["storage_nbytes"] - item["offset"])
        dtype = getattr(torch, item["dtype"])
        key = (rank, item["pointer"])
        existing = runtime._device_buffers.get(key)
        if existing is not None:
            if (
                not isinstance(existing, BorrowedBufferView)
                or existing.backing.identity != backing.identity
                or existing.offset != view.offset
            ):
                raise ValueError("Borrowed pointer collides with a different live runtime allocation.")
            view = existing
        else:
            runtime._device_buffers[key] = view
        tensor = DeviceTensor(item["pointer"], item["shape"], dtype, buffer=view)
        shards.append(tensor)
    return StackedDeviceTensor(shards, (len(shards), *shards[0].shape), tuple(range(len(shards))))
