# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Root-owned two-L2 graph session over already validated MPMD arguments.

No ordinary task-frame checks are overridden. One ordinary warmup establishes
canonical storage, bounds, the communication domain and each rank callable.
The session then reserves those same ChipWorkers through the existing graph
API. Native callers use captured stream-value gates over imported NPU storage.
"""

import contextlib
import ctypes
import sys
import threading
import traceback


class StreamGate:
    def __init__(self):
        self.lib = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/lib64/libascendcl.so")
        self.constants = None
        self.lib.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.aclrtMemcpyAsync.restype = ctypes.c_int
        for name in ("aclrtValueWrite", "aclrtValueWait"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
            fn.restype = ctypes.c_int

    def prepare_constants(self, device):
        import torch

        self.constants = torch.tensor([0, 1], dtype=torch.int64, device=device)
        torch.npu.synchronize(device)

    def write(self, tensor, value, stream):
        if self.constants is None:
            raise RuntimeError("Prepare gate constants outside capture.")
        if value not in (0, 1):
            raise ValueError("Stream gates accept0/1 only.")
        rc = self.lib.aclrtMemcpyAsync(
            tensor.data_ptr(), 8, self.constants.data_ptr() + int(value) * 8, 8, 3, int(stream)
        )
        if rc:
            raise RuntimeError(f"gate D2D write failed:{rc}")

    def wait(self, tensor, value, stream):
        rc = self.lib.aclrtValueWait(tensor.data_ptr(), int(value), 0, int(stream))
        if rc:
            raise RuntimeError(f"aclrtValueWait failed:{rc}")


class NativeCapturedBridge:
    def __init__(self, request, done, epoch=None):
        self.request = request
        self.done = done
        self.gate = StreamGate()
        self.gate.prepare_constants(done.device)
        self.enqueues = 0
        self.epoch = epoch

    def enqueue(self):
        import torch

        stream = torch.npu.current_stream(self.request.device).npu_stream
        if self.epoch is not None:
            self.epoch.add_(1)
        self.gate.write(self.done, 0, stream)
        self.gate.write(self.request, 1, stream)
        self.gate.wait(self.done, 1, stream)
        self.enqueues += 1


class _ValidatedArgsProxy:
    def __init__(self, original):
        self.original = original
        self.last = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run_materialized(self, cid, args, config, *rest, **options):
        result = self.original.run_materialized(cid, args, config, *rest, **options)
        self.last = (int(cid), args, config)
        return result

    def _submit_chip_run_materialized(self, cid, args, config, *rest, **options):
        # All normal native validation, admission and generation checks run.
        result = self.original._submit_chip_run_materialized(cid, args, config, *rest, **options)
        self.last = (int(cid), args, config)
        return result


class _ChildGraphSession:
    def __init__(self, worker, proxy, retained, connection, rank, owned_weight_names=()):
        self.worker = worker
        self.proxy = proxy
        self.retained = retained
        self.connection = connection
        self.rank = rank
        self.token = None
        self.graph = None
        self.owned_weight_names = frozenset(owned_weight_names)
        self.stream = None
        self.replays = 0
        self.gate = StreamGate()
        self.closed = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.serve, name="mpmd-graph-control", daemon=True)
        self.thread.start()

    def prepare(self):
        import torch

        if self.graph is not None:
            raise RuntimeError("MPMD graph is already prepared.")
        if self.proxy.last is None:
            raise RuntimeError("A fully completed ordinary MPMD warmup is required.")
        torch.npu.set_device(self.rank)
        cid, args, config = self.proxy.last
        self.arguments = args
        host_args = [
            dict(index=i, shape=list(args.tensor(i).shapes), dtype=str(args.tensor(i).dtype))
            for i in range(args.tensor_count())
            if not args.tensor(i).child_memory
        ]
        if host_args:
            print("GRAPH_STATIC_HOST_ARGS", host_args, flush=True)
        # The SDK's idle Python mailbox poller stays alive while its graph
        # reservation excludes ordinary runs. Give the graph control thread
        # bounded GIL access instead of paying the default5ms quantum.
        self.previous_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0001)
        self.token = self.proxy.original._prepare_graph_run(cid, args, config)
        self.local_request = torch.zeros(1, dtype=torch.int64, device=f"npu:{self.rank}")
        self.gate.prepare_constants(self.local_request.device)
        self.stream = torch.npu.SyncLaunchStream(device=self.rank)
        self.start_event = torch.npu.Event(enable_timing=True)
        self.end_event = torch.npu.Event(enable_timing=True)
        self.start_event.record(self.stream)
        self.end_event.record(self.stream)
        self.stream.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.stream):
            stream = torch.npu.current_stream().npu_stream
            self.gate.wait(self.local_request, 1, stream)
            self.worker.enqueue_graph_run(self.token, stream)
            self.gate.write(self.local_request, 0, stream)
            self.gate.write(self.retained["graph_done"], 1, stream)
        from torch_npu.multiprocessing.reductions import _npu_reduce_tensor

        return dict(captured=True, rank=self.rank, request_ipc=_npu_reduce_tensor(self.local_request)[1])

    def replay(self):
        if self.graph is None:
            raise RuntimeError("Prepare the graph before replay.")
        import torch

        with torch.npu.stream(self.stream):
            self.start_event.record(self.stream)
            self.graph.replay()
            self.end_event.record(self.stream)
        self.stream.synchronize()
        self.replays += 1
        stamp = float(self.start_event.recorded_time())
        valid = stamp != getattr(self, "last_event_stamp", None)
        self.last_event_stamp = stamp
        return dict(
            replays=self.replays,
            rank=self.rank,
            graph_and_gate_ms=float(self.start_event.elapsed_time(self.end_event)),
            event_stamp=stamp,
            event_timestamp_advanced=valid,
        )

    def close(self):
        if self.closed:
            return
        if self.graph is not None:
            self.stream.synchronize()
            self.graph.reset()
            self.graph = None
            self.start_event = None
            self.end_event = None
        if self.token is not None:
            self.worker.finalize_graph_run(self.token)
            self.token = None
        self.arguments = None
        self.closed = True
        if self.owned_weight_names:
            # Native cache/control allocations have the opposite ownership
            # direction. Retire their imports before native owners exit, while
            # keeping original L2 weights alive until native consumers exit.
            import gc

            import torch

            self.proxy.last = None
            for name in tuple(self.retained):
                if name not in self.owned_weight_names:
                    self.retained.pop(name)
            torch.npu.synchronize()
            gc.collect()
            torch.npu.ipc_collect()
        if hasattr(self, "previous_switch_interval"):
            sys.setswitchinterval(self.previous_switch_interval)

    def serve(self):
        import torch

        torch.npu.set_device(self.rank)
        while not self.stop_event.is_set():
            if not self.connection.poll(0.1):
                continue
            command = self.connection.recv()
            try:
                if command == "advance_controls":
                    if self.graph is not None:
                        raise RuntimeError("Ordinary warmup must precede graph reservation.")
                    self.retained["control_i32"][1].add_(1)
                    torch.npu.synchronize()
                    reply = dict(epoch=int(self.retained["control_i32"][1].cpu()))
                elif command == "prepare":
                    reply = self.prepare()
                elif command == "replay":
                    reply = self.replay()
                elif command == "close":
                    self.close()
                    self.connection.send(dict(ok=True, replays=self.replays, rank=self.rank))
                    break
                else:
                    raise ValueError(f"Unknown graph control:{command}")
                self.connection.send(dict(ok=True, **reply))
            except BaseException as error:
                traceback.print_exc()
                self.connection.send(dict(ok=False, error=f"{type(error).__name__}: {error}", rank=self.rank))


@contextlib.contextmanager
def install_in_importer(retained, connection, rank, owned_weight_names=()):
    import simpler.worker as module

    cls = module.ChipWorker
    original_init = cls.init
    original_finalize = cls.finalize
    sessions = {}

    def init(worker, *args, **kwargs):
        result = original_init(worker, *args, **kwargs)
        proxy = _ValidatedArgsProxy(worker._impl)
        worker._impl = proxy
        sessions[id(worker)] = _ChildGraphSession(worker, proxy, retained, connection, rank, owned_weight_names)
        return result

    def finalize(worker):
        session = sessions.pop(id(worker), None)
        if session is not None:
            session.close()
            # Native caller/root must have joined graph-control close first.
            if session.thread.is_alive():
                session.stop_event.set()
                session.thread.join(timeout=1)
                session.connection.close()
                if session.thread.is_alive():
                    raise RuntimeError("Graph control must close before ChipWorker teardown.")
            import gc

            import torch

            session.local_request = None
            session.gate.constants = None
            session.proxy.last = None
            gc.collect()
            torch.npu.ipc_collect()
            worker._impl = session.proxy.original
        return original_finalize(worker)

    cls.init = init
    cls.finalize = finalize
    try:
        yield
    finally:
        cls.init = original_init
        cls.finalize = original_finalize


class MPMDGraphSession:
    """One root coordinator reserves and replays both existing L2 workers."""

    def __init__(self, runtime, connections):
        if runtime._w.level != 3 or len(runtime._w._chip_pids) != 2 or runtime._w._sub_pids:
            raise ValueError("MPMD graph requires one L3/two L2, with no L1.")
        self.runtime = runtime
        self.connections = connections
        self.prepared = False
        self.pending = False
        self.replays = 0
        self.closed = False

    def _replies(self, timeout=180):
        replies = []
        for channel in self.connections:
            if not channel.poll(timeout):
                raise TimeoutError("L2 graph control timed out.")
            reply = channel.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply)
            replies.append(reply)
        return replies

    def prepare(self):
        if self.prepared:
            raise RuntimeError("Already prepared.")
        for channel in self.connections:
            channel.send("prepare")
        replies = self._replies()
        self.prepared = True
        return replies

    def launch(self):
        if not self.prepared or self.pending or self.closed:
            raise RuntimeError("Invalid MPMD graph launch state.")
        for channel in self.connections:
            channel.send("replay")
        self.pending = True

    def wait(self):
        if not self.pending:
            raise RuntimeError("No pending graph replay.")
        replies = self._replies()
        self.pending = False
        self.replays += 1
        return replies

    def close(self):
        if self.closed:
            return
        if self.pending:
            raise RuntimeError("Drain the native-stream gate before graph close.")
        for channel in self.connections:
            channel.send("close")
        self._replies()
        self.closed = True


def promote_graph_ordinals(runtime):
    """Allocate generated communication ordinal state through normal GPU APIs.

    The compiler's per-rank ordinal is a mutable state buffer, not a hot host
    input. A graph session keeps it on-device; all task arguments still pass
    ordinary canonical-import and bounds validation before graph preparation.
    """
    import regex as re
    import torch

    keepalive = getattr(runtime, "_graph_ordinal_host_keepalive", None)
    if keepalive is None:
        keepalive = runtime._graph_ordinal_host_keepalive = []
    promoted = []
    for program, state in runtime._states.items():
        for frame_index, frame in enumerate(state["base_tensor_frames"]):
            for name, values in list(frame.items()):
                if not re.fullmatch(r"__comm_d[0-9]+_ord", name):
                    continue
                if len(values) != 2:
                    raise ValueError("Graph ordinal requires two ranks.")
                for value in values:
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cpu"
                        or value.dtype != torch.int32
                        or tuple(value.shape) != (1,)
                    ):
                        raise ValueError("Unexpected generated communication ordinal.")
                keepalive.extend(values)
                frame[name] = [
                    runtime.alloc_tensor((1,), torch.int32, init=value, worker_id=rank)
                    for rank, value in enumerate(values)
                ]
                promoted.append(dict(name=name, frame=frame_index))
    return promoted
