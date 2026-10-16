# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One L3 owns prepared full-model graphs on its two existing L2 workers."""

import argparse
import json
import multiprocessing
import os
import threading
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import Listener
from pathlib import Path

from pypto.runtime import RunConfig, configure_log

from vllm_ascend.pypto.qwen38.borrowed_npu_storage import bind_imported_tensor, import_in_chip_bootstrap
from vllm_ascend.pypto.qwen38.mpmd_graph_session import MPMDGraphSession, promote_graph_ordinals
from vllm_ascend.pypto.qwen38.mpmd_vllm_server import DTYPES, allocate, compile_programs, orchestration_optimization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--l2-owned-weights", action="store_true")
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-log-level", default="warn")
    parser.add_argument("--diagnostic-ordinary-warmups", type=int, default=0)
    parser.add_argument("--fuse-reduction-norm", action="store_true")
    parser.set_defaults(phase_nz=False, nz_tile_policy=None)
    parser.add_argument("--fuse-gdn-ba", action="store_true")
    parser.add_argument("--fuse-gdn-qkv-z", action="store_true")
    parser.add_argument("--fuse-attention-qkv", action="store_true")
    parser.add_argument("--decode-norm", action="store_true")
    parser.add_argument("--projection-tile-policy", type=Path)
    parser.add_argument("--cache-host-bindings", action="store_true")
    parser.add_argument(
        "--fuse-swiglu-down",
        "--fuse-gate-up-swiglu",
        dest="fuse_swiglu_down",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    configure_log(args.runtime_log_level)
    if os.environ.get("PYPTO_QWEN_WEIGHT_CACHE_BYPASS", "0") == "1" and not args.l2_owned_weights:
        parser.error("Native-owned IPC weights cannot use BYPASS; enable --l2-owned-weights or disable BYPASS.")
    if args.l2_owned_weights and args.phase_nz:
        parser.error(
            "L2-owned shared ND weights currently require phase_nz=False; use the retained NZ branch separately."
        )
    policy = json.loads(args.projection_tile_policy.read_text()) if args.projection_tile_policy else None
    listener = Listener(args.socket, family="AF_UNIX")
    connections = []
    report = dict(
        backend="native prefill plus full64-layer PTO MPMD TP2/PP1 graphs",
        complete=False,
        graph_mode="FULL",
        neural_fallback=False,
        cache_contents_copied=False,
        external_rpc_per_decode=False,
        projection_tile_policy=policy,
        fuse_swiglu_down=args.fuse_swiglu_down,
        fuse_gdn_ba=args.fuse_gdn_ba,
        replays=0,
        graph_and_gate_gpu_samples=[],
    )
    try:
        hellos = {}
        for _ in range(2):
            channel = listener.accept()
            connections.append(channel)
            if not channel.poll(180):
                raise TimeoutError("Native rank manifest did not arrive.")
            hello = channel.recv()
            rank = hello["rank"]
            if rank not in (0, 1) or rank in hellos:
                raise ValueError("Expected native TP ranks0/1.")
            hellos[rank] = (hello, channel)
        native = [hellos[r][0] for r in (0, 1)]
        connections = [hellos[r][1] for r in (0, 1)]
        if native[0]["dimensions"] != native[1]["dimensions"]:
            raise ValueError("Native cache dimensions differ.")
        specs, programs, generation = compile_programs(
            args.build_root,
            native[0]["dimensions"],
            300,
            args.fuse_swiglu_down,
            policy,
            graph_stop_guard=True,
            decode_norm=args.decode_norm,
            phase_nz=args.phase_nz,
            fuse_reduction_norm=args.fuse_reduction_norm,
            nz_tile_policy=json.loads(args.nz_tile_policy.read_text()) if args.nz_tile_policy else None,
            fuse_gdn_ba=args.fuse_gdn_ba,
            fuse_gdn_qkv_z=args.fuse_gdn_qkv_z,
            fuse_attention_qkv=args.fuse_attention_qkv,
        )
        report.update(
            build_generation=str(generation),
            native_owner_pids=[n["owner_pid"] for n in native],
            cache_dimensions=native[0]["dimensions"],
        )
        context = multiprocessing.get_context("spawn")
        receivers = []
        senders = []
        graph_receivers = []
        graph_senders = []
        for _ in range(2):
            a, b = context.Pipe()
            receivers.append(a)
            senders.append(b)
            a, b = context.Pipe()
            graph_receivers.append(a)
            graph_senders.append(b)
        config = RunConfig(platform="a2a3", ring_task_window=16384, ring_heap=128 * 1024 * 1024, ring_dep_pool=65536)
        stop_armed = threading.Event()
        peer_released = threading.Event()
        control_errors = []

        def close_control():
            try:
                frames = []
                for channel in connections:
                    if not channel.poll(1800):
                        raise TimeoutError("Native graph owner did not close.")
                    frames.append(channel.recv())
                if not all(f.get("command") == "close" for f in frames):
                    raise ValueError("Graph session accepts lifecycle control only.")
                stop_armed.set()
                for channel in connections:
                    channel.send(dict(ok=True, stop_armed=True))
                frames = []
                for channel in connections:
                    if not channel.poll(180):
                        raise TimeoutError("Native caller graph did not release peer storage.")
                    frames.append(channel.recv())
                if not all(f.get("command") == "peer_released" for f in frames):
                    raise ValueError("Native caller must release imported request storage.")
                peer_released.set()
            except BaseException as error:
                control_errors.append(error)
                traceback.print_exc()
                stop_armed.set()

        with (
            import_in_chip_bootstrap(
                [n["manifest"] for n in native],
                senders,
                graph_senders,
                native_connections=connections if args.l2_owned_weights else None,
            ),
            orchestration_optimization("O1"),
            programs[0].prepare(config=config, persistent=True, reset_persistent_windows=False) as runtime,
        ):
            imported = [c.recv() for c in receivers]
            if not all(i["ok"] for i in imported):
                raise RuntimeError(imported)
            records = [i["records"] for i in imported]
            report["l2_weight_memory"] = [
                {
                    name: {k: item[k] for k in ("owner_allocated_bytes", "owner_reserved_bytes")}
                    for name, item in r.items()
                    if item["allocation_owner"] == "L2"
                }
                for r in records
            ]
            args.report.write_text(json.dumps(report, indent=2))
            report["weight_allocation_owners"] = [
                {
                    name: item["allocation_owner"]
                    for name, item in r.items()
                    if name.startswith("weights") or name == "lm_head"
                }
                for r in records
            ]
            report["topology"] = dict(
                l3_pid=os.getpid(),
                l3_level=runtime._w.level,
                l2_pids=list(runtime._w._chip_pids),
                sub_pids=list(runtime._w._sub_pids),
            )
            borrowed = {name: bind_imported_tensor(runtime, records, name) for name in native[0]["manifest"]}
            required = ("control_i64", "control_i32", "native_metadata", "graph_done")
            if not all(name in borrowed for name in required):
                raise ValueError("Graph controls must be native-owned GPU tensors.")
            local = dict(borrowed)
            for name, shape, dtype, _ in specs[0].arguments:
                if name not in local:
                    local[name] = allocate(runtime, shape, DTYPES[dtype])
            tensors = [local[name] for name, _, _, _ in specs[0].arguments]
            report["promoted_ordinals"] = promote_graph_ordinals(runtime)
            # This ordinary request validates every argument and establishes
            # the domain before the graph reserves the exact same workers.
            tick = time.perf_counter()
            runtime.run(programs[0], *tensors, config=config)
            report["untimed_dummy_warmup_ms"] = (time.perf_counter() - tick) * 1000
            session = MPMDGraphSession(runtime, graph_receivers)
            report["ordinary_gpu_control_ms"] = []
            initial_epoch = 1
            for _ in range(args.diagnostic_ordinary_warmups):
                for channel in graph_receivers:
                    channel.send("advance_controls")
                epochs = session._replies()
                if epochs[0]["epoch"] != epochs[1]["epoch"]:
                    raise ValueError("Diagnostic warmup epochs differ.")
                initial_epoch = epochs[0]["epoch"]
                tick = time.perf_counter()
                runtime.run(programs[0], *tensors, config=config)
                report["ordinary_gpu_control_ms"].append((time.perf_counter() - tick) * 1000)
            prepared = session.prepare()
            for rank, channel in enumerate(connections):
                channel.send(
                    dict(
                        ok=True,
                        topology=report["topology"],
                        generation=str(generation),
                        graph_prepared=True,
                        initial_epoch=initial_epoch,
                        graph_request_ipc=prepared[rank]["request_ipc"],
                    )
                )
            for reply in prepared:
                reply.pop("request_ipc")
            report["child_graphs"] = prepared
            args.report.write_text(json.dumps(report, indent=2))
            controller = threading.Thread(target=close_control, name="mpmd-graph-lifecycle", daemon=True)
            controller.start()
            while True:
                session.launch()
                completed = session.wait()
                report["replays"] = session.replays
                report["child_replays"] = completed
                report["graph_and_gate_gpu_samples"].append([reply["graph_and_gate_ms"] for reply in completed])
                if stop_armed.is_set():
                    break
                if session.replays % 16 == 0:
                    args.report.write_text(json.dumps(report, indent=2))
            for channel in connections:
                channel.send(dict(ok=True, release_peer=True))
            if not peer_released.wait(180):
                raise TimeoutError("Native owners did not release caller graphs.")
            controller.join(timeout=1)
            if control_errors:
                raise control_errors[0]
            session.close()
            report["stop_replays"] = 1
            report["neural_replays"] = session.replays - 1
            if args.l2_owned_weights:
                # Child graphs/imports are retired, but their original
                # weight allocations must outlive the native consumers.
                for channel in connections:
                    channel.send(dict(ok=True, closed=True))
                deadline = time.monotonic() + 180

                def owner_alive(pid):
                    try:
                        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
                    except FileNotFoundError:
                        return False

                while any(owner_alive(n["owner_pid"]) for n in native):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Native weight consumers did not exit before L2 owners.")
                    time.sleep(0.1)
                report["native_weight_consumers_exited_before_l2_release"] = True
            report["complete"] = True
        # Ordinary native-owned imports keep their original close protocol.
        if not args.l2_owned_weights:
            for channel in connections:
                channel.send(dict(ok=True, closed=True))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        for channel in connections:
            with suppress(OSError, EOFError):
                channel.send(dict(ok=False, error=report["error"]))
        traceback.print_exc()
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        for channel in connections:
            channel.close()
        listener.close()
        Path(args.socket).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
