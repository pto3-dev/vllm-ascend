# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Drain both PTO graph workers before the native HTTP engine closes."""

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    from vllm.entrypoints.openai import api_server

    original = api_server.build_async_engine_client

    @asynccontextmanager
    async def observed_client(*args, **kwargs):
        async with original(*args, **kwargs) as engine:
            options.state_dir.mkdir(parents=True, exist_ok=True)
            control_path = options.state_dir / f"http-control-{os.getpid()}.sock"
            closed = False
            lock = asyncio.Lock()

            async def collect_and_close():
                nonlocal closed
                async with lock:
                    if closed:
                        return
                    stats = await engine.collective_rpc("pto_mpmd_stats", timeout=180)
                    (options.state_dir / "worker_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
                    result = await engine.collective_rpc("pto_mpmd_close", timeout=300)
                    (options.state_dir / "worker_close.json").write_text(json.dumps(result, indent=2) + "\n")
                    closed = True

            async def control(reader, writer):
                try:
                    message = json.loads(await reader.readline())
                    if message != {"command": "collect-and-close"}:
                        raise ValueError("Unsupported lifecycle command")
                    await collect_and_close()
                    reply = {"ok": True, "mpmd_closed": closed}
                except Exception as error:
                    reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    (options.state_dir / "lifecycle_error.json").write_text(json.dumps(reply, indent=2) + "\n")
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(control, path=str(control_path))
            os.chmod(control_path, 0o600)
            (options.state_dir / "lifecycle_control.json").write_text(
                json.dumps({"socket": str(control_path)}, indent=2) + "\n"
            )
            try:
                yield engine
            finally:
                try:
                    await collect_and_close()
                finally:
                    server.close()
                    await server.wait_closed()
                    control_path.unlink(missing_ok=True)

    api_server.build_async_engine_client = observed_client
    from vllm.entrypoints.cli.main import main as cli_main

    try:
        cli_main()
    finally:
        api_server.build_async_engine_client = original


if __name__ == "__main__":
    main()
