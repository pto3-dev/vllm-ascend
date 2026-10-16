# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Orderly HTTP shutdown drains PTO while the native consumers are alive."""

import asyncio
import builtins
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

from vllm_ascend.pypto.qwen38 import http_server


def test_http_shutdown_drains_before_native_engine_exit(monkeypatch, tmp_path):
    events = []

    class Engine:
        async def collective_rpc(self, name, timeout):
            events.append(name)
            return [{"ok": True}, {"ok": True}]

    @asynccontextmanager
    async def original(*args, **kwargs):
        events.append("native-enter")
        yield Engine()
        events.append("native-exit")

    server = SimpleNamespace(build_async_engine_client=original)

    async def serve():
        async with server.build_async_engine_client():
            events.append("serve")

    def cli():
        asyncio.run(serve())

    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "vllm.entrypoints.openai":
            return SimpleNamespace(api_server=server)
        if name == "vllm.entrypoints.cli.main":
            return SimpleNamespace(main=cli)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(sys, "argv", ["http_server", "--state-dir", str(tmp_path), "serve", "model"])
    http_server.main()
    assert events == ["native-enter", "serve", "pto_mpmd_stats", "pto_mpmd_close", "native-exit"]
    assert server.build_async_engine_client is original
    assert json.loads((tmp_path / "worker_close.json").read_text()) == [{"ok": True}, {"ok": True}]
    assert not list(tmp_path.glob("*.sock"))
