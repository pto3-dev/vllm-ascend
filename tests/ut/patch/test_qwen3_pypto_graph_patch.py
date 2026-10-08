# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch contracts for the original 40-layer graph, without NPU execution."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.patch.worker import patch_qwen3_pypto_graph as patch
from vllm_ascend.pypto.qwen3_graph import Qwen3OriginalGraphBridge


def context(decode=1, actual=1, mode=CUDAGraphMode.FULL):
    metadata = SimpleNamespace(num_decode_tokens=decode, num_actual_tokens=actual)
    return SimpleNamespace(attn_metadata={"layer0": metadata}, cudagraph_runtime_mode=mode)


def test_owned_decode_skips_native_attention_update(monkeypatch):
    original = Mock()
    monkeypatch.setattr(patch, "_BRIDGE", object())
    monkeypatch.setattr(patch, "_ATTENTION_UPDATE", original)
    patch._attention_update(object(), context(), 1)
    original.assert_not_called()


@pytest.mark.parametrize("ctx", [context(decode=0), context(mode=CUDAGraphMode.NONE), context(2, 2)])
def test_other_attention_updates_are_preserved(monkeypatch, ctx):
    original = Mock(return_value="native")
    monkeypatch.setattr(patch, "_BRIDGE", object())
    monkeypatch.setattr(patch, "_ATTENTION_UPDATE", original)
    assert patch._attention_update("stream", ctx, ctx.attn_metadata["layer0"].num_actual_tokens) == "native"
    original.assert_called_once()


def test_prefill_keeps_original_model_forward(monkeypatch):
    model = object()
    original = Mock(return_value="prefill")
    monkeypatch.setattr(patch, "_BRIDGE", SimpleNamespace(model=model))
    monkeypatch.setattr(patch, "_MODEL_FORWARD", original)
    monkeypatch.setattr(patch, "get_forward_context", lambda: context(0, 3, CUDAGraphMode.NONE))
    assert patch._model_forward(model, "ids", "positions", inputs_embeds="embeds") == "prefill"
    original.assert_called_once_with(model, "ids", "positions", None, "embeds")


def test_decode_bypasses_python_layer_loop(monkeypatch):
    model = object()
    bridge = SimpleNamespace(model=model, decode=Mock(return_value="decode"))
    original = Mock()
    monkeypatch.setattr(patch, "_BRIDGE", bridge)
    monkeypatch.setattr(patch, "_MODEL_FORWARD", original)
    monkeypatch.setattr(patch, "get_forward_context", lambda: context())
    assert patch._model_forward(model, "ids", "positions", inputs_embeds="embeds") == "decode"
    bridge.decode.assert_called_once()
    original.assert_not_called()


def test_prepare_precedes_capture_and_graph_is_tracked(monkeypatch):
    order = []
    ctx = context()
    ctx.batch_descriptor = "batch1"
    graph = object()
    wrapper = SimpleNamespace(runtime_mode=CUDAGraphMode.FULL, concrete_aclgraph_entries={})

    def captured(self, *args, **kwargs):
        order.append("capture")
        self.concrete_aclgraph_entries[ctx.batch_descriptor] = SimpleNamespace(aclgraph=graph)
        return "result"

    bridge = SimpleNamespace(
        prepare_graph=lambda metadata: order.append("prepare"),
        track_graph=lambda owner: order.append("track"),
    )
    monkeypatch.setattr(patch, "_BRIDGE", bridge)
    monkeypatch.setattr(patch, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(patch, "_GRAPH_CALL", captured)
    assert patch._graph_call(wrapper) == "result"
    assert order == ["prepare", "capture", "track"]


@pytest.mark.parametrize("pinned", [False, True])
def test_eager_uses_normal_run_and_graph_uses_enqueue(monkeypatch, pinned):
    bridge = object.__new__(Qwen3OriginalGraphBridge)
    for name in ("hidden", "output", "seq", "slot", "table", "key", "value"):
        setattr(bridge, name, MagicMock())
    bridge.arguments = object()
    bridge.handle, bridge.call_config = object(), object()
    bridge.token = object() if pinned else None
    bridge.device_id = 0
    bridge.caches = [(MagicMock(), MagicMock())]
    bridge.worker = Mock()
    bridge.model = SimpleNamespace(norm=Mock(return_value="normalized"))
    metadata = MagicMock()
    metadata.pypto_seq_lens_device.device.type = "npu"
    stream = Mock(npu_stream=123)
    monkeypatch.setattr(torch.npu, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: pinned)
    assert bridge.decode(SimpleNamespace(shape=(1, 5120)), metadata) == "normalized"
    if pinned:
        bridge.worker.enqueue_graph_run.assert_called_once_with(bridge.token, 123)
        bridge.worker.run.assert_not_called()
        stream.synchronize.assert_not_called()
    else:
        bridge.worker.run.assert_called_once_with(bridge.handle, bridge.arguments, bridge.call_config)
        bridge.worker.enqueue_graph_run.assert_not_called()
        stream.synchronize.assert_called_once()
