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

"""Guards for the experimental Qwen3-14B PyPTO Decode MLP patch."""

from types import SimpleNamespace

import pytest

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.patch.worker import patch_qwen3_pypto_mlp as patch
from vllm_ascend.patch.worker import patch_qwen3_pypto_qkv as qkv_patch


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, False),
        ({}, False),
        ({"layer": SimpleNamespace(attn_state=AscendAttentionState.PrefillNoCache)}, False),
        ({"layer": SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)}, True),
        ({"layer": object()}, False),
    ],
)
def test_pure_decode_guard(monkeypatch, metadata, expected):
    monkeypatch.setattr(
        patch,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    assert patch._pure_decode() is expected


def test_prefill_uses_original_forward(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(patch, "_pure_decode", lambda: False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD",
        lambda *args: calls.append(args) or sentinel,
    )
    layer, positions, hidden, residual = (object() for _ in range(4))
    assert patch._qwen3_decode_forward(layer, positions, hidden, residual) is sentinel
    assert calls == [(layer, positions, hidden, residual)]


def test_decode_routes_both_rmsnorms_to_pypto(monkeypatch):
    calls = []
    input_norm = object()
    post_norm = object()
    hidden = object()
    residual = object()
    attention_output = object()
    normed_input = object()
    normed_post = object()
    mlp_output = object()

    def fake_rms_forward(norm, value, previous_residual, stage):
        calls.append((stage, norm, value, previous_residual))
        if stage == "input":
            return normed_input, residual
        return normed_post, residual

    layer = SimpleNamespace(
        input_layernorm=input_norm,
        post_attention_layernorm=post_norm,
        self_attn=lambda **kwargs: calls.append(("attention", kwargs)) or attention_output,
        mlp=lambda value: calls.append(("mlp", value)) or mlp_output,
    )
    monkeypatch.setattr(patch, "_pure_decode", lambda: True)
    monkeypatch.setattr(patch, "_RMS_MODE", "replace")
    monkeypatch.setattr(patch, "_MODE", "off")
    monkeypatch.setattr(patch, "rms_forward", fake_rms_forward)
    assert patch._qwen3_decode_forward(layer, 7, hidden, None) == (mlp_output, residual)
    assert calls == [
        ("input", input_norm, hidden, None),
        ("attention", {"positions": 7, "hidden_states": normed_input}),
        ("post_attention", post_norm, attention_output, residual),
        ("mlp", normed_post),
    ]


def test_decode_routes_post_rms_and_mlp_to_one_callable(monkeypatch):
    calls = []
    residual = object()
    normed_input = object()
    attention_output = SimpleNamespace(device=SimpleNamespace(index=4))
    mlp_output = object()
    next_residual = object()
    norm_weight = object()
    gate_up_weight = object()
    down_weight = object()

    class FusedExecutor:
        def run(self, *args):
            calls.append(("fused", args))
            return mlp_output, next_residual

    layer = SimpleNamespace(
        input_layernorm=object(),
        post_attention_layernorm=SimpleNamespace(weight=norm_weight),
        self_attn=lambda **kwargs: attention_output,
        mlp=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=gate_up_weight),
            down_proj=SimpleNamespace(weight=down_weight),
        ),
    )
    monkeypatch.setattr(patch, "_pure_decode", lambda: True)
    monkeypatch.setattr(patch, "_RMS_MODE", "replace")
    monkeypatch.setattr(patch, "_MODE", "replace")
    monkeypatch.setattr(patch, "_FUSED_POST_RMS_MLP", True)
    monkeypatch.setattr(patch, "rms_forward", lambda *args: (normed_input, residual))
    monkeypatch.setattr(patch, "_get_fused_executor", lambda device_id: FusedExecutor())

    result = patch._qwen3_decode_forward(layer, 7, object(), None)
    assert result == (mlp_output, next_residual)
    assert calls == [("fused", (attention_output, residual, norm_weight, gate_up_weight, down_weight))]


def test_decode_routes_input_rms_and_qkv_to_one_callable(monkeypatch):
    calls = []
    hidden = SimpleNamespace(device=SimpleNamespace(index=4))
    qkv = object()
    residual = object()
    attention_output = object()
    normed_post = object()
    mlp_output = object()
    input_weight = object()
    qkv_weight = object()

    class InputExecutor:
        def run(self, *args):
            calls.append(("input_fused", args))
            return qkv, residual

    attention = SimpleNamespace(qkv_proj=SimpleNamespace(weight=qkv_weight, bias=None))
    layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=input_weight),
        post_attention_layernorm=object(),
        self_attn=attention,
        mlp=lambda value: mlp_output,
    )
    monkeypatch.setattr(patch, "_pure_decode", lambda: True)
    monkeypatch.setattr(patch, "_RMS_MODE", "replace")
    monkeypatch.setattr(patch, "_MODE", "off")
    monkeypatch.setattr(patch, "_FUSED_INPUT_RMS_QKV", True)
    monkeypatch.setattr(patch, "_FUSED_POST_RMS_MLP", False)
    monkeypatch.setattr(patch, "_get_fused_input_executor", lambda device_id: InputExecutor())
    monkeypatch.setattr(
        qkv_patch,
        "run_qwen3_attention_from_qkv",
        lambda self_attn, positions, projected: (
            calls.append(("attention", self_attn, positions, projected)) or attention_output
        ),
    )
    monkeypatch.setattr(patch, "rms_forward", lambda *args: (normed_post, residual))

    result = patch._qwen3_decode_forward(layer, 7, hidden, None)
    assert result == (mlp_output, residual)
    assert calls == [
        ("input_fused", (hidden, None, input_weight, qkv_weight)),
        ("attention", attention, 7, qkv),
    ]
