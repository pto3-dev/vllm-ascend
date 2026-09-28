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
