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

"""Guards for the opt-in Qwen3-14B PyPTO QKV projection patch."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker import patch_qwen3_pypto_qkv as patch


def test_fused_pa_prefers_device_seq_lens_only_when_enabled(monkeypatch):
    host = torch.tensor([129], dtype=torch.int32)
    device = torch.tensor([129], dtype=torch.int32)
    metadata = SimpleNamespace(seq_lens=host, pypto_seq_lens_device=device)
    monkeypatch.setitem(patch.envs.env_variables, "VLLM_ASCEND_PYPTO_QWEN3_DEVICE_SEQ_LENS", lambda: False)
    assert patch._fused_pa_seq_lens(metadata) is host
    monkeypatch.setitem(patch.envs.env_variables, "VLLM_ASCEND_PYPTO_QWEN3_DEVICE_SEQ_LENS", lambda: True)
    assert patch._fused_pa_seq_lens(metadata) is device


def test_prefill_uses_original_attention(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(patch, "_pure_decode", lambda: False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD",
        lambda *args: calls.append(args) or sentinel,
    )
    layer, positions, hidden = (object() for _ in range(3))
    assert patch._qwen3_attention_forward(layer, positions, hidden) is sentinel
    assert calls == [(layer, positions, hidden)]


def test_unsupported_qkv_weight_is_rejected(monkeypatch):
    monkeypatch.setattr(patch, "_pure_decode", lambda: True)
    layer = SimpleNamespace(
        q_size=5120,
        kv_size=1024,
        qkv_proj=SimpleNamespace(weight=torch.empty(1, 5120, dtype=torch.bfloat16)),
    )
    hidden = torch.empty(1, 5120, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="bias-free ND weights"):
        patch._qwen3_attention_forward(layer, torch.zeros(1), hidden)


def test_rope_shadow_dispatches_before_inplace_native_rope(monkeypatch):
    class IdentityNorm:
        def __call__(self, value):
            return value

    class InplaceRotary:
        is_neox_style = True
        rotary_dim = 128

        def _match_cos_sin_cache_dtype(self, value):
            return torch.zeros(2, 128, dtype=value.dtype)

        def __call__(self, positions, q, k):
            q.add_(1)
            k.add_(1)
            return q, k

    seen = {}

    class FakeExecutor:
        def run(self, q, k, cos, sin):
            seen["q"] = q.clone()
            seen["k"] = k.clone()
            return q.clone(), k.clone()

    monkeypatch.setattr(patch, "_QK_ROPE_MODE", "off")
    monkeypatch.setattr(patch, "_ROPE_MODE", "shadow")
    monkeypatch.setattr(patch, "_ROPE_DISPATCH_COUNT", 0)
    monkeypatch.setattr(patch, "_get_rope_executor", lambda device_id: FakeExecutor())

    layer = SimpleNamespace(
        head_dim=128,
        q_norm=IdentityNorm(),
        k_norm=IdentityNorm(),
        rotary_emb=InplaceRotary(),
    )
    q = torch.zeros(1, 5120, dtype=torch.bfloat16)
    k = torch.zeros(1, 1024, dtype=torch.bfloat16)
    native_q, native_k = patch._run_rope_path(layer, torch.tensor([0]), q, k)

    assert torch.count_nonzero(seen["q"]) == 0
    assert torch.count_nonzero(seen["k"]) == 0
    assert torch.all(native_q == 1)
    assert torch.all(native_k == 1)


def test_fused_attention_uses_layer_metadata_and_cache(monkeypatch):
    batch = 2
    qkv = torch.zeros(batch, 7168, dtype=torch.bfloat16)
    key_cache = torch.zeros(4, 128, 8, 128, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    metadata = SimpleNamespace(
        attn_state=patch.AscendAttentionState.DecodeOnly,
        block_tables=torch.tensor([[2, 1], [3, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([129, 7], dtype=torch.int32),
        slot_mapping=torch.tensor([257, 391], dtype=torch.int64),
    )
    seen = {}

    class FakeRotary:
        is_neox_style = True
        rotary_dim = 128

        def _match_cos_sin_cache_dtype(self, value):
            return torch.zeros(384, 128, dtype=value.dtype)

    class FakeExecutor:
        def run(self, *args):
            seen["args"] = args
            return torch.ones(batch, 40, 128, dtype=torch.bfloat16)

    layer = SimpleNamespace(
        attn=SimpleNamespace(
            layer_name="model.layers.3.self_attn.attn",
            impl=SimpleNamespace(
                key_cache=key_cache,
                value_cache=value_cache,
                sliding_window=None,
            ),
        ),
        rotary_emb=FakeRotary(),
        q_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16)),
        k_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16)),
    )
    monkeypatch.setattr(
        patch,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"model.layers.3.self_attn.attn": metadata}),
    )
    monkeypatch.setattr(
        patch,
        "_get_fused_pa_executor",
        lambda device_id: FakeExecutor(),
    )

    result = patch._run_fused_pa(layer, qkv)

    assert tuple(result.shape) == (batch, 40, 128)
    assert seen["args"][0] is qkv
    assert seen["args"][1] is key_cache
    assert seen["args"][2] is value_cache
    assert seen["args"][3] is metadata.block_tables
    assert seen["args"][4] is metadata.seq_lens
    assert seen["args"][5] is metadata.slot_mapping
    assert seen["args"][6].dtype == torch.float32
    assert seen["args"][8].shape == (1, 128)


def test_fused_attention_rejects_uninitialized_cache(monkeypatch):
    layer = SimpleNamespace(
        attn=SimpleNamespace(
            layer_name="layer",
            impl=SimpleNamespace(
                key_cache=None,
                value_cache=None,
                sliding_window=None,
            ),
        )
    )
    metadata = SimpleNamespace(attn_state=patch.AscendAttentionState.DecodeOnly)
    monkeypatch.setattr(
        patch,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )

    with pytest.raises(RuntimeError, match="initialized pure-Decode"):
        patch._run_fused_pa(
            layer,
            torch.zeros(1, 7168, dtype=torch.bfloat16),
        )
