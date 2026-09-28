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

"""Guards for the Qwen3-14B PyPTO paged-attention shadow patch."""

from types import SimpleNamespace

import torch

from vllm_ascend.patch.worker import patch_qwen3_pypto_attention as patch


def _backend() -> SimpleNamespace:
    return SimpleNamespace(
        num_heads=40,
        num_kv_heads=8,
        head_size=128,
        sliding_window=None,
        key_cache=torch.zeros(4, 128, 8, 128, dtype=torch.bfloat16),
        value_cache=torch.zeros(4, 128, 8, 128, dtype=torch.bfloat16),
    )


def test_shadow_returns_native_output_and_passes_device_metadata(monkeypatch):
    backend = _backend()
    query = torch.zeros(2, 40, 128, dtype=torch.bfloat16)
    native = torch.ones_like(query)
    metadata = SimpleNamespace(
        block_tables=torch.tensor([[2], [3]], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
    )
    seen = {}

    class FakeExecutor:
        def run(self, query, key_cache, value_cache, block_tables, seq_lens):
            seen["query"] = query
            seen["key_cache"] = key_cache
            seen["value_cache"] = value_cache
            seen["block_tables"] = block_tables
            seen["seq_lens"] = seq_lens
            return native.clone()

    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD_PAGED_ATTENTION",
        lambda *args: native,
    )
    monkeypatch.setattr(patch, "_get_executor", lambda device_id: FakeExecutor())
    monkeypatch.setattr(patch, "_DISPATCH_COUNT", 0)
    monkeypatch.setattr(patch, "_MAX_ABS", 0.0)
    monkeypatch.setattr(patch, "_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_REF_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_NUMEL", 0)

    result = patch._forward_paged_attention_shadow(
        backend,
        query,
        metadata,
        torch.empty_like(query),
    )

    assert result is native
    assert seen["query"] is query
    assert seen["key_cache"] is backend.key_cache
    assert seen["value_cache"] is backend.value_cache
    assert seen["block_tables"] is metadata.block_tables
    assert seen["seq_lens"] is metadata.seq_lens
    assert patch._DISPATCH_COUNT == 1
    assert patch._MAX_ABS == 0.0


def test_non_qwen_attention_remains_native(monkeypatch):
    backend = _backend()
    backend.num_heads = 32
    query = torch.zeros(2, 32, 128, dtype=torch.bfloat16)
    native = torch.ones_like(query)
    metadata = SimpleNamespace()
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD_PAGED_ATTENTION",
        lambda *args: native,
    )
    monkeypatch.setattr(
        patch,
        "_get_executor",
        lambda device_id: (_ for _ in ()).throw(AssertionError("unexpected dispatch")),
    )

    result = patch._forward_paged_attention_shadow(
        backend,
        query,
        metadata,
        torch.empty_like(query),
    )

    assert result is native


def test_fused_decode_shadow_returns_native_and_dispatches(monkeypatch):
    backend = _backend()
    query = torch.zeros(2, 40, 128, dtype=torch.bfloat16)
    key = torch.zeros(2, 8, 128, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    native = torch.ones_like(query)
    metadata = SimpleNamespace(
        block_tables=torch.tensor([[2], [3]], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        attn_state=patch.AscendAttentionState.DecodeOnly,
    )
    seen = {}

    class FakeExecutor:
        def run(self, query, key_cache, value_cache, block_tables, seq_lens):
            seen["query"] = query
            seen["block_tables"] = block_tables
            seen["seq_lens"] = seq_lens
            return native.clone()

    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD_FUSED_ATTENTION",
        lambda *args: native,
    )
    monkeypatch.setattr(patch, "_get_executor", lambda device_id: FakeExecutor())
    monkeypatch.setattr(patch, "_DISPATCH_COUNT", 0)
    monkeypatch.setattr(patch, "_MAX_ABS", 0.0)
    monkeypatch.setattr(patch, "_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_REF_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_NUMEL", 0)

    result = patch._forward_fused_attention_shadow(
        backend,
        query,
        key,
        value,
        metadata,
        torch.empty_like(query),
    )

    assert result is native
    assert seen["query"] is query
    assert seen["block_tables"] is metadata.block_tables
    assert seen["seq_lens"] is metadata.seq_lens
    assert patch._DISPATCH_COUNT == 1


def test_cpu_attention_reference_rounding_variants():
    query = torch.zeros(2, 40, 128, dtype=torch.bfloat16)
    key_cache = torch.zeros(4, 128, 8, 128, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    value_cache[2].fill_(2.0)
    value_cache[3].fill_(3.0)
    block_tables = torch.tensor([[2], [3]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)

    references = patch._cpu_attention_references(query, key_cache, value_cache, block_tables, seq_lens)

    expected = torch.empty_like(query)
    expected[0].fill_(2.0)
    expected[1].fill_(3.0)
    assert set(references) == {"fp32", "bf16_exp_before_sum", "normalized_bf16"}
    for reference in references.values():
        torch.testing.assert_close(reference, expected, rtol=0.0, atol=0.0)


def test_replace_returns_pypto_output(monkeypatch):
    backend = _backend()
    query = torch.zeros(2, 40, 128, dtype=torch.bfloat16)
    native = torch.ones_like(query)
    pypto = torch.full_like(query, 2.0)
    metadata = SimpleNamespace(
        block_tables=torch.tensor([[2], [3]], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
    )

    class FakeExecutor:
        def run(self, query, key_cache, value_cache, block_tables, seq_lens):
            return pypto

    monkeypatch.setattr(
        patch,
        "_ORIGINAL_FORWARD_PAGED_ATTENTION",
        lambda *args: native,
    )
    monkeypatch.setattr(patch, "_get_executor", lambda device_id: FakeExecutor())
    monkeypatch.setattr(patch, "_MODE", "replace")
    monkeypatch.setattr(patch, "_DISPATCH_COUNT", 0)
    monkeypatch.setattr(patch, "_MAX_ABS", 0.0)
    monkeypatch.setattr(patch, "_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_REF_SUM_SQUARE", 0.0)
    monkeypatch.setattr(patch, "_NUMEL", 0)

    result = patch._forward_paged_attention_shadow(
        backend,
        query,
        metadata,
        torch.empty_like(query),
    )

    assert result is pypto
    assert patch._DISPATCH_COUNT == 1
    assert patch._MAX_ABS == 1.0
