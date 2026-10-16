# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Storage and rejection contracts for scheduler-owned Qwen3.8 caches."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "vllm_ascend/pypto/qwen3_8_cache.py"
spec = importlib.util.spec_from_file_location("qwen38_cache_contract", path)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


@pytest.mark.parametrize("block_size", [128, 1536])
def test_full_attention_storage_alias(block_size):
    cache = torch.zeros(2, 4, block_size, 4, 256, dtype=torch.bfloat16)
    key, value = cache_module.full_attention_views(cache, 4)
    assert key.data_ptr() == cache[0].data_ptr()
    assert value.data_ptr() == cache[1].data_ptr()
    assert key.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    key[block_size + 17, 2, 5] = 3
    assert cache[0, 1, 17, 2, 5] == 3
    assert cache[1, 1, 17, 2, 5] == 0


def test_noncontiguous_cache_is_rejected_without_conversion():
    cache = torch.zeros(2, 4, 128, 4, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        cache_module.full_attention_views(cache.transpose(1, 2), 4)


def test_gdn_state_objects_and_semantic_axes_are_preserved():
    conv = torch.zeros(4, 3, 10240, dtype=torch.bfloat16)
    state = torch.arange(4 * 48 * 128 * 128, dtype=torch.float32).reshape(4, 48, 128, 128)
    actual_conv, actual_state = cache_module.gdn_state_views((conv, state), 16, 48)
    assert actual_conv is conv and actual_state is state
    assert not torch.equal(actual_state, actual_state.transpose(-1, -2))


def test_inconsistent_slot_counts_are_rejected():
    conv = torch.zeros(4, 3, 10240, dtype=torch.bfloat16)
    state = torch.zeros(3, 48, 128, 128)
    with pytest.raises(ValueError, match="slot counts"):
        cache_module.gdn_state_views((conv, state), 16, 48)


def test_non_fp32_recurrent_state_is_rejected():
    conv = torch.zeros(4, 3, 10240, dtype=torch.bfloat16)
    state = torch.zeros(4, 48, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        cache_module.gdn_state_views((conv, state), 16, 48)
