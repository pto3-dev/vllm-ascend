# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Borrow native Qwen3.8-27B cache storage without constructing a mirror pool."""

import torch


def full_attention_views(cache, kv_heads: int):
    """Return [physical_token, KV_head, 256] views of native BSND storage."""
    if isinstance(cache, torch.Tensor):
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError("Expected native KV shape [2, blocks, block_size, KV_heads, 256].")
        key, value = cache.unbind(0)
    else:
        if len(cache) != 2:
            raise ValueError("Expected a pair of native key/value buffers.")
        key, value = cache
    for tensor in (key, value):
        if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (kv_heads, 256):
            raise ValueError(f"Unsupported native KV shape: {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("PTO Qwen3.8 requires contiguous BF16 BSND caches; NZ/C8 caches are unsupported.")
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("Key and value cache shapes/devices must agree.")
    # view raises for incompatible strides and cannot allocate new storage.
    return key.view(-1, kv_heads, 256), value.view(-1, kv_heads, 256)


def gdn_state_views(cache, key_heads: int, value_heads: int):
    """Return the scheduler-owned conv and recurrent buffers unchanged.

    This pinned Ascend backend addresses the recurrent buffer in K,V order,
    even though the inherited shape calculator labels its last axes V,K.
    Both dimensions are 128. The recurrent PTO kernel selects layout='KV'.
    """
    if len(cache) != 2:
        raise ValueError("Expected native convolution and recurrent state buffers.")
    conv, recurrent = cache
    channels = (2 * key_heads + value_heads) * 128
    if conv.ndim != 3 or tuple(conv.shape[1:]) != (3, channels):
        raise ValueError(f"Unsupported native convolution state: {tuple(conv.shape)}")
    if recurrent.ndim != 4 or tuple(recurrent.shape[1:]) != (value_heads, 128, 128):
        raise ValueError(f"Unsupported native GDN state: {tuple(recurrent.shape)}")
    if conv.dtype != torch.bfloat16 or recurrent.dtype != torch.float32:
        raise ValueError("Expected BF16 convolution state and FP32 recurrent state.")
    if not conv.is_contiguous() or not recurrent.is_contiguous():
        raise ValueError("Native GDN state buffers must be contiguous.")
    if conv.shape[0] != recurrent.shape[0] or conv.device != recurrent.device:
        raise ValueError("GDN state slot counts/devices must agree.")
    return conv, recurrent
