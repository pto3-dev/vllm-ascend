# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in original PyPTO 40-layer Decode inside vLLM's FULL Decode graph."""

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3 import Qwen3Model

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.pypto.qwen3_graph import Qwen3OriginalGraphBridge

_BRIDGE = None
_MODEL_FORWARD = Qwen3Model.forward
_GRAPH_CALL = ACLGraphWrapper.__call__
_ATTENTION_UPDATE = AscendAttentionBackendImpl.update_graph_params


def _metadata(context=None):
    context = get_forward_context() if context is None else context
    metadata = context.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    value = next(iter(metadata.values()))
    if isinstance(value, list):
        raise ValueError("Original PyPTO graph does not support microbatch metadata")
    return value


def _is_single_decode(metadata) -> bool:
    return metadata is not None and metadata.num_decode_tokens == metadata.num_actual_tokens == 1


def _model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    global _BRIDGE
    if intermediate_tensors is not None:
        raise ValueError("Original PyPTO graph requires PP=1")
    if _BRIDGE is None:
        if torch_npu_capturing():
            raise RuntimeError("Original PyPTO bridge must initialize during eager profiling")
        _BRIDGE = Qwen3OriginalGraphBridge(self)
    elif _BRIDGE.model is not self:
        raise RuntimeError("One original PyPTO graph model is supported per worker")
    metadata = _metadata()
    if _is_single_decode(metadata):
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        return _BRIDGE.decode(hidden, metadata)
    return _MODEL_FORWARD(self, input_ids, positions, intermediate_tensors, inputs_embeds)


def torch_npu_capturing() -> bool:
    import torch

    return torch.npu.is_current_stream_capturing()


def _graph_call(self, *args, **kwargs):
    context = get_forward_context()
    owns_decode = context.cudagraph_runtime_mode == self.runtime_mode == CUDAGraphMode.FULL
    if owns_decode:
        metadata = _metadata()
        if not _is_single_decode(metadata):
            raise ValueError("Original PyPTO graph requires FULL_DECODE_ONLY and capture sizes [1]")
        if _BRIDGE is None:
            raise RuntimeError("Original PyPTO bridge was not initialized before graph capture")
        _BRIDGE.prepare_graph(metadata)
    result = _GRAPH_CALL(self, *args, **kwargs)
    if owns_decode:
        entry = self.concrete_aclgraph_entries[context.batch_descriptor]
        if entry.aclgraph is not None:
            _BRIDGE.track_graph(entry.aclgraph)
    return result


def _attention_update(update_stream, forward_context, num_tokens, *args, **kwargs):
    metadata = _metadata(forward_context)
    if (
        _BRIDGE is not None
        and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        and _is_single_decode(metadata)
        and num_tokens == 1
    ):
        # No native Attention task groups were captured for this Decode graph.
        # Its metadata is read by captured copies into PyPTO's fixed buffers.
        return
    return _ATTENTION_UPDATE(update_stream, forward_context, num_tokens, *args, **kwargs)


Qwen3Model.forward = _model_forward
ACLGraphWrapper.__call__ = _graph_call
AscendAttentionBackendImpl.update_graph_params = staticmethod(_attention_update)
