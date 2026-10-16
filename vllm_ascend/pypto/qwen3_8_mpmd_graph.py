# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Full64-layer MPMD graphs with native-owned weights and hybrid cache pools."""

import gc
import time
import weakref

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

from vllm_ascend.pypto.qwen3_8_mpmd import Qwen38MPMDExecutor


class Qwen38MPMDGraphExecutor(Qwen38MPMDExecutor):
    def __init__(self, causal, config, settings):
        super().__init__(causal, config, settings)
        self.bridge = None
        self.graph_replays = 0
        self.graph_captures = 0
        self.caller_graphs = []
        self.last_service_reply = None
        self.control_i64 = None
        self.control_i32 = None
        self.native_metadata = None
        self.validation_observer = None
        self.input_prep_records = []
        self.source_signatures = set()
        self.native_graph_warmup_calls = 0

    def _receive(self, timeout=180):
        reply = super()._receive(timeout)
        self.last_service_reply = reply
        return reply

    def prepare_graph(self, context, device):
        if self.bridge is not None:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MPMD graph preparation must precede capture.")
        self._prepare_shared_weights()
        self._ensure_outputs(device)
        cache = self.causal.model.layers[3].self_attn.attn.kv_cache[context.virtual_engine]
        cols = (self.max_seq + cache[0].shape[1] - 1) // cache[0].shape[1]
        self.control_i64 = torch.zeros(3, dtype=torch.int64, device=device)
        self.control_i32 = torch.tensor(
            [self.rank, 1, 0, 0, 0, 1, 0, 1] + [0] * cols + [1], dtype=torch.int32, device=device
        )
        self.native_metadata = torch.zeros((64, 3 + cols), dtype=torch.int32, device=device)
        self.native_metadata[:, 1].fill_(1)
        self.native_metadata[3::4, 2].fill_(1)
        self.graph_done = torch.zeros(1, dtype=torch.int64, device=device)
        self.exported.update(
            control_i64=self.control_i64,
            control_i32=self.control_i32,
            native_metadata=self.native_metadata,
            graph_done=self.graph_done,
        )
        self._connect(context)
        reply = self.last_service_reply
        if not reply.get("graph_prepared") or reply["initial_epoch"] < 1:
            raise RuntimeError("L3 did not prepare both full-model graphs.")
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        from vllm_ascend.pypto.qwen38.mpmd_graph_session import NativeCapturedBridge

        request = rebuild_npu_tensor(*reply["graph_request_ipc"])
        self.bridge = NativeCapturedBridge(request, self.graph_done, self.control_i32[1:2])
        self.last_service_reply = None
        self.capture_stream = torch.npu.SyncLaunchStream(device=device)
        self.zero_row = torch.zeros(3 + cols, dtype=torch.int32, device=device)
        self.one = torch.ones(1, dtype=torch.int32, device=device)

    def update_inputs(self, context, inputs):
        begin = time.perf_counter()
        source_signature = []
        ids = inputs.get("input_ids")
        positions = inputs["positions"]
        source_signature.extend([ids.data_ptr() if ids is not None else None, positions.data_ptr()])
        if positions.ndim == 2:
            positions = positions[0]
        if ids is None or ids.numel() != 1 or inputs.get("inputs_embeds") is not None:
            raise ValueError("Full MPMD graph requires one text decode token.")
        self.control_i64[:1].copy_(ids.reshape(-1))
        self.control_i64[1:2].copy_(positions.reshape(-1))
        rows = []
        indices = []
        unique = {}
        first_full = None
        cols = self.table_cols
        for index, layer in enumerate(self.causal.model.layers):
            full = index % 4 == 3
            key = layer.self_attn.attn.layer_name if full else layer.linear_attn.prefix
            meta = context.attn_metadata[key]
            if full:
                if first_full is None:
                    first_full = meta
                seq = getattr(meta, "pypto_seq_lens_device", None)
                if seq is None or seq.device.type != "npu":
                    raise ValueError("Graph metadata requires device sequence lengths.")
                values = (seq, meta.block_tables)
            else:
                if meta.spec_sequence_masks is not None:
                    raise ValueError("Speculative GDN metadata is unsupported.")
                values = (meta.non_spec_state_indices_tensor, meta.has_initial_state)
            source_signature.extend([value.data_ptr() for value in (values if full else values[:1])])
            identity = (full,) + tuple(
                None if value is None else (value.data_ptr(), tuple(value.shape), value.dtype) for value in values
            )
            if identity not in unique:
                row = self.zero_row.clone()
                if full:
                    row[1:2].copy_(self.one)
                    row[2:3].copy_(seq.reshape(-1)[:1].clamp_min(1))
                    row[3:].copy_(meta.block_tables.reshape(-1)[:cols])
                else:
                    row[:1].copy_(values[0].reshape(-1)[:1].clamp_min(0))
                    row[1:2].copy_(self.one if values[1] is None else values[1].reshape(-1)[:1])
                unique[identity] = len(rows)
                rows.append(row)
            indices.append(unique[identity])
        if getattr(self, "layer_group_indices", None) != indices:
            self.layer_group_indices = indices
            self.layer_group_index_tensor = torch.tensor(indices, dtype=torch.int64, device=ids.device)
        self.native_metadata.copy_(torch.stack(rows).index_select(0, self.layer_group_index_tensor))
        self.control_i64[2:3].copy_(first_full.slot_mapping.reshape(-1)[:1].clamp_min(0))
        self.control_i32[6:8].copy_(self.native_metadata[0, :2])
        self.control_i32[8 : 8 + cols].copy_(self.native_metadata[3, 3:])
        self.control_i32[8 + cols : 9 + cols].copy_(self.native_metadata[3, 2:3])
        source_signature.append(first_full.slot_mapping.data_ptr())
        self.source_signatures.add(tuple(source_signature))
        self.input_prep_records.append(dict(cpu_ms=(time.perf_counter() - begin) * 1000, metadata_groups=len(rows)))
        if self.validation_observer is not None:
            self.validation_observer(ids, positions)

    def input_signature(self, context, inputs):
        tensors = [inputs["input_ids"], inputs["positions"]]
        for index, layer in enumerate(self.causal.model.layers):
            if index % 4 == 3:
                meta = context.attn_metadata[layer.self_attn.attn.layer_name]
                tensors.extend([meta.pypto_seq_lens_device, meta.block_tables, meta.slot_mapping])
            else:
                meta = context.attn_metadata[layer.linear_attn.prefix]
                tensors.extend([meta.non_spec_state_indices_tensor, meta.has_initial_state])
        return tuple(
            None if value is None else (value.data_ptr(), tuple(value.shape), str(value.dtype)) for value in tensors
        )

    def validate_replay_inputs(self, context, inputs):
        signature = self.input_signature(context, inputs)
        if signature != self.capture_signature:
            raise ValueError("Native graph input addresses or shapes changed; recapture is required.")
        if self.validation_observer is not None:
            positions = inputs["positions"]
            positions = positions[0] if positions.ndim == 2 else positions
            self.validation_observer(inputs["input_ids"], positions)

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            self._prepare_shared_weights()
            self.last_real = False
            if not context.attn_metadata:
                self.profile_calls += 1
            elif positions.shape[-1] > 1:
                self.native_prefill_calls += 1
            elif self.bridge is None:
                self.native_graph_warmup_calls += 1
            else:
                raise RuntimeError("Full MPMD graph decode may not fall back to native neural execution.")
            return self.native_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        if self.bridge is None:
            raise RuntimeError("ACLGraph hook must prepare MPMD before capture.")
        self.update_inputs(context, dict(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds))
        self.bridge.enqueue()
        self.graph_captures += 1
        self.last_real = True
        return self.hidden[:1]

    def compute_logits(self, causal, hidden_states):
        value = super().compute_logits(causal, hidden_states)
        return value

    def stats(self):
        value = super().stats()
        value.update(
            graph_mode="FULL",
            graph_replays=self.graph_replays,
            graph_captures=self.graph_captures,
            model_calls=self.graph_replays,
            external_rpc_per_decode=False,
            graph_prepared=self.bridge is not None,
            metadata_packing_captured=True,
            native_graph_warmup_calls=self.native_graph_warmup_calls,
            metadata_source_signatures=len(self.source_signatures),
            input_prep=self.input_prep_records,
        )
        return value

    @torch.inference_mode()
    def close(self):
        if self.connection is None:
            return
        self.connection.send(dict(command="close"))
        reply = self._receive(180)
        if not reply.get("stop_armed"):
            raise RuntimeError("L3 did not arm graph shutdown.")
        self.control_i32[3].fill_(1)
        torch.npu.synchronize()
        self.bridge.epoch = None
        self.bridge.enqueue()
        torch.npu.synchronize()
        reply = self._receive(180)
        if not reply.get("release_peer"):
            raise RuntimeError("L3 did not drain the stop graph.")
        for reference in self.caller_graphs:
            graph = reference()
            if graph is not None:
                graph.reset()
        self.caller_graphs.clear()
        self.bridge = None
        self.last_service_reply = None
        gc.collect()
        torch.npu.ipc_collect()
        self.connection.send(dict(command="peer_released"))
        reply = self._receive(180)
        if not reply.get("closed"):
            raise RuntimeError("L3 graph teardown did not finish.")
        self.connection.close()
        self.connection = None


def install_mpmd_graph_hooks():
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker import model_runner_v1

    if getattr(ACLGraphWrapper, "_qwen38_mpmd_graph_hook", False):
        return
    original_call = ACLGraphWrapper.__call__
    original_update = model_runner_v1.update_full_graph_params
    original_model_forward = model_runner_v1.NPUModelRunner._model_forward

    def model_forward(runner, *args, **kwargs):
        causal = getattr(runner.model, "language_model", runner.model)
        executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
        if isinstance(executor, Qwen38MPMDGraphExecutor):
            executor.prefill_complete = runner.num_discarded_requests == 0
        return original_model_forward(runner, *args, **kwargs)

    model_runner_v1.NPUModelRunner._model_forward = model_forward

    def graph_call(wrapper, *args, **kwargs):
        causal = getattr(wrapper.runnable, "language_model", wrapper.runnable)
        executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
        if not isinstance(executor, Qwen38MPMDGraphExecutor):
            return original_call(wrapper, *args, **kwargs)
        context = get_forward_context()
        active = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        executor.last_real = active
        context.qwen38_pto_mpmd_graph = active
        if active:
            executor.prepare_graph(context, kwargs["positions"].device)
            wrapper.runnable._qwen38_mpmd_capture_stream = executor.capture_stream
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            if entry is not None and entry.aclgraph is not None:
                executor.validate_replay_inputs(context, kwargs)
                executor.graph_replays += 1
            else:
                executor.update_inputs(context, kwargs)
                executor.capture_signature = executor.input_signature(context, kwargs)
        result = original_call(wrapper, *args, **kwargs)
        if active:
            graph = wrapper.concrete_aclgraph_entries[context.batch_descriptor].aclgraph
            if not any(reference() is graph for reference in executor.caller_graphs):
                executor.caller_graphs.append(weakref.ref(graph))
        return result

    def update_params(*args, **kwargs):
        context = args[2] if len(args) > 2 else kwargs["forward_context"]
        if getattr(context, "qwen38_pto_mpmd_graph", False):
            return
        return original_update(*args, **kwargs)

    ACLGraphWrapper.__call__ = graph_call
    ACLGraphWrapper._qwen38_mpmd_graph_hook = True
    model_runner_v1.update_full_graph_params = update_params
