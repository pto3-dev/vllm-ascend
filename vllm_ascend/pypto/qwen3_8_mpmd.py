# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ordinary-text full TP2/PP1 PTO MPMD with one native cache allocation.

vLLM owns scheduling, cache allocation, input preparation and sampling. A single
CPU L3 service submits both complete model ranks to two Simpler L2 workers. The
service imports native allocator handles; it never mirrors cache contents.
"""

import atexit
import os
import time
from multiprocessing.connection import Client
from types import MethodType

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

from vllm_ascend.pypto.qwen3_8_cache import full_attention_views, gdn_state_views


class Qwen38MPMDExecutor:
    def __init__(self, causal, config, settings):
        parallel = config.parallel_config
        if settings.get("native_reference", False):
            raise ValueError("Native-reference diagnostics are not a PTO integration mode.")
        if parallel.tensor_parallel_size != 2 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Full PTO MPMD requires TP2/PP1.")
        if not config.model_config.enforce_eager and not settings.get("graph", False):
            raise ValueError(
                "The current cross-rank L3 service requires enforce_eager; MPMD ACLGraph replay is not enabled."
            )
        text = config.model_config.hf_text_config
        if (text.hidden_size, text.num_hidden_layers, text.vocab_size) != (5120, 64, 248320):
            raise ValueError("Full PTO MPMD supports the dense Qwen3.8-27B text checkpoint.")
        if config.speculative_config or config.lora_config or config.model_config.quantization:
            raise ValueError("Speculative, LoRA and quantized MPMD execution are unsupported.")
        self.causal = causal
        self.settings = settings
        self.rank = get_tensor_model_parallel_rank()
        self.max_seq = config.model_config.max_model_len
        if self.max_seq not in (512, 2304):
            raise ValueError("Full MPMD supports max_model_len=512 or2304.")
        self.connection = None
        self.exported = {}
        self.cache_views = {}
        self.topology = None
        self.generation = None
        self.native_forward = causal.model.forward
        self.native_logits = causal.compute_logits
        self.shared_weights = None
        self.shared_weight_info = None
        self.native_decode_calls = 0
        self.native_prefill_calls = 0
        self.sequence = 0
        self.profile_calls = 0
        self.model_calls = 0
        self.hidden = None
        self.logits = None
        self.last_real = False
        self.records = []
        self.profile_next = None
        self.profile_controls = None
        self.validation_position = None
        self.captured_positions = set()
        causal.model.do_not_compile = True
        atexit.register(self.close)

    def _prepare_shared_weights(self):
        if self.shared_weights is not None:
            return
        from models.qwen3_8_27b.model_spec import ModelSpec

        from vllm_ascend.pypto.qwen3_8_shared_weights import bank_native_weights

        start = time.perf_counter()
        spec = ModelSpec(tokens=1, fuse_mlp=True, native_weight_order=True)
        self.shared_weights, self.shared_weight_info = bank_native_weights(self.causal, spec)
        self.shared_weight_info["initialization_ms"] = (time.perf_counter() - start) * 1000

    def _ensure_outputs(self, device):
        if self.hidden is None:
            self.hidden = torch.zeros((self.max_seq, 5120), dtype=torch.bfloat16, device=device)
            self.logits = torch.zeros((1, 248320), dtype=torch.float32, device=device)

    def _connect(self, context):
        from torch_npu.multiprocessing.reductions import _npu_reduce_tensor

        if self.shared_weights is None:
            raise RuntimeError("Shared ND weights must be prepared before MPMD startup.")
        cache_rows = block_size = slots = None
        manifest = {}
        for index, layer in enumerate(self.causal.model.layers):
            if index % 4 == 3:
                attn = layer.self_attn
                native = attn.attn.kv_cache[context.virtual_engine]
                key, value = full_attention_views(native, 2)
                size = native[0].shape[1]
                dims = (key.shape[0], size)
                if cache_rows is not None and dims != (cache_rows, block_size):
                    raise ValueError("Full-attention cache capacities differ by layer.")
                cache_rows, block_size = dims
                self.cache_views.update({f"l{index}_keys": key, f"l{index}_values": value})
            else:
                conv, state = gdn_state_views(layer.linear_attn.kv_cache[context.virtual_engine], 8, 24)
                if slots is not None and slots != conv.shape[0]:
                    raise ValueError("GDN cache capacities differ by layer.")
                slots = conv.shape[0]
                self.cache_views.update({f"l{index}_conv_state": conv, f"l{index}_recurrent_state": state})
        if None in (cache_rows, block_size, slots):
            raise ValueError("Native cache allocation is incomplete.")
        self.table_cols = (self.max_seq + block_size - 1) // block_size
        if block_size % 128:
            raise ValueError("PTO attention requires native block size divisible by 128.")
        # Submit each shared native allocation once. Simpler forbids two
        # writable overlapping arguments in one task; typed aliases are built
        # inside the chip orchestration and use each layer's native metadata.
        storage_names = {}
        pools = []
        layout = []
        for name, value in self.cache_views.items():
            storage = value.untyped_storage()
            identity = storage.data_ptr()
            pool = storage_names.get(identity)
            if pool is None:
                pool = f"cache_pool{len(pools)}"
                storage_names[identity] = pool
                raw = torch.empty(0, dtype=torch.bfloat16, device=value.device).set_(
                    storage, 0, (storage.nbytes() // 2,), (1,)
                )
                self.exported[pool] = raw
                pools.append((pool, storage.nbytes()))
            offset = value.data_ptr() - storage.data_ptr()
            dtype = "FP32" if value.dtype == torch.float32 else "BF16"
            layout.append((name, pool, offset, list(value.shape), dtype))
        rotary = self.causal.model.layers[3].self_attn.rotary_emb.cos_sin_cache[: self.max_seq]
        self.exported.update(self.shared_weights)
        self.exported.update(
            cos=rotary[:, :32].to(dtype=torch.bfloat16).contiguous(),
            sin=rotary[:, 32:].to(dtype=torch.bfloat16).contiguous(),
        )
        self.exported.update(global_logits=self.logits, hidden_result=self.hidden)
        torch.npu.synchronize()
        for name, value in self.exported.items():
            if not value.is_contiguous():
                raise ValueError(f"Native borrowed tensor {name} must be contiguous.")
            _, args = _npu_reduce_tensor(value)
            manifest[name] = args
        self.connection = Client(self.settings["socket"], family="AF_UNIX")
        self.connection.send(
            dict(
                rank=self.rank,
                owner_pid=os.getpid(),
                manifest=manifest,
                dimensions=dict(
                    max_seq=self.max_seq,
                    cache_rows=cache_rows,
                    cache_block_size=block_size,
                    state_slots=slots,
                    native_cache_pools=pools,
                    native_cache_layout=layout,
                ),
            )
        )
        reply = self._receive(1800)
        self.topology = reply["topology"]
        self.generation = reply["generation"]

    def _receive(self, timeout=180):
        if not self.connection.poll(timeout):
            raise TimeoutError("PTO MPMD service did not complete the two-rank request.")
        reply = self.connection.recv()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "PTO MPMD service failed."))
        return reply

    def _controls(self, context, ids, positions):
        tokens = ids.numel()
        first_gdn = self.causal.model.layers[0].linear_attn.prefix
        first_full = self.causal.model.layers[3].self_attn.attn.layer_name
        gdn = context.attn_metadata[first_gdn]
        full = context.attn_metadata[first_full]
        if gdn.spec_sequence_masks is not None:
            raise ValueError("Speculative GDN metadata is unsupported.")
        indices = gdn.non_spec_state_indices_tensor
        starts = gdn.non_spec_query_start_loc
        if indices.numel() != 1 or starts.numel() != 2:
            raise ValueError("Full PTO MPMD currently supports one ordinary-text request.")
        # Hybrid cache groups may share the same physical pool while using
        # different block/state indices. Read each unique metadata tensor once.
        copied = {}

        def host_values(value):
            key = (value.data_ptr(), tuple(value.shape), str(value.dtype))
            if key not in copied:
                copied[key] = value.cpu().reshape(-1).tolist()
            return copied[key]

        layer_metadata = []
        for index, layer in enumerate(self.causal.model.layers):
            key = layer.self_attn.attn.layer_name if index % 4 == 3 else layer.linear_attn.prefix
            meta = context.attn_metadata[key]
            if index % 4 == 3:
                if meta.slot_mapping.data_ptr() != full.slot_mapping.data_ptr():
                    raise ValueError("Multiple full-attention slot mappings require separate append controls.")
                seq = getattr(meta, "pypto_seq_lens_device", None)
                if seq is None:
                    seq = meta.seq_lens
                layer_metadata.append(
                    [0, 1, int(host_values(seq)[0])] + host_values(meta.block_tables)[: self.table_cols]
                )
            else:
                if meta.non_spec_state_indices_tensor.numel() != 1:
                    raise ValueError("MPMD requires batch1 GDN metadata.")
                slot = int(host_values(meta.non_spec_state_indices_tensor)[0])
                initial = 1 if meta.has_initial_state is None else int(host_values(meta.has_initial_state)[0])
                layer_metadata.append([slot, initial, 0] + [0] * self.table_cols)
        length = getattr(full, "pypto_seq_lens_device", None)
        if length is None:
            length = full.seq_lens
        controls = dict(
            ids=host_values(ids),
            positions=host_values(positions),
            slots=host_values(full.slot_mapping)[:tokens],
            starts=host_values(starts),
            state_index=layer_metadata[0][0],
            initial=layer_metadata[0][1],
            layer_metadata=layer_metadata,
            table=layer_metadata[3][3:],
            length=layer_metadata[3][2],
        )
        if controls["starts"] != [0, tokens] or controls["state_index"] < 0:
            raise ValueError("Invalid native GDN request boundaries or state slot.")
        if controls["length"] > self.max_seq or min(controls["slots"]) < 0:
            raise ValueError("Invalid native KV length or physical slot mapping.")
        return controls

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        context = get_forward_context()
        self._prepare_shared_weights()
        tokens = positions.shape[-1]
        native = bool(self.settings.get("native_reference", False)) or not context.attn_metadata or tokens != 1
        if context.attn_metadata and not native:
            meta = context.attn_metadata[self.causal.model.layers[0].linear_attn.prefix]
            native = bool(getattr(meta, "num_prefills", 0))
        self.validation_position = None
        if self.settings.get("logits_validation_dir") and context.attn_metadata and tokens == 1:
            meta = context.attn_metadata[self.causal.model.layers[0].linear_attn.prefix]
            if not getattr(meta, "num_prefills", 0):
                position = int(positions.reshape(-1)[0].cpu())
                if position in (300, 338) and position not in self.captured_positions:
                    self.validation_position = position
        if native:
            self.last_real = False
            if not context.attn_metadata:
                self.profile_calls += 1
            elif tokens > 1:
                self.native_prefill_calls += 1
            else:
                self.native_decode_calls += 1
            return self.native_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        self._ensure_outputs(positions.device)
        if input_ids is None or inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("Full PTO MPMD requires ordinary token IDs without embedding overrides or PP tensors.")
        if positions.ndim == 2:
            if (
                positions.shape[0] != 3
                or not torch.equal(positions[0], positions[1])
                or not torch.equal(positions[0], positions[2])
            ):
                raise ValueError("Full PTO MPMD requires identical ordinary-text MRoPE axes.")
            positions = positions[0]
        if self.connection is None:
            self._connect(context)
        control_start = time.perf_counter()
        controls = self._controls(context, input_ids, positions)
        control_end = time.perf_counter()
        torch.npu.synchronize()
        synchronized = time.perf_counter()
        capture = self.profile_next
        # Native's free-block queue rotates physical slots between requests.
        # Keep deps armed until a fresh prefill uses the exact timing controls.
        if capture == "deps" and controls != self.profile_controls:
            capture = None
        self.connection.send(dict(command="run", sequence=self.sequence, controls=controls, profile=capture))
        reply = self._receive()
        completed = time.perf_counter()
        self.records.append(
            dict(
                reply,
                control_ms=(control_end - control_start) * 1000,
                synchronize_ms=(synchronized - control_end) * 1000,
                rpc_wait_ms=(completed - synchronized) * 1000,
            )
        )
        if capture == "timing":
            self.profile_controls = controls
        if capture is not None:
            self.profile_next = None
        self.sequence += 1
        self.model_calls += 1
        self.last_real = True
        return self.hidden[: input_ids.numel()]

    def compute_logits(self, causal, hidden_states):
        # Logits are computed, widened and gathered by the PTO model itself.
        logits = self.logits if self.last_real else self.native_logits(hidden_states)
        position = getattr(self, "validation_position", None)
        if position is not None:
            from pathlib import Path

            folder = Path(self.settings["logits_validation_dir"]) / f"rank{self.rank}"
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    position=position,
                    backend="pto" if self.last_real else "native",
                    hidden=hidden_states.detach().float().cpu(),
                    logits=logits.detach().float().cpu(),
                ),
                folder / f"position{position}.pt",
            )
            self.captured_positions.add(position)
            self.validation_position = None
        return logits

    def stats(self):
        return dict(
            pto_executor=not bool(self.settings.get("native_reference", False)),
            full_mpmd_model=not bool(self.settings.get("native_reference", False)),
            tp=2,
            pp=1,
            topology=self.topology,
            graph_mode="eager",
            native_weights_offloaded=False,
            native_prefill_calls=self.native_prefill_calls,
            native_decode_calls=self.native_decode_calls,
            shared_weight_info=self.shared_weight_info,
            neural_fallback=False,
            cache_contents_copied=False,
            borrowed_cache_tensors=len(self.cache_views),
            borrowed_cache_allocations=sum(name.startswith("cache_pool") for name in self.exported),
            model_calls=self.model_calls,
            native_profile_calls=self.profile_calls,
            build_generation=self.generation,
            jit_misses=0,
            mpmd_calls=self.records,
        )

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(dict(command="close"))
                self._receive(180)
            except (EOFError, BrokenPipeError, OSError):
                pass
            self.connection.close()
            self.connection = None
        # Keep exports alive until native worker exit; the L2 importer drains
        # before dropping its allocator references. No borrowed storage is freed.


def attach_mpmd(causal, config, settings):
    if not settings.get("graph", False) or settings.get("phase_nz", False):
        raise ValueError("Qwen3.8 PTO integration requires FULL graph mode and shared ND weights.")
    from vllm_ascend.pypto.qwen3_8_lib import ensure_model_root
    from vllm_ascend.pypto.qwen3_8_mpmd_graph import Qwen38MPMDGraphExecutor, install_mpmd_graph_hooks
    from vllm_ascend.pypto.qwen38.l2_owned_native import install_native_weight_adoption

    ensure_model_root()
    install_native_weight_adoption()
    install_mpmd_graph_hooks()
    executor = Qwen38MPMDGraphExecutor(causal, config, settings)
    causal._qwen38_pto_mpmd_executor = executor
    causal.model.forward = MethodType(executor.forward, causal.model)
    causal.compute_logits = MethodType(executor.compute_logits, causal)


def close_mpmd(model):
    """Drain the L3/L2 service while both native cache owners remain alive."""
    causal = getattr(model, "language_model", model)
    active = getattr(causal, "_qwen38_pto_mpmd_executor", None)
    if active is not None:
        active.close()
    return dict(mpmd_closed=active is not None)


def arm_profile(model, mode):
    """Capture one decode after the caller starts a fresh native-prefill request."""
    if mode not in ("timing", "deps"):
        raise ValueError("Invalid profiling mode.")
    causal = getattr(model, "language_model", model)
    executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
    if executor is None or executor.settings.get("native_reference"):
        raise ValueError("Profiling requires the full MPMD executor.")
    if executor.profile_next is not None:
        raise ValueError("A capture is already armed.")
    executor.profile_next = mode
    return dict(profile_armed=mode)


def profile_status(model):
    causal = getattr(model, "language_model", model)
    executor = causal._qwen38_pto_mpmd_executor
    return dict(
        pending=executor.profile_next, last_profile=executor.records[-1].get("profile") if executor.records else None
    )
