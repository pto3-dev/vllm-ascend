# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental graph adapter for PyPTO-Lib's original 40-layer Decode.

This deliberately preserves decode_fwd_layers' arithmetic, including FP32
inter-layer carry. Graph/eager parity does NOT establish native vLLM parity.
The initial integration is TP=PP=1, BF16, single-sequence Decode only.
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm_ascend.pypto.qwen3_graph_config import get_graph_paths

if TYPE_CHECKING:
    from pypto.ir.compiled_program import CompiledProgram
    from simpler.task_interface import ChipTensor
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Model

    from vllm_ascend.attention.attention_v1 import AscendMetadata

LOG = logging.getLogger(__name__)
LAYERS, HIDDEN, INTERMEDIATE, KV_HIDDEN, HEAD_DIM, BATCH_PAD = 40, 5120, 17408, 1024, 128, 16
KV_BLOCK_SIZE, KV_HEADS = 128, 8
PLATFORM = "a2a3"
RUNTIME_NAME = "tensormap_and_ringbuffer"
RING_TASK_WINDOW = 1024
RING_HEAP_BYTES = 32 * 1024 * 1024
RING_DEP_POOL = 32768


def compile_original_decode40(root: str, build_dir: str, device_id: int) -> CompiledProgram:
    """Compile the original body, without allocating dummy model weights."""
    from pypto.runtime import ExecutionMode, RunConfig

    model_dir = Path(root).resolve() / "models" / "qwen3_14b"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    name = "_vllm_pypto_original_decode40"
    spec = importlib.util.spec_from_file_location(name, model_dir / "decode_fwd.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import PyPTO-Lib decode_fwd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._CHUNK_NLAYERS = LAYERS
    shapes = (
        (BATCH_PAD, HIDDEN),
        (LAYERS, HIDDEN),
        (LAYERS * HIDDEN, HIDDEN),
        (LAYERS * HIDDEN, KV_HIDDEN),
        (LAYERS * HIDDEN, KV_HIDDEN),
        (LAYERS, HEAD_DIM),
        (LAYERS, HEAD_DIM),
        (1,),
        (1,),
        (1,),
        (128, HEAD_DIM),
        (128, HEAD_DIM),
        (LAYERS * KV_BLOCK_SIZE * KV_HEADS, HEAD_DIM),
        (LAYERS * KV_BLOCK_SIZE * KV_HEADS, HEAD_DIM),
        (LAYERS * HIDDEN, HIDDEN),
        (LAYERS * HIDDEN, INTERMEDIATE),
        (LAYERS * HIDDEN, INTERMEDIATE),
        (LAYERS * INTERMEDIATE, HIDDEN),
        (LAYERS, HIDDEN),
        (BATCH_PAD, HIDDEN),
    )
    fp32 = {1, 5, 6, 10, 11, 18}
    int32 = {7, 8, 9}
    samples = [
        torch.empty(
            shape, device="meta", dtype=(torch.float32 if i in fp32 else torch.int32 if i in int32 else torch.bfloat16)
        )
        for i, shape in enumerate(shapes)
    ]
    return module.decode_fwd_layers.compile(
        *samples,
        config=RunConfig(
            execution_mode=ExecutionMode.ONBOARD,
            platform=PLATFORM,
            device_id=device_id,
            save_kernels=True,
            save_kernels_dir=build_dir,
        ),
    )


class Qwen3OriginalGraphBridge:
    """Pin one original 40-layer callable for vLLM-owned graph capture.

    Native weights are retained on CPU, and staged one layer at a time only for
    native Prefill. This avoids a second 26 GB device copy. It is an opt-in
    functional prototype, not a fair Prefill performance comparison.
    """

    def __init__(self, model: Qwen3Model) -> None:
        from simpler.task_interface import CallConfig, ChipWorker
        from simpler_setup.runtime_builder import RuntimeBuilder

        self.model = model
        self.device_id = torch.npu.current_device()
        self.device = torch.device(f"npu:{self.device_id}")
        self.token = None
        self.arguments = None
        self.graphs: list[weakref.ReferenceType] = []
        self.closed = False
        config = model.config
        expected = {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
        }
        for name, value in expected.items():
            if getattr(config, name, None) != value:
                raise ValueError(f"Original PyPTO Decode requires {name}={value}")
        if getattr(config, "rms_norm_eps", None) != 1e-6:
            raise ValueError("Original PyPTO Decode requires rms_norm_eps=1e-6")
        if len(model.layers) != LAYERS or model.quant_config is not None:
            raise ValueError("Original PyPTO Decode requires TP=PP=1 and unquantized BF16")
        self._validate_weights()
        root, build_dir = get_graph_paths()
        compiled = compile_original_decode40(root, build_dir, self.device_id)
        LOG.info("Compiled original PyPTO 40-layer Decode callable")
        self._pack_weights()
        cache = model.layers[0].self_attn.rotary_emb.cos_sin_cache.float()
        cos, sin = cache.chunk(2, dim=-1)
        self.cos = torch.cat((cos, cos), dim=-1).contiguous()
        self.sin = torch.cat((sin, sin), dim=-1).contiguous()
        self.hidden = torch.empty((BATCH_PAD, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self.output = torch.empty_like(self.hidden)
        self.seq = torch.ones(1, dtype=torch.int32, device=self.device)
        self.slot = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.worker = ChipWorker()
        self.worker.init(
            device_id=self.device_id,
            bins=RuntimeBuilder(platform=PLATFORM).get_binaries(RUNTIME_NAME, build=False),
        )
        self.handle = self.worker.register_callable(compiled.chip_callable)
        self.call_config = CallConfig()
        self.call_config.runtime_env.ring_task_window = RING_TASK_WINDOW
        self.call_config.runtime_env.ring_heap = RING_HEAP_BYTES
        self.call_config.runtime_env.ring_dep_pool = RING_DEP_POOL
        atexit.register(self.close)

    def _validate_weights(self) -> None:
        for layer in self.model.layers:
            for tensor, shape in (
                (layer.self_attn.qkv_proj.weight, (HIDDEN + 2 * KV_HIDDEN, HIDDEN)),
                (layer.self_attn.o_proj.weight, (HIDDEN, HIDDEN)),
                (layer.mlp.gate_up_proj.weight, (2 * INTERMEDIATE, HIDDEN)),
                (layer.mlp.down_proj.weight, (HIDDEN, INTERMEDIATE)),
            ):
                if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                    raise ValueError("Original Decode requires contiguous ND BF16, TP=1 weights")

    @staticmethod
    def _projection_modules(layer: Qwen3DecoderLayer) -> tuple[torch.nn.Module, ...]:
        return layer.self_attn.qkv_proj, layer.self_attn.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj

    def _stage_native_projection(self, module: torch.nn.Module, weight: torch.Tensor) -> None:
        """Retain native Prefill behavior with one temporary projection weight."""
        original = module.forward

        def native_projection(*args, **kwargs):
            try:
                module.weight.data = weight.to(self.device)
                return original(*args, **kwargs)
            finally:
                module.weight.data = weight

        module.forward = native_projection

    def _pack_weights(self) -> None:
        # Preserve the native parameters on CPU before freeing their NPU storage.
        # Prefill's original layer forward remains in use, with temporary copies.
        self.host_weights = []
        for layer in self.model.layers:
            modules = self._projection_modules(layer)
            weights = tuple(module.weight.detach().cpu() for module in modules)
            for module, weight in zip(modules, weights):
                module.weight.data = weight
            self.host_weights.append(weights)
        torch.npu.synchronize(self.device_id)
        torch.npu.empty_cache()

        def bank(select: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]) -> torch.Tensor:
            cpu = torch.cat([select(weights).contiguous() for weights in self.host_weights], dim=0)
            return cpu.to(self.device).contiguous()

        self.wq = bank(lambda w: w[0][:HIDDEN].t())
        self.wk = bank(lambda w: w[0][HIDDEN : HIDDEN + KV_HIDDEN].t())
        self.wv = bank(lambda w: w[0][HIDDEN + KV_HIDDEN :].t())
        self.wo = bank(lambda w: w[1].t())
        self.gate = bank(lambda w: w[2][:INTERMEDIATE].t())
        self.up = bank(lambda w: w[2][INTERMEDIATE:].t())
        self.down = bank(lambda w: w[3].t())
        self.input_norm = torch.stack([layer.input_layernorm.weight.float() for layer in self.model.layers])
        self.q_norm = torch.stack([layer.self_attn.q_norm.weight.float() for layer in self.model.layers])
        self.k_norm = torch.stack([layer.self_attn.k_norm.weight.float() for layer in self.model.layers])
        self.post_norm = torch.stack([layer.post_attention_layernorm.weight.float() for layer in self.model.layers])
        for layer, weights in zip(self.model.layers, self.host_weights):
            for module, weight in zip(self._projection_modules(layer), weights):
                self._stage_native_projection(module, weight)

    @staticmethod
    def chip_tensor(tensor: torch.Tensor) -> ChipTensor:
        from simpler.task_interface import ChipTensor, DataType

        mapping = {torch.bfloat16: DataType.BFLOAT16, torch.float32: DataType.FLOAT32, torch.int32: DataType.INT32}
        if not tensor.is_contiguous() or tensor.device.type != "npu":
            raise ValueError("Graph arguments must be contiguous device-owned tensors")
        return ChipTensor.make(tensor.data_ptr(), tuple(tensor.shape), mapping[tensor.dtype], child_memory=True)

    def prepare(self, metadata: AscendMetadata) -> None:
        """Allocate fixed argument storage before eager execution or capture."""
        from simpler.task_interface import ChipStorageTaskArgs

        if self.arguments is not None:
            return
        self.caches = [layer.self_attn.attn.kv_cache[0] for layer in self.model.layers]
        if any(not isinstance(cache, (tuple, list)) or len(cache) != 2 for cache in self.caches):
            raise ValueError("Expected separate native K/V cache tensors")
        shape = self.caches[0][0].shape
        if len(shape) != 4 or tuple(shape[1:]) != (KV_BLOCK_SIZE, KV_HEADS, HEAD_DIM):
            raise ValueError(f"Expected native BSND KV cache, got {tuple(shape)}")
        if any(k.shape != shape or v.shape != shape for k, v in self.caches):
            raise ValueError("All 40 native KV pools must have the same shape")
        if any(
            tensor.dtype != torch.bfloat16 or tensor.device != self.device or not tensor.is_contiguous()
            for pair in self.caches
            for tensor in pair
        ):
            raise ValueError("Original Decode requires contiguous BF16 native KV cache on this NPU")
        self.key = torch.empty((LAYERS, *shape), dtype=torch.bfloat16, device=self.device)
        self.value = torch.empty_like(self.key)
        self.table = torch.empty_like(metadata.block_tables[:1].reshape(-1))
        if self.table.dtype != torch.int32:
            raise ValueError("Native block tables must be INT32")
        tensors = (
            self.hidden,
            self.input_norm,
            self.wq,
            self.wk,
            self.wv,
            self.q_norm,
            self.k_norm,
            self.seq,
            self.table,
            self.slot,
            self.cos,
            self.sin,
            self.key.view(-1, HEAD_DIM),
            self.value.view(-1, HEAD_DIM),
            self.wo,
            self.gate,
            self.up,
            self.down,
            self.post_norm,
            self.output,
        )
        arguments = ChipStorageTaskArgs()
        for tensor in tensors:
            arguments.add_tensor(self.chip_tensor(tensor))
        self.arguments = arguments

    def prepare_graph(self, metadata: AscendMetadata) -> None:
        """Pin launch resources outside capture, after ordinary eager warmup."""
        if self.token is not None:
            return
        self.prepare(metadata)
        self.token = self.worker.prepare_graph_run(self.handle, self.arguments, self.call_config)
        LOG.info("Prepared fixed-address PyPTO 40-layer graph resources")

    def decode(self, hidden_states: torch.Tensor, metadata: AscendMetadata) -> torch.Tensor:
        if hidden_states.shape != (1, HIDDEN):
            raise ValueError("Initial original Decode graph supports one sequence only")
        if self.arguments is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Original Decode graph was not prepared before capture")
            self.prepare(metadata)
        self.hidden.zero_()
        self.hidden[:1].copy_(hidden_states)
        seq_lens = metadata.pypto_seq_lens_device
        if seq_lens is None or seq_lens.device.type != "npu":
            raise ValueError("Original Decode graph requires device-side sequence lengths")
        self.seq.copy_(seq_lens[:1])
        self.table.copy_(metadata.block_tables[:1].reshape(-1))
        self.slot.copy_(metadata.slot_mapping[:1])
        for i, (key, value) in enumerate(self.caches):
            self.key[i].copy_(key)
            self.value[i].copy_(value)
        stream = torch.npu.current_stream(self.device_id)
        if self.token is not None:
            self.worker.enqueue_graph_run(self.token, stream.npu_stream)
        else:
            # vLLM profiles and warms up before preparing capture resources.
            # Simpler's private streams must observe Torch's argument copies.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Ordinary worker.run() cannot execute inside capture")
            stream.synchronize()
            self.worker.run(self.handle, self.arguments, self.call_config)
        for i, (key, value) in enumerate(self.caches):
            key.copy_(self.key[i])
            value.copy_(self.value[i])
        return self.model.norm(self.output[:1])

    def track_graph(self, graph: torch.npu.NPUGraph) -> None:
        if not any(ref() is graph for ref in self.graphs):
            self.graphs.append(weakref.ref(graph))

    def close(self) -> None:
        if self.closed:
            return
        # vLLM can replay on a stream other than the original capture stream.
        # Drain all submitted work before destroying graphs or pinned pointers.
        torch.npu.synchronize(self.device_id)
        for ref in self.graphs:
            graph = ref()
            if graph is not None:
                graph.reset()
        if self.token is not None:
            self.worker.finalize_graph_run(self.token)
        self.worker.finalize()
        self.closed = True
