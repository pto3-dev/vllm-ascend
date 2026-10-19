# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One CPU-only L3 dispatches both complete PTO TP ranks over borrowed caches."""

import hashlib
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
from pypto import backend, ir
from pypto.ir import DistributedConfig
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

from vllm_ascend.pypto.qwen3_8_lib import ensure_model_root
from vllm_ascend.pypto.qwen38.cache_packet_adapter import contract_cache_packet

ensure_model_root()
build_model = importlib.import_module("models.qwen3_8_27b.model_program").build_model
ModelSpec = importlib.import_module("models.qwen3_8_27b.model_spec").ModelSpec
_model_runtime = importlib.import_module("models.qwen3_8_27b.runtime")
allocate = _model_runtime.allocate
orchestration_optimization = _model_runtime.orchestration_optimization

NZ = ("5120x17408", "5120x5120", "5120x3072", "5120x6144", "5120x512", "3072x5120", "5120x124160")
DTYPES = dict(BF16=torch.bfloat16, FP32=torch.float32, INT32=torch.int32, INT64=torch.int64, UINT8=torch.uint8)


def compile_programs(
    root,
    dimensions,
    prefill_tokens,
    fuse_swiglu_down=False,
    projection_tile_policy=None,
    graph_stop_guard=False,
    decode_norm=False,
    phase_nz=False,
    fuse_reduction_norm=False,
    nz_tile_policy=None,
    fuse_gdn_ba=False,
    fuse_gdn_qkv_z=False,
    fuse_attention_qkv=False,
):
    if phase_nz or nz_tile_policy:
        raise ValueError("This integration supports shared ND weights only.")
    backend.set_backend_type(backend.BackendType.Ascend910B)
    if projection_tile_policy is None:
        policy_path = ensure_model_root() / "models/qwen3_8_27b/projection_tiles.json"
        projection_tile_policy = json.loads(policy_path.read_text())
    digest = hashlib.sha256()
    for path in sorted(Path(__import__("models.qwen3_8_27b", fromlist=[""]).__file__).parent.glob("*")):
        if path.is_file() and path.suffix in (".py", ".h", ".json"):
            digest.update(path.name.encode() + path.read_bytes())
    digest.update(json.dumps(dimensions, sort_keys=True).encode())
    digest.update(Path(__file__).with_name("cache_packet_adapter.py").read_bytes())
    digest.update(b"native-prefill-shared-ND-decode-v1")
    digest.update(os.environ.get("PYPTO_QWEN_WEIGHT_CACHE_BYPASS", "0").encode())
    digest.update(
        json.dumps(
            dict(
                fuse_swiglu_down=fuse_swiglu_down,
                projection_tile_policy=projection_tile_policy,
                graph_stop_guard=graph_stop_guard,
                decode_norm=decode_norm,
                phase_nz=phase_nz,
                fuse_reduction_norm=fuse_reduction_norm,
                nz_tile_policy=nz_tile_policy,
                fuse_gdn_ba=fuse_gdn_ba,
                fuse_gdn_qkv_z=fuse_gdn_qkv_z,
                fuse_attention_qkv=fuse_attention_qkv,
            ),
            sort_keys=True,
        ).encode()
    )
    if nz_tile_policy:
        if not phase_nz:
            raise ValueError("NZ tile policy requires phase NZ.")
        for key, (nt, kt, transpose) in nz_tile_policy.items():
            if key not in NZ or kt != 256 or transpose:
                raise ValueError("NZ tuning must preserve the256-row/256-K resident storage tiles.")
    generation = root / digest.hexdigest()[:16]
    programs = []
    specs = []
    for tokens in (1,):
        spec = ModelSpec(
            fuse_attention_qkv=fuse_attention_qkv and tokens == 1,
            fuse_gdn_qkv_z=fuse_gdn_qkv_z and tokens == 1,
            fuse_gdn_ba=fuse_gdn_ba,
            tokens=tokens,
            fuse_mlp=True,
            pack_host_controls=True,
            independent_scratch=True,
            packed_weight_policy=(nz_tile_policy or {}) if phase_nz else None,
            nz_weight_shapes=NZ if phase_nz else (),
            native_weight_order=True,
            native_reduce_round=True,
            native_qk_rope=True,
            graph_stop_guard=graph_stop_guard,
            **dimensions,
        )
        typed_spec = replace(spec, native_cache_pools=(), native_cache_layout=())
        path = generation / f"tokens{tokens}"
        fn = build_model(
            typed_spec,
            path / "source",
            looped=True,
            cube_bf16_store=True,
            norm_tile=128,
            norm_output_tile=1280,
            adaptive_launch=True,
            fuse_reduction_residual=True,
            fuse_reduction_norm=fuse_reduction_norm,
            gdn_decode_views=True,
            early_dispatch=True,
            adaptive_vector_launch=True,
            fuse_swiglu_down=fuse_swiglu_down,
            decode_tile_map=projection_tile_policy,
            decode_norm=decode_norm,
        )
        distribution = DistributedConfig(device_ids=[0, 1], num_sub_workers=0)
        with orchestration_optimization("O1"):
            if (path / "build" / "attention_program_manifest.json").exists() and (
                path / "build" / "mlp_input_manifest.json"
            ).exists():
                compiled = DistributedCompiledProgram.from_dir(
                    path / "build", platform="a2a3", distributed_config=distribution
                )
            else:
                ir.compile(fn, output_dir=str(path / "build"), platform="a2a3", distributed_config=distribution)
                contract_cache_packet(path / "build", typed_spec, spec)
                adapters = importlib.import_module("models.qwen3_8_27b.attention_program_adapter")
                importlib.import_module("models.qwen3_8_27b.fixed_mlp_adapter").install_fixed_mlp(path / "build")
                adapters.install_attention_program(path / "build")
                importlib.import_module("models.qwen3_8_27b.mlp_input_adapter").install_mlp_input(path / "build")
                compiled = DistributedCompiledProgram.from_dir(
                    path / "build", platform="a2a3", distributed_config=distribution
                )
        specs.append(spec)
        programs.append(compiled)
        print("COMPILED_SHARED_CACHE", tokens, len(spec.arguments), path, flush=True)
    return specs, programs, generation
