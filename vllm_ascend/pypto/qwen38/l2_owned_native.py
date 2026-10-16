# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Startup-only weight ownership transfer; native prefill borrows L2 storage."""

import gc
import time
from math import prod


def rebind_bank(causal, spec, name, bank):
    """Replace every native parameter view into one packed four-layer bank."""

    def bind(parameter, key, shape=None):
        _, offset, original = spec.weights[index, key]
        dimensions = original if shape is None else shape
        parameter.data = bank[offset : offset + prod(dimensions)].view(dimensions)

    if name == "lm_head":
        causal.lm_head.weight.data = bank
        return
    group = int(name.removeprefix("weights"))
    for index in range(group * 4, group * 4 + 4):
        layer = causal.model.layers[index]
        bind(layer.input_layernorm.weight, "input_layernorm.weight")
        bind(layer.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        bind(layer.mlp.gate_up_proj.weight, "mlp.gate_proj.weight", (17408, 5120))
        bind(layer.mlp.down_proj.weight, "mlp.down_proj.weight")
        if index % 4 == 3:
            attn = layer.self_attn
            bind(attn.qkv_proj.weight, "self_attn.q_proj.weight", (7168, 5120))
            bind(attn.o_proj.weight, "self_attn.o_proj.weight")
            bind(attn.q_norm.weight, "self_attn.q_norm.weight")
            bind(attn.k_norm.weight, "self_attn.k_norm.weight")
        else:
            attn = layer.linear_attn
            bind(attn.in_proj_qkvz.weight, "linear_attn.in_proj_qkv.weight", (8192, 5120))
            bind(attn.in_proj_ba.weight, "linear_attn.in_proj_b.weight", (48, 5120))
            bind(attn.out_proj.weight, "linear_attn.out_proj.weight")
            bind(attn.norm.weight, "linear_attn.norm.weight")


def release_native_weight_imports(executor):
    """After graphs drain, release borrowed parameters before L2 allocator exit."""
    import torch

    names = getattr(executor, "l2_owned_weight_names", set())
    if not names:
        return
    torch.npu.synchronize()
    bases = {executor.shared_weights[name].untyped_storage().data_ptr() for name in names}
    for parameter in executor.causal.parameters():
        if parameter.untyped_storage().data_ptr() in bases:
            parameter.data = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
    for name in names:
        executor.shared_weights.pop(name, None)
        executor.exported.pop(name, None)
    gc.collect()
    torch.npu.ipc_collect()
    executor.l2_owned_weight_names = set()


def install_native_weight_adoption():
    from vllm_ascend.pypto.qwen3_8_mpmd import Qwen38MPMDExecutor

    if getattr(Qwen38MPMDExecutor, "_l2_weight_adoption_installed", False):
        return
    original_receive = Qwen38MPMDExecutor._receive

    def receive(executor, timeout=180):
        import torch
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        from vllm_ascend.pypto.qwen3_8_lib import ensure_model_root

        ensure_model_root()
        from models.qwen3_8_27b.model_spec import ModelSpec

        deadline = time.monotonic() + timeout
        while True:
            reply = original_receive(executor, max(0.1, deadline - time.monotonic()))
            command = reply.get("command")
            if command == "adopt_l2_weight":
                if not executor.settings.get("l2_owned_weights"):
                    raise ValueError("Unexpected L2 weight adoption request")
                if executor.bridge is not None:
                    raise RuntimeError("Weight ownership must settle before graph capture")
                name = reply["name"]
                bank = rebuild_npu_tensor(*reply["ipc"])
                previous = executor.shared_weights[name]
                if (
                    bank.shape != previous.shape
                    or bank.dtype != previous.dtype
                    or bank.device != previous.device
                    or not bank.is_contiguous()
                ):
                    raise ValueError("Replacement weight bank metadata differs")
                spec = ModelSpec(tokens=1, fuse_mlp=True, native_weight_order=True)
                rebind_bank(executor.causal, spec, name, bank)
                executor.shared_weights[name] = bank
                executor.exported[name] = bank
                if not hasattr(executor, "l2_owned_weight_names"):
                    executor.l2_owned_weight_names = set()
                executor.l2_owned_weight_names.add(name)
                previous = None
                bank = None
                torch.npu.synchronize()
                gc.collect()
                executor.connection.send(dict(command="adopted_l2_weight", name=name))
            elif command == "old_weight_released":
                torch.npu.synchronize()
                torch.npu.ipc_collect()
                torch.npu.empty_cache()
                torch.npu.synchronize()
                executor.connection.send(dict(command="old_weight_retired", name=reply["name"]))
                executor.shared_weight_info.update(
                    allocation_owner="L2",
                    native_prefill_borrows_weights=True,
                    ownership_transferred_banks=len(executor.l2_owned_weight_names),
                    startup_copy_only=True,
                    full_weight_mirror=False,
                )
            elif reply.get("release_peer"):
                release_native_weight_imports(executor)
                return reply
            else:
                return reply

    Qwen38MPMDExecutor._receive = receive
    Qwen38MPMDExecutor._l2_weight_adoption_installed = True
