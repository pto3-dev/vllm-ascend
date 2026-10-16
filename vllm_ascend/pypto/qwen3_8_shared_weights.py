# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One resident ND weight representation for native prefill and PTO decode."""

from math import prod

import torch


def bank_native_weights(causal, spec):
    """Rebind native parameters to shared banks, one four-layer group at a time.

    The temporary group allocation is released by replacing native parameter
    storage after its values are copied. Small transposed convolution parameters
    have a separate PTO view; large projections have one resident copy.
    """
    if not spec.native_weight_order or spec.packed_weight_policy is not None:
        raise ValueError("Shared ND banks require the native fused BA order and ND projections.")
    device = causal.lm_head.weight.device
    result = {}
    copy_bytes = 0

    def bind(parameter, bank, offset, shape):
        parameter.data = bank[offset : offset + prod(shape)].view(shape)

    with torch.no_grad():
        for group, count in spec.group_counts.items():
            bank = torch.empty(count, dtype=torch.bfloat16, device=device)
            for index in range(group * 4, group * 4 + 4):
                layer = causal.model.layers[index]
                if layer.layer_scale:
                    raise ValueError("Layer scale is unsupported by the PTO model.")
                gate, up = layer.mlp.gate_up_proj.weight.split(8704, dim=0)
                values = {
                    "input_layernorm.weight": layer.input_layernorm.weight,
                    "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
                    "mlp.gate_proj.weight": gate,
                    "mlp.up_proj.weight": up,
                    "mlp.down_proj.weight": layer.mlp.down_proj.weight,
                }
                if index % 4 == 3:
                    attn = layer.self_attn
                    q, k, v = attn.qkv_proj.weight.split([6144, 512, 512], dim=0)
                    values.update(
                        {
                            "self_attn.q_proj.weight": q,
                            "self_attn.k_proj.weight": k,
                            "self_attn.v_proj.weight": v,
                            "self_attn.o_proj.weight": attn.o_proj.weight,
                            "self_attn.q_norm.weight": attn.q_norm.weight,
                            "self_attn.k_norm.weight": attn.k_norm.weight,
                        }
                    )
                else:
                    attn = layer.linear_attn
                    qkv, z = attn.in_proj_qkvz.weight.split([5120, 3072], dim=0)
                    b, a = attn.in_proj_ba.weight.split(24, dim=0)
                    values.update(
                        {
                            "linear_attn.in_proj_qkv.weight": qkv,
                            "linear_attn.in_proj_z.weight": z,
                            "linear_attn.in_proj_b.weight": b,
                            "linear_attn.in_proj_a.weight": a,
                            "linear_attn.out_proj.weight": attn.out_proj.weight,
                            "linear_attn.conv1d.weight": attn.conv1d.weight.view(5120, 4).T,
                            "linear_attn.norm.weight": attn.norm.weight,
                        }
                    )
                for (number, key), (_, offset, shape) in spec.weights.items():
                    if number != index:
                        continue
                    value = values[key]
                    if tuple(value.shape) != shape:
                        raise ValueError(f"Native TP shape mismatch: {index} {key} {tuple(value.shape)} != {shape}")
                    bank[offset : offset + value.numel()].copy_(value.reshape(-1))
                    copy_bytes += value.numel() * 2
                for key, parameter in [
                    ("input_layernorm.weight", layer.input_layernorm.weight),
                    ("post_attention_layernorm.weight", layer.post_attention_layernorm.weight),
                    ("mlp.down_proj.weight", layer.mlp.down_proj.weight),
                ]:
                    _, offset, shape = spec.weights[index, key]
                    bind(parameter, bank, offset, shape)
                _, offset, _ = spec.weights[index, "mlp.gate_proj.weight"]
                bind(layer.mlp.gate_up_proj.weight, bank, offset, (17408, 5120))
                if index % 4 == 3:
                    _, offset, _ = spec.weights[index, "self_attn.q_proj.weight"]
                    bind(attn.qkv_proj.weight, bank, offset, (7168, 5120))
                    for key, parameter in [
                        ("self_attn.o_proj.weight", attn.o_proj.weight),
                        ("self_attn.q_norm.weight", attn.q_norm.weight),
                        ("self_attn.k_norm.weight", attn.k_norm.weight),
                    ]:
                        _, offset, shape = spec.weights[index, key]
                        bind(parameter, bank, offset, shape)
                else:
                    _, offset, _ = spec.weights[index, "linear_attn.in_proj_qkv.weight"]
                    bind(attn.in_proj_qkvz.weight, bank, offset, (8192, 5120))
                    _, offset, _ = spec.weights[index, "linear_attn.in_proj_b.weight"]
                    bind(attn.in_proj_ba.weight, bank, offset, (48, 5120))
                    for key, parameter in [
                        ("linear_attn.out_proj.weight", attn.out_proj.weight),
                        ("linear_attn.norm.weight", attn.norm.weight),
                    ]:
                        _, offset, shape = spec.weights[index, key]
                        bind(parameter, bank, offset, shape)
                del values
            torch.npu.synchronize()
            # Local slice references can otherwise keep the previous group alive.
            del gate, up, q, k, v, a, b, qkv, z
            result[f"weights{group}"] = bank
            torch.npu.empty_cache()
        floating = torch.empty(spec.float_weights_count, dtype=torch.float32, device=device)
        for index, layer in enumerate(causal.model.layers):
            if index % 4 == 3:
                continue
            ordinal = index - index // 4
            attn = layer.linear_attn
            floating[ordinal * 48 : ordinal * 48 + 24].copy_(attn.A_log)
            floating[ordinal * 48 + 24 : ordinal * 48 + 48].copy_(attn.dt_bias)
            attn.A_log.data = floating[ordinal * 48 : ordinal * 48 + 24]
            attn.dt_bias.data = floating[ordinal * 48 + 24 : ordinal * 48 + 48]
        result.update(
            float_weights=floating,
            embedding=causal.model.embed_tokens.weight.detach(),
            final_norm=causal.model.norm.weight.detach().view(1, 5120),
            lm_head=causal.lm_head.weight.detach(),
        )
    return result, dict(rebanked_bytes=copy_bytes, layout="shared ND, Cube GM-to-L1 ND2NZ", full_weight_mirror=False)
