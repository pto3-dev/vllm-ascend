# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the dense-shape native route guards without importing NPU kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_qwen3_5.py"


def method(class_name, name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def evaluate(node, **namespace):
    return eval(compile(ast.Expression(body=node), str(SOURCE), "eval"), namespace)


@pytest.mark.parametrize(
    "shape,prefills,speculative,expected",
    [
        ((5120, 16, 48, 128), 0, False, True),
        ((4096, 16, 48, 128), 0, False, False),
        ((5120, 8, 48, 128), 0, False, False),
        ((5120, 16, 24, 128), 0, False, False),
        ((5120, 16, 48, 64), 0, False, False),
        ((4096, 8, 24, 64), 1, False, True),
        ((4096, 8, 24, 64), 0, True, True),
    ],
)
def test_gdn_split_guard(shape, prefills, speculative, expected):
    body = method("AscendQwen3_5GatedDeltaNet", "_forward_core")
    assignment = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "split_dense_27b" for t in n.targets)
    )
    branch = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If)
        and any(isinstance(t, ast.Name) and t.id == "split_dense_27b" for t in ast.walk(n.test))
    )
    hidden, key_heads, value_heads, head_dim = shape
    split = evaluate(
        assignment.value,
        self=NS(hidden_size=hidden, num_k_heads=key_heads, num_v_heads=value_heads, head_k_dim=head_dim),
    )
    result = evaluate(
        branch.test,
        split_dense_27b=split,
        attn_metadata=NS(num_prefills=prefills),
        spec_sequence_masks=object() if speculative else None,
    )
    assert result is expected


@pytest.mark.parametrize(
    "model_type,heads,dim,expected",
    [
        ("qwen3_5", 24, 256, False),
        ("qwen3_5", 12, 256, True),
        ("qwen3_5", 24, 128, True),
        ("qwen3_next", 24, 256, False),
        ("qwen3_next", 12, 128, False),
    ],
)
def test_fused_qk_route(model_type, heads, dim, expected):
    body = method("AscendQwen3NextAttention", "forward")
    branch = next(n for n in body.body if isinstance(n, ast.If))
    actual = evaluate(branch.test, self=NS(config=NS(model_type=model_type), num_heads=heads, head_dim=dim))
    assert actual is expected
