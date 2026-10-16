# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise runner graph routing without loading NPU worker dependencies."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"


class GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2

    def has_full_cudagraphs(self):
        return self is GraphMode.FULL


class CompileMode(Enum):
    NONE = 0
    VLLM_COMPILE = 3


def runner_class():
    tree = ast.parse(RUNNER.read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")


@pytest.mark.parametrize(
    "compile_mode,graph_mode,eager,expected",
    [
        (CompileMode.NONE, GraphMode.FULL, False, True),
        (CompileMode.NONE, GraphMode.PIECEWISE, False, False),
        (CompileMode.VLLM_COMPILE, GraphMode.PIECEWISE, False, True),
        (CompileMode.VLLM_COMPILE, GraphMode.NONE, False, False),
        (CompileMode.NONE, GraphMode.FULL, True, False),
        (CompileMode.VLLM_COMPILE, GraphMode.FULL, True, False),
    ],
)
def test_graph_route(compile_mode, graph_mode, eager, expected):
    node = next(n for n in runner_class().body if isinstance(n, ast.FunctionDef) and n.name == "_use_aclgraph")
    namespace = {"CUDAGraphMode": GraphMode, "CompilationMode": CompileMode}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=compile_mode, cudagraph_mode=graph_mode),
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    assert namespace["_use_aclgraph"](runner) is expected


@pytest.mark.parametrize(
    "multimodal,language_only,expected",
    [
        (True, True, False),
        (True, False, True),
        (True, None, True),
        (False, True, False),
        (False, False, False),
        (False, None, False),
    ],
)
def test_text_only_uses_token_id_input(multimodal, language_only, expected):
    initialize = next(n for n in runner_class().body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(
        n
        for n in ast.walk(initialize)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "is_multimodal_model" for t in n.targets)
    )
    config = SimpleNamespace(
        is_multimodal_model=multimodal,
        multimodal_config=None if language_only is None else SimpleNamespace(language_model_only=language_only),
    )
    expression = ast.Expression(body=assignment.value)
    assert eval(compile(expression, str(RUNNER), "eval"), {"self": SimpleNamespace(model_config=config)}) is expected
