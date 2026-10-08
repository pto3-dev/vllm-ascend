# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Simpler graph capture with changing Qwen3 Decode inputs."""

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_graph_replays_changed_inputs(tmp_path, with_residual):
    from simpler.task_interface import ChipStorageTaskArgs

    from vllm_ascend.pypto.qwen3_rmsnorm import Qwen3RMSNormExecutor

    root = os.getenv("VLLM_ASCEND_PYPTO_LIB_ROOT")
    if not root or not torch.npu.is_available():
        pytest.skip("PyPTO-Lib and an Ascend NPU are required")
    device_id = int(os.environ.get("VLLM_ASCEND_PYPTO_TEST_DEVICE", "0"))
    torch.npu.set_device(device_id)
    executor = Qwen3RMSNormExecutor(
        root,
        str(tmp_path / "rmsnorm"),
        device_id,
    )
    x = torch.randn(16, 5120, dtype=torch.bfloat16, device=f"npu:{device_id}")
    weight = torch.randn(1, 5120, dtype=torch.bfloat16, device=x.device)
    output = torch.empty_like(x)
    residual = torch.randn_like(x) if with_residual else None
    residual_out = torch.empty_like(x) if with_residual else None
    args = ChipStorageTaskArgs()
    tensors = (x, residual, weight, output, residual_out) if with_residual else (x, weight, output)
    for tensor in tensors:
        args.add_tensor(executor._chip_tensor(tensor))
    worker = executor._worker
    handle = executor._fused_handle if with_residual else executor._plain_handle
    token = worker.prepare_graph_run(handle, args, executor._call_config)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream(device=device_id)
    try:
        # vLLM warmup and capture use different streams.
        worker.enqueue_graph_run(token, torch.npu.current_stream(device_id).npu_stream)
        torch.npu.synchronize(device_id)
        with torch.npu.graph(graph, stream=stream):
            worker.enqueue_graph_run(token, stream.npu_stream)
        # Copies happen before each replay, never inside the captured callable.
        for seed in (17, 29, 43):
            torch.manual_seed(seed)
            changed = torch.randn_like(x)
            with torch.npu.stream(stream):
                x.copy_(changed)
                graph.replay()
            stream.synchronize()
            if with_residual:
                expected, _, expected_residual = torch_npu.npu_add_rms_norm(changed, residual, weight.view(-1), 1e-6)
                assert torch.equal(residual_out, expected_residual)
            else:
                expected, _ = torch_npu.npu_rms_norm(changed, weight.view(-1), 1e-6)
            delta = output.float() - expected.float()
            assert torch.isfinite(output).all()
            assert (delta.norm() / expected.float().norm()).item() < 0.005
            assert delta.abs().max().item() <= 0.0625
    finally:
        torch.npu.synchronize(device_id)
        graph.reset()
        worker.finalize_graph_run(token)
        executor.close()
