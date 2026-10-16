# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Account resident PTO weights before native activation profiling."""


def install_weight_initialization():
    import torch

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.load_model

    def load_model(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        causal = getattr(runner.model, "language_model", runner.model)
        executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
        if executor is not None:
            # Weight rebanking has a one-time temporary. Complete it before
            # activation profiling resets its peak-memory counters. Account
            # any permanent allocation change as weights, preserving the
            # normal vLLM available-cache calculation and its checks.
            before = torch.npu.memory_allocated()
            initialization_stream = torch.npu.SyncLaunchStream(device=next(causal.parameters()).device)
            with torch.npu.stream(initialization_stream):
                executor._prepare_shared_weights()
            initialization_stream.synchronize()
            torch.npu.synchronize()
            after = torch.npu.memory_allocated()
            delta = after - before
            runner.model_memory_usage += delta
            executor.shared_weight_info.update(
                initialized_before_activation_profile=True,
                resident_allocation_delta_bytes=delta,
                accounted_weights_bytes=int(runner.model_memory_usage),
            )
        return result

    NPUModelRunner.load_model = load_model


install_weight_initialization()


class Qwen38WorkerExtension:
    def pto_mpmd_stats(self):
        causal = getattr(self.get_model(), "language_model", self.get_model())
        executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
        return executor.stats() if executor is not None else {"pto_executor": False}

    def pto_mpmd_close(self):
        from vllm_ascend.pypto.qwen3_8_mpmd import close_mpmd

        return close_mpmd(self.get_model())
