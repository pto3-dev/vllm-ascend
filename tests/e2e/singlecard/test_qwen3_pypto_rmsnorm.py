#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""NPU parity for Qwen3-14B PyPTO Decode RMSNorm against Ascend native ops."""

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend import envs  # noqa: E402
from vllm_ascend.pypto.qwen3_rmsnorm import Qwen3RMSNormExecutor  # noqa: E402

ROOT = envs.VLLM_ASCEND_PYPTO_LIB_ROOT
DEVICE_ID = int(os.getenv("VLLM_ASCEND_PYPTO_TEST_DEVICE", "0"))


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    if ROOT is None or not torch.npu.is_available():
        pytest.skip("PyPTO-Lib and an Ascend NPU are required")
    torch.npu.set_device(DEVICE_ID)
    instance = Qwen3RMSNormExecutor(ROOT, str(tmp_path_factory.mktemp("qwen3-rmsnorm")), DEVICE_ID)
    yield instance
    instance.close()


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_matches_native(executor, batch, with_residual):
    torch.manual_seed(20260924 + batch)
    device = torch.device(f"npu:{DEVICE_ID}")
    hidden = torch.randn((batch, 5120), device=device, dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(5120, device=device)).to(torch.bfloat16)
    residual = torch.randn_like(hidden) if with_residual else None
    if residual is None:
        expected, _ = torch_npu.npu_rms_norm(hidden, weight, 1e-6)
        expected_residual = hidden
    else:
        expected, _, expected_residual = torch_npu.npu_add_rms_norm(hidden, residual, weight, 1e-6)

    actual, actual_residual = executor.run(hidden, weight, residual)
    assert torch.equal(actual_residual, expected_residual)
    assert torch.isfinite(actual).all()
    delta = actual.float() - expected.float()
    relative_l2 = delta.norm() / expected.float().norm()
    row_l2 = delta.norm(dim=1) / expected.float().norm(dim=1)
    assert relative_l2.item() < 0.005, (
        f"row_l2={row_l2.cpu().tolist()}, "
        f"actual_row0={actual[0, :8].cpu().tolist()}, "
        f"expected_row0={expected[0, :8].cpu().tolist()}, "
        f"actual_last={actual[-1, :8].cpu().tolist()}, "
        f"expected_last={expected[-1, :8].cpu().tolist()}"
    )
    assert delta.abs().max().item() <= 0.0625
