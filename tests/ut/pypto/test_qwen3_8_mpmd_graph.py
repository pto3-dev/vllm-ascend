# SPDX-License-Identifier: Apache-2.0
"""Captured input guards must preserve native slot reuse and reject relocation."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from vllm_ascend.pypto.qwen3_8_mpmd_graph import Qwen38MPMDGraphExecutor


class GraphInputContracts(unittest.TestCase):
    def setUp(self):
        self.executor = Qwen38MPMDGraphExecutor.__new__(Qwen38MPMDGraphExecutor)
        self.executor.causal = NS(
            model=NS(layers=[NS(linear_attn=NS(prefix="gdn"))] * 3 + [NS(self_attn=NS(attn=NS(layer_name="full")))])
        )
        self.gdn = NS(non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32), has_initial_state=None)
        self.full = NS(
            pypto_seq_lens_device=torch.tensor([301], dtype=torch.int32),
            block_tables=torch.tensor([[3, 4, 5, 6]], dtype=torch.int32),
            slot_mapping=torch.tensor([384], dtype=torch.int64),
        )
        self.context = NS(attn_metadata=dict(gdn=self.gdn, full=self.full))
        self.inputs = dict(input_ids=torch.tensor([7]), positions=torch.tensor([300]))
        self.executor.capture_signature = self.executor.input_signature(self.context, self.inputs)
        self.executor.validation_observer = None

    def test_partial_prefill_keeps_nd_until_last_chunk(self):
        self.executor.last_real = False
        self.executor.native_logits = Mock(return_value="native logits")
        self.executor.prefill_complete = False
        self.assertEqual(self.executor.compute_logits(None, None), "native logits")
        self.executor.prefill_complete = True
        self.executor.compute_logits(None, None)

    def test_pto_decode_does_not_trigger_weight_phase_conversion(self):
        self.executor.last_real = True
        self.executor.logits = object()
        self.executor.prefill_complete = True
        self.assertIs(self.executor.compute_logits(None, None), self.executor.logits)

    def test_rotated_native_slots_keep_capture_when_addresses_are_stable(self):
        self.gdn.non_spec_state_indices_tensor.fill_(5)
        self.full.block_tables.fill_(7)
        self.full.slot_mapping.fill_(897)
        self.full.pypto_seq_lens_device.add_(1)
        self.inputs["input_ids"].fill_(12197)
        self.executor.validate_replay_inputs(self.context, self.inputs)

    def test_relocated_native_table_is_rejected_before_replay(self):
        self.full.block_tables = self.full.block_tables.clone()
        with self.assertRaisesRegex(ValueError, "recapture"):
            self.executor.validate_replay_inputs(self.context, self.inputs)

    def test_changed_native_shape_is_rejected_before_replay(self):
        self.full.block_tables = self.full.block_tables[:, :2]
        with self.assertRaisesRegex(ValueError, "recapture"):
            self.executor.validate_replay_inputs(self.context, self.inputs)

    def test_validation_observes_real_replay_input_outside_capture(self):
        self.executor.validation_observer = Mock()
        self.executor.validate_replay_inputs(self.context, self.inputs)
        self.executor.validation_observer.assert_called_once_with(self.inputs["input_ids"], self.inputs["positions"])


if __name__ == "__main__":
    unittest.main()
