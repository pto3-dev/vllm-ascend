# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""MPMD contracts for native pooled caches and per-layer scheduler metadata."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from vllm_ascend.pypto import qwen3_8_mpmd as bridge


class MPMDContracts(unittest.TestCase):
    def test_graph_mode_is_rejected_before_native_device_allocation(self):
        config = NS(
            parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1), model_config=NS(enforce_eager=False)
        )
        with self.assertRaisesRegex(ValueError, "MPMD ACLGraph replay"):
            bridge.Qwen38MPMDExecutor(None, config, {})

    def test_pp2_is_rejected(self):
        config = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=2))
        with self.assertRaisesRegex(ValueError, "TP2/PP1"):
            bridge.Qwen38MPMDExecutor(None, config, {})

    def test_manifest_context_is_accepted_without_device_allocation(self):
        text = NS(hidden_size=5120, num_hidden_layers=64, vocab_size=248320)
        config = NS(
            parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1),
            model_config=NS(enforce_eager=False, hf_text_config=text, max_model_len=2304, quantization=None),
            speculative_config=None,
            lora_config=None,
        )
        causal = NS(model=NS(forward=Mock()), compute_logits=Mock())
        with (
            patch.object(bridge, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(bridge.atexit, "register"),
        ):
            executor = bridge.Qwen38MPMDExecutor(causal, config, dict(graph=True))
        self.assertEqual(executor.max_seq, 2304)
        self.assertIsNone(executor.hidden)
        self.assertIsNone(executor.connection)

    def test_unqualified_context_is_rejected_before_device_allocation(self):
        text = NS(hidden_size=5120, num_hidden_layers=64, vocab_size=248320)
        config = NS(
            parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1),
            model_config=NS(enforce_eager=False, hf_text_config=text, max_model_len=513, quantization=None),
            speculative_config=None,
            lora_config=None,
        )
        with (
            patch.object(bridge, "get_tensor_model_parallel_rank", return_value=0),
            self.assertRaisesRegex(ValueError, "512 or2304"),
        ):
            bridge.Qwen38MPMDExecutor(None, config, dict(graph=True))

    def make_executor(self):
        executor = bridge.Qwen38MPMDExecutor.__new__(bridge.Qwen38MPMDExecutor)
        layers = []
        metadata = {}
        starts = torch.tensor([0, 2], dtype=torch.int32)
        full = NS(
            slot_mapping=torch.tensor([384, 385], dtype=torch.int64),
            block_tables=torch.tensor([[3, 4, 5, 6]], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
        )
        for index in range(4):
            if index == 3:
                layers.append(NS(self_attn=NS(attn=NS(layer_name="full"))))
                metadata["full"] = full
            else:
                prefix = f"gdn{index}"
                layers.append(NS(linear_attn=NS(prefix=prefix)))
                metadata[prefix] = NS(
                    spec_sequence_masks=None,
                    non_spec_state_indices_tensor=torch.tensor([index + 1], dtype=torch.int32),
                    non_spec_query_start_loc=starts,
                    has_initial_state=torch.tensor([False]),
                )
        executor.causal = NS(model=NS(layers=layers))
        executor.max_seq = 512
        executor.table_cols = 4
        return executor, NS(attn_metadata=metadata)

    def test_shared_cache_layers_keep_distinct_native_state_indices(self):
        executor, context = self.make_executor()
        values = executor._controls(context, torch.tensor([100, 101]), torch.tensor([0, 1]))
        self.assertEqual([row[0] for row in values["layer_metadata"][:3]], [1, 2, 3])
        self.assertEqual([row[1] for row in values["layer_metadata"][:3]], [0, 0, 0])
        self.assertEqual(values["slots"], [384, 385])
        self.assertEqual(values["layer_metadata"][3], [0, 1, 2, 3, 4, 5, 6])

    def test_negative_native_append_slot_is_rejected(self):
        executor, context = self.make_executor()
        context.attn_metadata["full"].slot_mapping[0] = -1
        with self.assertRaisesRegex(ValueError, "physical slot"):
            executor._controls(context, torch.tensor([100, 101]), torch.tensor([0, 1]))

    def test_profile_forward_keeps_native_execution_and_weights(self):
        executor = bridge.Qwen38MPMDExecutor.__new__(bridge.Qwen38MPMDExecutor)
        executor.profile_calls = 0
        executor.native_prefill_calls = 0
        executor.settings = {}
        expected = torch.zeros((2, 5120), dtype=torch.bfloat16)
        executor.native_forward = Mock(return_value=expected)
        with (
            patch.object(executor, "_prepare_shared_weights"),
            patch.object(executor, "_connect") as connect,
            patch.object(bridge, "get_forward_context", return_value=NS(attn_metadata=None)),
        ):
            value = executor.forward(None, torch.tensor([1, 2]), torch.tensor([0, 1]))
        self.assertIs(value, expected)
        connect.assert_not_called()
        executor.native_forward.assert_called_once()
        self.assertFalse(executor.last_real)
        self.assertEqual(executor.profile_calls, 1)

    def test_prefill_logits_keep_native_head_and_collective(self):
        executor = bridge.Qwen38MPMDExecutor.__new__(bridge.Qwen38MPMDExecutor)
        executor.last_real = False
        executor.logits = None
        executor.native_logits = Mock(return_value="native-logits")
        hidden = object()
        self.assertEqual(executor.compute_logits(None, hidden), "native-logits")
        executor.native_logits.assert_called_once_with(hidden)

    def test_shared_nd_bank_matches_native_fused_ba_order(self):
        from models.qwen3_8_27b.model_spec import ModelSpec

        spec = ModelSpec(tokens=1, fuse_mlp=True, native_weight_order=True)
        for index in range(64):
            if index % 4 == 3:
                continue
            _, b_offset, b_shape = spec.weights[index, "linear_attn.in_proj_b.weight"]
            _, a_offset, _ = spec.weights[index, "linear_attn.in_proj_a.weight"]
            self.assertEqual(b_shape, (24, 5120))
            self.assertEqual(a_offset, b_offset + 24 * 5120)

    def test_logits_return_pto_result_without_native_head_or_collective(self):
        executor = bridge.Qwen38MPMDExecutor.__new__(bridge.Qwen38MPMDExecutor)
        executor.last_real = True
        executor.logits = torch.empty((1, 248320), dtype=torch.float32)
        self.assertIs(executor.compute_logits(None, None), executor.logits)


if __name__ == "__main__":
    unittest.main()
