# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded offline vLLM smoke for the original PyPTO 40-layer graph."""

import argparse
import json

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.97,
        num_gpu_blocks_override=4,
        enable_prefix_caching=False,
        enforce_eager=args.eager,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]},
    )
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=5)
    records = []
    for prompt in ("Huawei is", "The capital of France is", "Huawei is"):
        output = llm.generate([prompt], sampling, use_tqdm=False)[0].outputs[0]
        records.append(
            {
                "prompt": prompt,
                "token_ids": list(output.token_ids),
                "text": output.text,
                "logprobs": [{str(token): entry.logprob for token, entry in step.items()} for step in output.logprobs],
            }
        )
    assert records[0]["token_ids"] == records[2]["token_ids"], records
    assert all(len(record["token_ids"]) == 4 for record in records), records
    print("ORIGINAL_GRAPH_SMOKE_RESULT=" + json.dumps(records, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
