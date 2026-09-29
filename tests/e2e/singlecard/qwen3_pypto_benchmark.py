"""Reproducible single-request Qwen3 Decode service benchmark.

Example: python tests/e2e/singlecard/qwen3_pypto_benchmark.py \
    --stage baseline --model-path /path/to/Qwen3-14B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request

from transformers import AutoTokenizer


def request_once(url: str, model: str, prompt: list[int], max_tokens: int, *, stage: str, repeat: int) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    first = last = None
    usage = None
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks.append(piece)
    done = time.perf_counter()
    if first is None or last is None or usage is None:
        raise RuntimeError("Missing output text or usage in streaming response")
    count = usage["completion_tokens"]
    if count != max_tokens:
        raise RuntimeError(f"Expected {max_tokens} output tokens, received {count}")
    e2el = last - start
    return {
        "stage": stage,
        "length": len(prompt),
        "repeat": repeat,
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": count,
        "ttft_s": first - start,
        "e2el_s": e2el,
        "tpot_ms": 1000 * (last - first) / (count - 1),
        "ott_toks": count / e2el,
        "stream_done_s": done - start,
        "chunks": len(chunks),
        "output_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="qwen3-14b-bench")
    parser.add_argument("--url", default="http://127.0.0.1:18084/v1/completions")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2 to calculate TPOT")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    passage = "Huawei develops AI systems and software. The model answers questions accurately. " * 200
    token_ids = tokenizer.encode(passage, add_special_tokens=False)
    lengths = (300, 600, 1000, 2000)
    if len(token_ids) < max(lengths):
        raise RuntimeError("Prompt source is shorter than 2000 tokens")
    prompts = {length: token_ids[:length] for length in lengths}

    request_once(args.url, args.served_model, prompts[300], 8, stage=args.stage, repeat=-1)
    request_once(args.url, args.served_model, prompts[2000], 4, stage=args.stage, repeat=-2)
    records = []
    for length in lengths:
        for repeat in range(args.repeats):
            record = request_once(
                args.url,
                args.served_model,
                prompts[length],
                args.output_tokens,
                stage=args.stage,
                repeat=repeat,
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
            records.append(record)
    for length in lengths:
        items = [r for r in records if r["length"] == length]
        median = {
            "stage": args.stage,
            "length": length,
            "repeats": args.repeats,
            "median_ttft_s": statistics.median(r["ttft_s"] for r in items),
            "median_tpot_ms": statistics.median(r["tpot_ms"] for r in items),
            "median_e2el_s": statistics.median(r["e2el_s"] for r in items),
            "median_ott_toks": statistics.median(r["ott_toks"] for r in items),
            "output_hashes_match": len({r["output_sha256"] for r in items}) == 1,
        }
        print(json.dumps(median, ensure_ascii=False), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
