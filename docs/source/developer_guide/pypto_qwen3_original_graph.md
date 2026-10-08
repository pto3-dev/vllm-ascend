# Original PyPTO Qwen3-14B Decode in a vLLM NPU graph

This opt-in functional prototype uses PyPTO-Lib's existing
`models/qwen3_14b/decode_fwd.py::decode_fwd_layers`, compiled with
`_CHUNK_NLAYERS=40`. It does not combine the previously installed per-layer
bridges. The library kernel's arithmetic is unchanged.

## Execution boundary

```text
vLLM request / scheduler / model runner
  ├─ Prefill → native Qwen3Model / native decoder layers
  └─ pure Decode, batch=1 → ACLGraphWrapper (FULL_DECODE_ONLY)
       ├─ first call: capture embedding, metadata/KV copies, one 40-layer
       │  ChipCallable launch, KV writeback, and final native RMSNorm
       └─ later calls: NPUGraph.replay(); Python Decoder loop is not rerun
```

The 40-layer callable includes input RMSNorm, QKV projection, Q/K norm and
RoPE, paged attention with cache append, output projection, post-attention
RMSNorm, MLP, and residual carry. vLLM retains scheduling, paged-cache ownership,
embedding, final RMSNorm, LM head, and sampling. Only the Decode model body is
replaced; this is not a graph of the entire serving engine.

## Modification points

- `patch/worker/patch_qwen3_pypto_graph.py` patches `Qwen3Model.forward()` and
  the graph wrapper. The original model forward handles Prefill. Graph
  resources are prepared before the wrapper enters capture. It also skips
  native Attention task-group parameter updates only for this owned pure
  Decode graph; no native Attention task groups were captured on that path.
- `pypto/qwen3_graph.py` validates the model, compiles the original body with
  metadata-only examples, packs transposed per-layer weights, and owns fixed
  argument buffers and a dedicated Simpler `ChipWorker`.
- `attention/attention_v1.py` provides the existing device-side
  `pypto_seq_lens_device` field when this mode is enabled. Captured copies read
  the runner's persistent metadata buffers at replay; CPU sequence lengths
  must not be frozen into the graph.
- `envs.py` and `patch/worker/__init__.py` add the default-off opt-in and reject
  simultaneous use of older per-layer replacement modes.
- PyPTO's Simpler submodule adds `prepare_graph_run`, `enqueue_graph_run`, and
  `finalize_graph_run` across Python, bindings, ChipWorker, common C API, and
  the a2a3 device runner. Graph execution still uses Simpler's device scheduler.

## Memory and correctness limits

The original kernel requires transposed layer-stacked weights and one stacked
BSND KV pool. To avoid keeping two complete copies on a roughly 32 GB card,
native projection parameters are retained on CPU and staged one projection
at a time for native Prefill. This preserves native Prefill operators but
adds transfers: **it is not a fair Prefill/TTFT performance baseline**.
Captured D2D copies synchronize the stacked KV pool with vLLM's native pools.

The original Decode kernel uses FP32 inter-layer carry; native vLLM has BF16
rounding boundaries. Graph-versus-eager equality for this callable does not
prove native-vLLM precision parity. Do not replace the established precision
validated per-layer path or publish a speedup without separate validation.

Initial limits: Qwen3-14B 40 layers, unquantized contiguous ND BF16, TP=PP=1,
one Decode sequence, capture sizes `[1]`, no speculative decoding, no
microbatching, no concurrent replay, no Simpler diagnostic collectors.
It is intended for bounded functionality tests, not production use.

## Bounded validation

In a container with these editable source trees installed, run:

```bash
cd /path/to/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=<device-id> \
VLLM_ASCEND_ENABLE_NZ=0 \
VLLM_ASCEND_PYPTO_QWEN3_ORIGINAL_GRAPH=1 \
VLLM_ASCEND_PYPTO_LIB_ROOT=/path/to/pypto-lib \
python tests/e2e/singlecard/qwen3_original_graph_smoke.py \
  --model /path/to/Qwen3-14B
```

Use `--eager` for the same original callable through ordinary `worker.run()`,
with a Torch-stream fence outside capture. This is independent of the new
graph enqueue path, which has no such host fence inside capture. Compare
the result records from both runs. The smoke uses three sequential requests,
including a repeated prompt, and four generated tokens per request. Stable
token IDs are a smoke check, not a full numerical or performance benchmark.
The smoke also records top-5 log probabilities, so stable text cannot conceal
numerical differences.

`test_qwen3_pypto_graph.py` separately tests plain and fused RMSNorm through
capture plus changed-input replay against native Torch-NPU RMSNorm. Passing
that test establishes the runtime graph mechanism only, not 40-layer accuracy.

Shutdown drains the device, resets captured graphs, finalizes the graph token,
then finalizes the worker. Pinned weights, metadata, cache, and arena storage
must not be freed while a graph can still replay.

## Validation record: 2026-09-30

This validation used an a2a3 development container with one physical NPU.
It is offline vLLM engine execution, not an HTTP-serving benchmark.

- Both a2a3 runtime variants rebuilt successfully from the current runtime
  checkout; the extension was built from the same checkout.
- 15 bridge/runtime unit tests passed, including Prefill preservation, graph
  preparation before capture, native Attention update bypass, and independent
  ordinary-run versus captured-enqueue dispatch.
- 2 changed-input RMSNorm graph tests and 6 ordinary RMSNorm tests passed.
- Native vLLM eager, original PyPTO ordinary `worker.run()`, and original PyPTO
  FULL Decode graph each completed three requests with four generated tokens.
  All 12 greedy token IDs matched across these three paths. FULL capture and
  actual `Replaying aclgraph` were logged, followed by normal shutdown.

**Full-model numerical acceptance remains open.** Stable greedy token IDs did
not imply identical probabilities. For the repeated `Huawei is` request,
the maximum absolute difference among matching top-5 token log probabilities
was:

| Path | Repeated-request max absolute logprob difference |
| ---- | ----------------------------------------------- |
| Native vLLM eager | 0 |
| Original PyPTO ordinary `worker.run()` | 0.1321683 |
| Original PyPTO graph, after ordinary warmup | 0.2344682 |

These are logprob differences, not hidden-state relative L2 errors, not a
percentage accuracy loss, and not a predefined acceptance tolerance.
The ordinary-run control also fluctuates, so these results do not establish
that graph capture alone caused the drift, or that the original kernel alone
is faulty. Weight/cache adaptation, original FP32 carry, non-deterministic
reductions, and runtime state reuse still need a fixed-input hidden-state/KV
comparison. This mode therefore remains experimental and default off. No
performance benefit or full-model numerical parity is claimed.

The retained validation artifacts include native, ordinary-run, captured-graph,
unit-dispatch, and RMSNorm logs. An earlier eager record used uncaptured graph
enqueue, not ordinary `worker.run()`; it must not be substituted for the
ordinary-run control.

## HTTP-serving performance: 300 input tokens, 2026-09-30

This measurement uses the actual vLLM OpenAI-compatible streaming
`/v1/completions` service, not a standalone PyPTO callable benchmark. The
worker imports editable vLLM and vLLM-Ascend checkouts, with
`VLLM_ASCEND_PYPTO_QWEN3_ORIGINAL_GRAPH=1`. Its generated
`orchestration/decode_fwd_layers.cpp` has a 40-iteration layer loop. Service
logs confirm FULL Decode graph capture and `Replaying aclgraph`, and five
successful HTTP requests: two warmups followed by three measured requests.

Configuration: one a2a3 NPU, BF16, TP=PP=1,
concurrency 1, prefix caching disabled, temperature 0, `ignore_eos=true`.
Every measured request has exactly 300 input and 128 output tokens. The two
warmups use the same 300-token input with 8 and 4 output tokens; no 2k-token
request is sent. Compilation, loading, and graph capture finish before the
client starts. The model/runtime arithmetic is unchanged for this run; only
the benchmark client gains an optional `--input-lengths` selection.

| Request | TTFT (s) | TPOT (ms/token) | Decode tokens/s | OTT (tokens/s) | E2EL (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.505460 | 45.256142 | 22.096448 | 17.647894 | 7.252990 |
| 2 | 1.517026 | 45.254509 | 22.097246 | 17.620299 | 7.264349 |
| 3 | 1.515970 | 45.242731 | 22.102998 | 17.626490 | 7.261797 |
| Median | 1.515970 | 45.254509 | 22.097246 | 17.626490 | 7.261797 |

The client measures TTFT from request start to the first nonempty text event,
and E2EL to the last nonempty text event. TPOT is `(E2EL - TTFT) / 127`;
Decode tokens/s (TPOPS for this benchmark) is `127 / (E2EL - TTFT)`;
OTT is `128 / E2EL`. Each request
returned 128 nonempty text events and usage reported 300/128 tokens. These
are client-observed service metrics, including scheduling, sampling, and HTTP
streaming overhead, not device-only kernel timings.

Reproduction, inside the prepared container:

```bash
cd /path/to/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=<device-id> \
OMP_NUM_THREADS=2 \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
VLLM_ASCEND_ENABLE_NZ=0 \
VLLM_BATCH_INVARIANT=1 \
VLLM_ASCEND_PYPTO_QWEN3_ORIGINAL_GRAPH=1 \
VLLM_ASCEND_PYPTO_QWEN3_DEBUG=0 \
VLLM_ASCEND_PYPTO_LIB_ROOT=/path/to/pypto-lib \
VLLM_ASCEND_PYPTO_QWEN3_BUILD_ROOT=/path/to/build/vllm_original_graph \
vllm serve /path/to/Qwen3-14B \
  --host 127.0.0.1 --port 18085 \
  --served-model-name qwen3-14b-graph-bench \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2304 --max-num-seqs 1 --max-num-batched-tokens 2304 \
  --gpu-memory-utilization 0.97 --num-gpu-blocks-override 18 \
  --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

In another container terminal, after `/health` reports ready:

```bash
cd /path/to/vllm-ascend
OMP_NUM_THREADS=2 python tests/e2e/singlecard/qwen3_pypto_benchmark.py \
  --stage original40-full-decode-graph-300 \
  --model-path /path/to/Qwen3-14B \
  --served-model qwen3-14b-graph-bench \
  --url http://127.0.0.1:18085/v1/completions \
  --input-lengths 300 --repeats 3 --output-tokens 128
```

The retained evidence includes result, summary, service, and environment logs.
The environment record includes source revisions,
dirty-source hashes, live worker settings, package paths, and the service
command. The maximum model length, batching limit, and 18-block KV pool were
kept the same as the preceding failed four-length run; the client alone limits
requests to 300 tokens. A preceding 2000-token warmup ran out of memory in
native Prefill's `gate_up_proj`, so there is no valid four-length result.

**These remain experimental performance numbers.** Native Prefill still
stages projection weights from CPU, and the Decode bridge copies the complete
40-layer KV pool into and out of its stacked buffers. TTFT includes this
prototype's CPU-to-NPU transfers. All three output text hashes matched, but
this does not close the numerical-parity issue above. No native-vLLM speedup
or production-readiness claim follows from this measurement.
