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
- `pypto/qwen3_graph_config.py` resolves the library and graph artifact paths;
  it has no shared per-operator worker registry.
- `attention/attention_v1.py` provides the existing device-side
  `pypto_seq_lens_device` field when this mode is enabled. Captured copies read
  the runner's persistent metadata buffers at replay; CPU sequence lengths
  must not be frozen into the graph.
- `envs.py` and `patch/worker/__init__.py` retain one default-off graph opt-in.
  Older per-operator adapters, shadow/replace modes, and their flags are removed.
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
prove native-vLLM precision parity. Do not publish a speedup or numerical
equality claim without separate validation. The earlier per-layer integration
is no longer an active alternative in this checkout.

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
PTOAS_ROOT=/path/to/pinned-ptoas \
python tests/e2e/singlecard/qwen3_original_graph_smoke.py \
  --model /path/to/Qwen3-14B
```

The smoke is graph-only; it does not expose a separate `--eager` integration.
vLLM's required profiling/warmup still uses ordinary `worker.run()` with a
Torch-stream fence before capture resources are pinned. Captured enqueue has
no such host fence inside capture. The smoke uses three sequential requests,
including a repeated prompt, and four generated tokens per request. Stable
token IDs are a smoke check, not a full numerical or performance benchmark.
The smoke also records top-5 log probabilities, so stable text cannot conceal
numerical differences.

`tests/ut/patch/test_qwen3_pypto_graph_patch.py` checks Decode routing, native
Prefill preservation, path resolution, preparation, launch selection, and
ordered teardown. Operator-specific RMSNorm graph tests were removed with
their adapters; they are not current 40-layer acceptance tests.

Shutdown drains the device, resets captured graphs, finalizes the graph token,
then finalizes the worker. Pinned weights, metadata, cache, and arena storage
must not be freed while a graph can still replay.

## Graph-only cleanup: 2026-10-09

The active adapter consists of `qwen3_graph.py`, `qwen3_graph_config.py`, and
`patch_qwen3_pypto_graph.py`, plus the worker import, environment definitions,
and device sequence-length metadata field. Ten old adapter/runtime files,
four operator patches, six operator-specific tests, two old design guides,
eight `_vllm.py` library kernels, and ten matching golden tests are removed.
Generic Simpler graph APIs and standalone PyPTO-Lib model implementations are
retained.

This is structural cleanup, not a precision repair or performance change.
The user-requested precision rollback remains in effect: original split-K
AtomicAdd, FP32 carry, 32 MiB heap, model-level hook, and compilation mode 0
are preserved. Historical validation below describes earlier source snapshots,
not fresh acceptance of this cleanup.

Cleanup verification used Python 3.11.14, CANN 8.5.1, one physical a2a3 NPU,
and the existing PTOAS 0.60 bundle matching this PyPTO checkout's pin. The stale
Simpler extension was rebuilt in place from the pinned runtime checkout,
using `CMAKE_BUILD_PARALLEL_LEVEL=2` and the `build_package_a2a3` build target.
No CANN or global PTOAS installation was changed.

- Adapter/environment tests: 23 passed, plus 48 environment subtests.
- Simpler ChipWorker tests: 34 passed, including its graph API contracts.
- Ruff checks, formatting, compileall, and whitespace checks passed.
- Active Python sources have no references to removed adapters or flags.
- Original Decode, PyPTO attention, and Prefill source hashes are unchanged.
- Fresh full-model graph smoke: three requests, four tokens per request,
  repeated-prompt token IDs matched, actual `Replaying aclgraph` was logged,
  and the engine shut down normally.

This is bounded functionality validation, not an HTTP performance benchmark
or a native-vLLM numerical comparison. Repeated-prompt log probabilities still
differ after the user-requested precision rollback; cleanup does not repair them.

## Historical validation record: 2026-09-30

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

## Lazy KV sync: 2026-10-09

An opt-in mode removes the per-step KV mirror from the captured Decode graph
and re-syncs the stacked pool only when native forwards have written it —
once per prefill. It changes no kernel, capture, or KV-cache management;
only the timing of the mirror copies.

`VLLM_ASCEND_PYPTO_QWEN3_LAZY_KV_SYNC=1` (default 0) gates the mode. The
mechanism is a dirty flag driven from the two Python hooks that run for every
graph-wrapper call (replays included; the captured runnable's Python does not
run during replay):

- The `Qwen3Model.forward` patch marks `_kv_dirty` on any native (non
  single-decode) forward, i.e. every prefill.
- The `ACLGraphWrapper.__call__` patch mirrors the native per-layer pools
  into the stacked pool eagerly on the current stream — outside capture,
  never recorded — before the next decode step, and clears the flag.

The kernel's in-place KV writes (cache append at `slot_mapping`) then stay
authoritative for the rest of that decode run; the write-back direction is
dropped for decode steps entirely.

```text
default:  replay = [metadata copies] [80x KV mirror in] [kernel] [80x KV mirror out] [norm]
lazy:     replay = [metadata copies] [kernel] [norm]     ← no KV copies captured
          prefill → next replay: one eager sync (~1.1 ms, ~0.74 GB), outside the graph
```

Correctness domain: identical to the bridge itself, plus one condition — the
stacked pool is re-synced only at prefill boundaries, which is exactly
correct while nothing else writes the native pools inside a decode run
(single sequence, prefix caching disabled, no KV connectors). Prefix-cache
block reuse or KV migration would need page-level dirty tracking.

Validation used the post-cleanup sources, the same container, one a2a3 NPU,
and the same benchmark method as the 2026-09-30 HTTP record above (two
warmups, three measured 300/128 requests, client-observed streaming TPOT):

| Mode | TPOT (median) | E2EL (median) | Output hash |
| --- | ---: | ---: | --- |
| Per-step mirror (default, 2026-09-30 record above) | 45.25 ms | 7.262 s | `7371ebd8…` |
| Lazy sync (this run) | **42.36 ms** | 6.92 s | `7371ebd8…` |

−2.90 ms/token; all three output text hashes match the default mode's
recorded hash, and the offline smoke passes with lazy sync enabled,
including the repeated-prompt assertion, which exercises the dirty-flag
re-sync path (prefill → sync → replay). The default path is unchanged:
`VLLM_ASCEND_PYPTO_QWEN3_LAZY_KV_SYNC=0` (or unset) reproduces the
per-step-mirror behavior.

This remains within the same experimental claims as the bridge: no
full-model numerical acceptance, no production-readiness claim.
