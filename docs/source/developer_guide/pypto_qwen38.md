# Qwen3.8-27B PTO TP2 decode

This default-off integration keeps native prefill, allocation and sampling,
then runs the complete 64-layer BF16 text decode through one CPU L3 and two
Simpler L2 workers. It supports TP2/PP1, one sequence, FULL_DECODE_ONLY graphs,
shared ND weights and native KV/GDN storage. Quantization, speculative decode,
LoRA and phase-NZ are unsupported. The selected baseline requires guarded attention/MLP static regions and fixed
placement at their outer boundaries; other work remains dynamically scheduled.

Install matching PyPTO and Simpler source/extension versions and the PyPTO-Lib
model in `models/qwen3_8_27b`. Set `VLLM_ASCEND_PYPTO_LIB_ROOT` to that checkout.
The Simpler build must provide the caller-owned graph API; an upstream runtime
pin without it cannot execute this path. Use compiler/runtime source builds that provide fixed placement, the static
program builders and mailbox wakeups. See the model README for the recorded
PTOAS/ISA identities and validation boundary.

Start the L3 service on the same host before the HTTP engine. With the selected
toolchain, model checkpoint and allocated device pair in the environment:

```bash
python -m vllm_ascend.pypto.qwen38.mpmd_vllm_graph_server \
  --socket "$PTO_SOCKET" --build-root "$PTO_BUILD_ROOT" --model "$MODEL" \
  --report "$PTO_REPORT" --l2-owned-weights --fuse-gdn-ba \
  --fuse-attention-qkv --decode-norm --fuse-reduction-norm \
  --projection-tile-policy "$VLLM_ASCEND_PYPTO_LIB_ROOT/models/qwen3_8_27b/projection_tiles.json"
```

Run the native server through the lifecycle wrapper:

```bash
python -m vllm_ascend.pypto.qwen38.http_server --state-dir "$PTO_STATE_DIR" \
  serve "$MODEL" --tensor-parallel-size 2 --max-model-len 512 \
  --max-num-seqs 1 --max-num-batched-tokens 512 --num-gpu-blocks-override 8 \
  --dtype bfloat16 --language-model-only --no-enable-prefix-caching \
  --gpu-memory-utilization 0.96 --seed 1024 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --additional-config "{\"qwen38_pto_mpmd\":{\"socket\":\"$PTO_SOCKET\",\"graph\":true,\"l2_owned_weights\":true}}" \
  --worker-extension-cls vllm_ascend.pypto.qwen38.worker_extension.Qwen38WorkerExtension
```

Use an absolute local Unix socket path. The two native ranks must connect to
one service with the same cache dimensions. Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`,
`OMP_NUM_THREADS=1`, `OMP_PROC_BIND=false` and `HCCL_BUFFSIZE=64` for the
recorded shared-ND configuration. Select the same visible pair for both
processes and source the installed CANN/NNAL environment before startup.
Use one explicit Python environment providing PyPTO, Simpler, vLLM and
vLLM-Ascend. Historical work-directory PYTHONPATH fallbacks are unnecessary.

The worker extension accounts packed weights before activation profiling.
With L2-owned weights, native prefill borrows the L2 allocations. If enabled,
`PYPTO_QWEN_WEIGHT_CACHE_BYPASS=1` applies only to read-only decode weight loads;
it is refused for native-owned IPC mappings. No per-token cache mirror or
external vLLM-to-L3 RPC is added by FULL replay.

Keep the lifecycle wrapper running through shutdown. It drains the stop replay,
resets caller graphs and releases peer imports before the native consumers exit;
L2 owners release their weight allocations last. A forced process kill cannot
provide this ordering. Initialization, capture and stop failures require cleanup
before reusing the service.

The selected source follows `INTEGRATION_BASELINE_20261017.md`, generation
`ea00209693a8e241`: recorded three-request median 39.311919 ms/token on cards
6/7 and 39.420060 ms/token on cards 1/3, B1/input300/output128/maxlen512.
These are historical measurements of the same source version, not a timing of
this relocated package. Revalidate output, all 391 complete worker replays,
absence of native decode fallback/JIT misses and ordered shutdown after moving
paths or updating dependencies. Keep performance probes separate from serving.

The model plan storage is zeroed and populated before graph preparation.
Internal trace flags are disabled before the first ordinary warmup; timed
replays do not collect internal timestamps. The service refuses generated ABI
mismatches instead of silently falling back to native decode.
