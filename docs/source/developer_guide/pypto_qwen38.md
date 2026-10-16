# Qwen3.8-27B PTO TP2 decode

This default-off integration keeps native prefill, allocation and sampling,
then runs the complete 64-layer BF16 text decode through one CPU L3 and two
Simpler L2 workers. It supports TP2/PP1, one sequence, FULL_DECODE_ONLY graphs,
shared ND weights and native KV/GDN storage. Quantization, speculative decode,
LoRA and phase-NZ are unsupported. Static scheduling is not required.

Install matching PyPTO and Simpler source/extension versions and the PyPTO-Lib
model in `models/qwen3_8_27b`. Set `VLLM_ASCEND_PYPTO_LIB_ROOT` to that checkout.
The Simpler build must provide the caller-owned graph API; an upstream runtime
pin without it cannot execute this path. Keep the PyPTO-owned PTOAS/ISA pins.

Start the L3 service on the same host before the HTTP engine. With the selected
toolchain, model checkpoint and allocated device pair in the environment:

```bash
python -m vllm_ascend.pypto.qwen38.mpmd_vllm_graph_server \
  --socket "$PTO_SOCKET" --build-root "$PTO_BUILD_ROOT" --model "$MODEL" \
  --report "$PTO_REPORT" --l2-owned-weights --fuse-gdn-ba \
  --fuse-attention-qkv --decode-norm --fuse-reduction-norm
```

Run the native server through the lifecycle wrapper:

```bash
python -m vllm_ascend.pypto.qwen38.http_server --state-dir "$PTO_STATE_DIR" \
  serve "$MODEL" --tensor-parallel-size 2 --max-model-len 512 \
  --max-num-seqs 1 --max-num-batched-tokens 512 --num-gpu-blocks-override 8 \
  --dtype bfloat16 --language-model-only --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --additional-config "{\"qwen38_pto_mpmd\":{\"socket\":\"$PTO_SOCKET\",\"graph\":true,\"l2_owned_weights\":true}}" \
  --worker-extension-cls vllm_ascend.pypto.qwen38.worker_extension.Qwen38WorkerExtension
```

Use an absolute local Unix socket path. The two native ranks must connect to
one service with the same cache dimensions. Set the supported allocator and
communication environment for the selected platform before startup.

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

The source is extracted from a 40.540270 ms/token recorded candidate, with
static scheduling and compiler mapping hints removed. That number is not a
measurement of this changed combination. Unit and compile-only checks do not
establish NPU end-to-end correctness or performance; revalidate both with the
selected dependency pins before enabling this path in production.
