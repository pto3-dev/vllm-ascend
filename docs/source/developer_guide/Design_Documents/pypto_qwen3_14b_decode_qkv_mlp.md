# Experimental PyPTO Qwen3-14B Decode RMSNorm + QKV + MLP + Fused Paged Attention

This opt-in integration can replace input/post-attention RMSNorm, packed
QKV projection, RoPE, attention, and MLP during pure Decode. The input/post
RMSNorm path also performs vLLM's fused BF16 residual add where needed. The
separate Q/K RMSNorm and KV-cache writes stay native unless fused attention
is selected. Output projection, embedding, and logits remain native. Prefill
is entirely native. This is not a 40-layer fused PyPTO model. The older
one-page attention replacement still executes native attention for online
comparison; no speedup has been established.

An additional fused-attention mode moves the replacement boundary earlier. It
consumes vLLM's post-input-RMS packed QKV, then performs Q/K RMSNorm, NeoX RoPE,
current-token KV-cache append, and multi-page GQA attention in one PyPTO
callable. Its `replace` mode skips native Q/K RMSNorm, RoPE, cache update, and
attention. Output projection, residuals, MLP routing, logits, and Prefill retain
their existing ownership.

## Call path

```text
ModelRunner._model_forward
  -> Qwen3ForCausalLM.forward
  -> Qwen3Model.forward (40 decoder layers)
     -> input RMSNorm (native or PyPTO; Decode-only replacement)
     -> Qwen3Attention.forward
        -> PyPTO fused Q/K norm + RoPE + KV append + multi-page attention
           (optional alternative; `replace` skips the four native steps below)
        -> PyPTO QKV projection (optional; Decode only)
        -> native Q/K RMSNorm
        -> PyPTO RoPE (optional; Decode only)
        -> native KV-cache write
        -> native fused-infer or paged attention
        -> PyPTO one-page attention (optional shadow/replace; Decode only)
        -> native output projection
     -> post-attention RMSNorm + residual add (native or PyPTO)
     -> MLP (native or PyPTO; Decode-only replacement)
     -> next layer
```

The opt-in patches in `vllm_ascend.patch.worker` inspect Ascend attention
metadata and run only in `DecodeOnly`. They require TP=1, contiguous BF16
weights, no QKV bias, hidden size 5120, QKV size 7168, intermediate size
17408, and 1–16 query tokens. Unsupported pure-Decode inputs fail explicitly.
The kernels are compiled from `models/qwen3_14b/qkv_vllm.py`,
`models/qwen3_14b/rope_vllm.py`, and `models/qwen3_14b/mlp_vllm.py` in
PyPTO-Lib on first eligible use. All callables register with **one shared
Simpler ChipWorker per NPU**; independently initialized workers on the same
card failed on the second Decode dispatch with a generic AICPU exception in
this environment. A fused Q/K RMSNorm + RoPE candidate is available only as a
shadow diagnostic because replacement changed selected-token log probabilities.
The shared worker is finalized at process exit. No weight concatenation or
offline repacking is required.

## One-page attention boundary

With `enforce_eager=True`, Qwen3 Decode in the tested vLLM-Ascend revision calls
`AscendAttentionBackendImpl.forward_fused_infer_attention`, not
`_npu_paged_attention`. The patch wraps both native attention paths and runs only
for `DecodeOnly`. `shadow` returns native output; `replace` returns PyPTO output.
Both modes currently execute native attention first, preserving KV-cache behavior
and online numerical comparison. `replace` is therefore a parity prototype, not
a performance implementation.

The executor in `vllm_ascend.pypto.qwen3_paged_attention` reads each request's
first physical block ID from the device block table, pads that small metadata and
`seq_lens` to 16 requests, and passes the full device-resident KV cache directly
to `models/qwen3_14b/paged_attention_vllm.py` through the shared Simpler worker.
It no longer gathers or copies KV blocks per layer. The kernel supports TP=1,
BF16, Qwen3-14B's 40 Q heads and 8 KV heads, 1–16 Decode requests, block size 128,
and at most one 128-token page per request. The cache-block dimension is dynamic,
so one compiled artifact accepts the cache allocation chosen by vLLM. Softmax and
the weighted-value reduction stay in FP32, with the value dimension tiled by 64
to fit the A2/A3 vector-buffer limit.

## Fused attention boundary

`VLLM_ASCEND_PYPTO_QWEN3_FUSED_PA_MODE` accepts `off`, `shadow`, or
`replace`. The patch enters before native Q/K RMSNorm and RoPE. It obtains the
current layer's `AscendMetadata` from `get_forward_context()`, and passes the
device-resident block table, sequence lengths, slot mapping, and that layer's KV
cache to `vllm_ascend.pypto.qwen3_fused_attention`.

The vLLM adapter compiles
`models/qwen3_14b/paged_attention_vllm.py::qwen3_vllm_fused_attention`.
vLLM's QKV is already post-input-RMS BF16, whereas PyPTO-Lib's serving pipeline
feeds FP32 projection accumulators plus a deferred inverse-RMS scalar. The
adapter therefore promotes vLLM QKV to FP32 and supplies an identity inverse-RMS
factor before reusing `paged_attention_pypto_swpipe`. vLLM's 64-wide cosine and
sine tables are duplicated to the 128-wide NeoX layout expected by that kernel.

`shadow` runs PyPTO first, then native Q/K RMSNorm, RoPE, cache update, and
attention; it returns native output and logs the difference. `replace` runs
only PyPTO for those four operations and then resumes at the native output
projection. The fused mode requires the older
`VLLM_ASCEND_PYPTO_QWEN3_PA_MODE`, QK-RoPE mode, and standalone RoPE mode to
be `off`; enabling both boundaries is rejected. Prefill remains native.

## Modes and run

`VLLM_ASCEND_PYPTO_QWEN3_QKV_MODE`,
`VLLM_ASCEND_PYPTO_QWEN3_ROPE_MODE`, and
`VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE` are independent. Each accepts
`off` (default), `shadow` (run PyPTO but return native output and log
differences), or `replace` (return PyPTO output). Set
`VLLM_ASCEND_PYPTO_LIB_ROOT` to
the matching PyPTO-Lib checkout when any mode is enabled.
`VLLM_ASCEND_PYPTO_QWEN3_QK_ROPE_MODE` accepts only `off` or `shadow`; it is
the fused Q/K RMSNorm + RoPE diagnostic and is not a replacement path.
`VLLM_ASCEND_PYPTO_QWEN3_PA_MODE` accepts `off`, `shadow`, or `replace`.
`replace` is validation-only: it returns the PyPTO context but still executes
native attention for comparison. `VLLM_ASCEND_PYPTO_QWEN3_BUILD_ROOT`
optionally overrides the common compiled-artifact root; each operator uses its
own child directory. Use a new root after changing a kernel to avoid reusing
stale compiled artifacts.

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
export VLLM_BATCH_INVARIANT=1  # Required by the native CANN 8.5.1 path tested here.
export VLLM_ASCEND_PYPTO_LIB_ROOT=/path/to/pypto-lib
export VLLM_ASCEND_PYPTO_QWEN3_QKV_MODE=replace
export VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE=replace
export VLLM_ASCEND_PYPTO_QWEN3_FUSED_PA_MODE=replace
export VLLM_ASCEND_PYPTO_QWEN3_ROPE_MODE=off
export VLLM_ASCEND_PYPTO_QWEN3_QK_ROPE_MODE=off
export VLLM_ASCEND_PYPTO_QWEN3_PA_MODE=off
python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-14B",
    dtype="bfloat16",
    max_model_len=128,
    max_num_seqs=2,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.97,
    enforce_eager=True,
)
for output in llm.generate(
    ["Huawei is", "The capital of China is"],
    SamplingParams(max_tokens=8, temperature=0.0, logprobs=1),
):
    print(output.outputs[0].text, output.outputs[0].token_ids)
PY
```

PyPTO, its A2/A3 Simpler runtime, vLLM, and vLLM-Ascend must be installed
from the matching source checkouts. The `VLLM_BATCH_INVARIANT` workaround
is specific to the tested CANN 8.5.1 image, not part of the PyPTO interface.

## Validation boundary

On one NPU, the QKV and RoPE golden tests passed. RoPE `shadow` was bitwise
exact for the sampled Decode calls. Combined QKV+RoPE+MLP `replace` used the
shared runtime and dispatched each callable 280 times (seven Decode steps × 40
layers). Its generated text, token IDs, and selected-token logprobs matched the
native baseline exactly for both prompts.

The fused Q/K RMSNorm + RoPE diagnostic did not pass the replacement gate:
shadow differences reached `max_abs=0.125`, and replacement changed selected-token
logprobs. It therefore remains shadow-only while Q/K RMSNorm stays native.

The one-page attention kernel passed its dedicated NPU golden test (`1 passed`)
across sequence lengths 1 through 128. At every diagnosed new-global-maximum
dispatch, the earlier BF16-exp implementation matched a CPU emulation of that
operation order bitwise. That evidence rules out a QK/PV codegen error for the
observed outliers; native differences came from rounding semantics. Worst
elements differed by exactly one BF16 ULP, for example `21.25` versus `21.125`.

The current FP32 softmax and weighted-value implementation is nearly identical to
the CPU FP32 reference (diagnostic RMSE around `1e-5`). In an attention-only
`replace` run with two prompts and eight generated tokens, all 16 generated token
IDs matched native. The largest selected-token logprob difference was about
`0.05384`. Across 280 layer dispatches, native-versus-PyPTO attention measured
`max_abs=0.25`, `RMSE=0.000860055`, and `reference_RMS=0.47196`. The `0.25`
maximum is magnitude-dependent BF16 spacing; it is not an FP32-reference error.

The initial combined QKV + RoPE + MLP + attention `replace` run also matched all
16 native token IDs. Its largest selected-token logprob difference was about
`0.04599`; attention comparison ended at `max_abs=0.25`, `RMSE=0.000862684`,
and `reference_RMS=0.471935` after 280 layer dispatches.

The expanded batch-8 gate covered prompt lengths 2, 5, 9, 17, 33, 65, 97, and
117, with eight generated tokens per request. All 64 generated token IDs matched
native. The largest selected-token logprob difference was `0.0757331`, and the
mean absolute difference was `0.00675748`. Attention comparison ended at
`max_abs=0.125`, `RMSE=0.000946627`, and `reference_RMS=0.604516` after 280
layer dispatches.

Replacing the compact-KV gather with direct physical-page reads produced exactly
the same generated texts, token IDs, and selected-token logprobs as the gathered
prototype. Its dedicated NPU golden test also passed non-contiguous physical
block IDs across sequence lengths 1 through 128.

This passes the bounded token-parity and `abs(logprob_delta) < 0.1` prototype
gate, but it is not bitwise parity. Keep `shadow` as the default and treat the
older one-page `replace` mode as validation-only.

The new vLLM fused-attention adapter passed its NPU golden for batch 8,
non-contiguous physical pages, current-token cache append, and sequence lengths
crossing the 128- and 256-token page boundaries. In a two-prompt, eight-token
full-model shadow run, 280 fused dispatches returned exactly the same generated
tokens and selected-token logprobs as the previously validated native-attention
run. Per-report attention RMSE ranged from about `0.0036` to `0.0056` against
reference RMS `1.32` to `2.04`; the worst observed element delta was
`0.140625`.

The fused `replace` run skipped native cache update and attention and preserved
all 16 generated token IDs. Its maximum selected-token logprob delta was
`0.06010664`, with mean absolute delta `0.01357792`. A separate multi-page
full-model gate used prompt lengths 5, 129, 257, and 381. All 8 generated token
IDs matched; maximum selected-token logprob delta was `0.00069015`, and mean
absolute delta was `0.00014077625`. These are correctness results, not a
performance claim.

A repeated multi-page gate then generated eight tokens for prompt lengths 5, 129,
257, and 369. It executed 280 fused replacements (seven Decode iterations times
40 layers), preserving all 32 token IDs. Maximum selected-token logprob delta was
`0.03571599`, and mean absolute delta was `0.0032033794`. This additionally
exercises repeated cache growth beyond the first page.

## Input/post-attention RMSNorm (Decode only)

`VLLM_ASCEND_PYPTO_QWEN3_RMS_MODE` accepts `off` (default), `shadow`, and
`replace`. It is independent of the QKV, fused-attention, and MLP modes.
The patch in `patch_qwen3_pypto_mlp.py` enters only when Ascend attention
metadata is `DecodeOnly`; Prefill calls the original vLLM layer forward.

```text
Qwen3DecoderLayer.forward
  -> rms_forward(input_layernorm, hidden_states, residual)
  -> Qwen3Attention.forward (native or the selected PyPTO path)
  -> rms_forward(post_attention_layernorm, attention_output, residual)
  -> Qwen3MLP.forward (native or the selected PyPTO path)
```

`patch_qwen3_pypto_rmsnorm.py` guards TP=1, BF16, hidden size 5120, epsilon
1e-6, no bias, and 1-16 Decode tokens. The adapter
`vllm_ascend.pypto.qwen3_rmsnorm` compiles
`models/qwen3_14b/rmsnorm_vllm.py` in the matching
`VLLM_ASCEND_PYPTO_LIB_ROOT` checkout,
registers its plain and fused-add callables with the shared Simpler worker,
and pads each batch to 16. The fused kernel rounds the residual sum to BF16
before FP32 variance/normalization, matching the native tensor boundary.
`shadow` returns native values and logs the per-call maximum differences;
`replace` returns PyPTO values.

On the tested A2/A3 NPU, six operator comparisons against
`torch_npu.npu_rms_norm` and `torch_npu.npu_add_rms_norm` passed for batch
sizes 1, 3, and 16. The two PyPTO CPU-Golden-versus-NPU tests and seven
Decode-routing unit tests passed. With the same Qwen3-14B weights and
deterministic sampling, three independent 32-token prompts matched the
native service's generated token sequence both with RMSNorm-only
replacement and with RMSNorm+QKV+fused attention+MLP replacement. A
three-request batch of 16-token generations also matched. In the combined
path, the largest selected-token logprob difference across these checks
was about 0.066. These are bounded correctness checks, not a proof for
all inputs or a performance claim.

covers repeated PyPTO cache appends after crossing page boundaries.
