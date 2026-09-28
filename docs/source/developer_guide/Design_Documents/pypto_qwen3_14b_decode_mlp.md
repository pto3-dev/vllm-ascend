# Experimental PyPTO Qwen3-14B Decode MLP

This opt-in path replaces **only the MLP** in each Qwen3-14B decoder layer during
pure Decode. Prefill, embedding, RMSNorm, attention, paged KV cache, residual
handling, and logits stay in native vLLM/vLLM-Ascend. It is not a fused
40-layer PyPTO model and has no established performance benefit.

## Call path

```text
ModelRunner._model_forward
  -> Qwen3ForCausalLM.forward
  -> Qwen3Model.forward (40 decoder layers)
  -> Qwen3DecoderLayer.forward
     -> native attention and post-attention RMSNorm
     -> PyPTO Qwen3MLPExecutor.run (Decode only)
     -> native residual and next layer
```

The opt-in patch is imported from `vllm_ascend.patch.worker`. It checks the
Ascend attention metadata for `DecodeOnly` and requires TP=1, contiguous BF16
ND weights with hidden size 5120 and intermediate size 17408, and 1–16 query
tokens. Unsupported pure-Decode inputs fail explicitly. Other attention
states use the unchanged native forward. The executor compiles
`models/qwen3_14b/mlp_vllm.py` from `VLLM_ASCEND_PYPTO_LIB_ROOT` on its first eligible
call, registers the resulting ChipCallable with Simpler, and reuses it for all
layers. It does not repack or concatenate model weights.

## Run modes

- `off` (default): entirely native.
- `shadow`: execute both MLPs, log max-absolute and RMS differences, and
  return the native result.
- `replace`: return the PyPTO result.

Set `VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE` to one of these values. For
`shadow` and `replace`, set `VLLM_ASCEND_PYPTO_LIB_ROOT` to the PyPTO-Lib
checkout. `VLLM_ASCEND_PYPTO_QWEN3_BUILD_ROOT` optionally overrides the common
compiled-artifact root (default: a temporary directory). PyPTO, the A2/A3 Simpler runtime,
vLLM, and vLLM-Ascend must be installed from their matching source checkouts.

Example for a single NPU in eager mode:

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
export VLLM_BATCH_INVARIANT=1  # Needed on the validated CANN 8.5.1 image.
export VLLM_ASCEND_PYPTO_LIB_ROOT=/path/to/pypto-lib
export VLLM_ASCEND_PYPTO_QWEN3_MLP_MODE=shadow
python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-14B",
    dtype="bfloat16",
    max_model_len=128,
    max_num_seqs=1,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.97,
    enforce_eager=True,
)
print(llm.generate(["Huawei is"], SamplingParams(max_tokens=3, temperature=0))[0].outputs[0].text)
PY
```

On the validated CANN 8.5.1 container, `VLLM_BATCH_INVARIANT=1` was also
needed to avoid an unavailable `AddRmsNormBias` operator in the native
normalization path. This setting is environment-specific, not part of the
PyPTO MLP interface.

## Validation boundary

On the validated single-card setup, the PyPTO MLP golden test passed on NPU 5.
For two prompts and eight greedy generated tokens each, `replace` issued
280 MLP dispatches (seven Decode steps × 40 layers); text, token IDs, and
selected-token logprobs matched the native run exactly. A matching `shadow`
run found `max_abs=0` at every one of its 280 MLP calls. This is a bounded
regression, not a proof of accuracy for all prompts, batch sizes, or
configurations. Run broader parity and benchmark suites before production use.
