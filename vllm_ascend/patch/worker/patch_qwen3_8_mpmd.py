# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Attach the optional complete PTO decode bridge to the native text model."""

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

_ORIGINAL_INIT = Qwen3_5ForCausalLM.__init__


def _initialize(self, *, vllm_config, prefix=""):
    _ORIGINAL_INIT(self, vllm_config=vllm_config, prefix=prefix)
    settings = vllm_config.additional_config.get("qwen38_pto_mpmd")
    if settings:
        from vllm_ascend.pypto.qwen3_8_mpmd import attach_mpmd

        attach_mpmd(self, vllm_config, settings)
    elif vllm_config.additional_config.get("qwen38_aclgraph_only", False):
        text = vllm_config.model_config.hf_text_config
        if text.hidden_size != 5120 or text.num_hidden_layers != 64:
            raise ValueError("qwen38_aclgraph_only requires the supported dense 27B text model.")
        self.model.do_not_compile = True


Qwen3_5ForCausalLM.__init__ = _initialize
