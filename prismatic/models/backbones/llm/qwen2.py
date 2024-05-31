"""
qwen2.py
Class definition for all LLMs derived from Qwen2ForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PhiPromptBuilder, PromptBuilder
from prismatic.models.backbones.mitigation import apply_mitigation

# fmt: off
QWEN2_MODELS = {
    "qwen-1_5-0p5b": {
        "llm_family": "qwen2", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen1.5-0.5B"
    },
    "qwen-1_5-0p5b-instruct": {
        "llm_family": "qwen2", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen1.5-0.5B-Chat"
    },
}
# fmt: on


class Qwen2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        cfg=None
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            cfg=cfg,
            **QWEN2_MODELS[llm_backbone_id],
        )

        # [Special Case] Qwen2 has a default token for padding, but no bos_token
        self.tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
        self.llm.config.bos_token_id = self.tokenizer.bos_token_id
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        self.llm = apply_mitigation(self.llm, cfg=cfg)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("phi-2"):
            return PhiPromptBuilder
        elif self.identifier.startswith("phi-1_5"):
            return PhiPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16