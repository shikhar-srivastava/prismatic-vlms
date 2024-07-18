from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PythiaPromptBuilder,\
                    PythiaInstructPromptBuilder, PromptBuilder
from prismatic.models.backbones.mitigation import apply_mitigation

# Registry ==> Support Phi Models (from HF Transformers)
# fmt: off
PYTHIA_MODELS = {
    "pythia-160m": {
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-160m-deduped"
    },
    "pythia-410m": {
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-410m-deduped"
    },
     "pythia-1b":{
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-1b-deduped"
    },
    "pythia-1p4b":{
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-1.4b-deduped"
    },
    "pythia-1p4b-instruct":{
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "lambdalabs/pythia-1.4b-deduped-synthetic-instruct"
    },
    "pythia-2p8b":{
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-2.8b-deduped"
    },
    "pythia-6p9b":{
        "llm_family": "pythia", "llm_cls": GPTNeoXForCausalLM, "hf_hub_path": "EleutherAI/pythia-6.9b-deduped"
    }
}
# fmt: on


class PythiaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        cfg = None, 
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            cfg=cfg,
            **PYTHIA_MODELS[llm_backbone_id],
        )

        # [Special Case] Phi PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        if self.inference_mode or not self.first_lora_after_warmup:
            self.llm = apply_mitigation(self.llm, cfg=cfg)
            
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("pythia"):
            if self.identifier.endswith("-instruct"):
                return PythiaInstructPromptBuilder
            else:
                return PythiaPromptBuilder
        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GPTNeoXLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16