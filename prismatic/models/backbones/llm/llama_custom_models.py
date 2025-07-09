"""
llama_custom_models.py

Class definition for custom LLMs with LNS and PRE norm support, derived from LlamaForCausalLM.
"""
from typing import Optional, Type
import os
import torch
from torch import nn as nn

# Import custom advanced modeling when NORM_TYPE is set to lns or pre
from prismatic.models.llama_custom.modeling_llama_advanced import LlamaForCausalLM, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    PurePromptBuilder,
)
from prismatic.models.backbones.mitigation import apply_mitigation

# Registry =>> Support Custom LLaMa Models with dynamic LNS/PRE norm support
# fmt: off
CUSTOM_LLAMA_MODELS = {
    # === Custom 130M LLaMa Models ===
    # Note: Normalization type (LNS vs PRE) is now controlled by command line flags
    # --use_lns or --use_pre, not by the model ID. This allows loading the same
    # checkpoint with different normalization schemes for MLLM training.
    "llama-130m": {
        "llm_family": "llama-custom", 
        "llm_cls": LlamaForCausalLM, 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "/localdisk/ssrivas9/large-activations/configs/llama_130m.json",
        "local_checkpoint_path": "/localdisk/ssrivas9/large-activations/runs/130m_res_LNS_lr1e-3/model_20001",
        "default_norm_type": "pre"  # Default if no command line override
    },
    
    # === Legacy support for backward compatibility ===
    # These entries are kept for existing scripts that reference them directly
    "llama-130m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": LlamaForCausalLM, 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "/localdisk/ssrivas9/large-activations/configs/llama_130m.json",
        "local_checkpoint_path": "/localdisk/ssrivas9/large-activations/runs/130m_res_LNS_lr1e-3/model_20001",
        "default_norm_type": "lns"
    },
    
    "llama-130m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": LlamaForCausalLM, 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "/localdisk/ssrivas9/large-activations/configs/llama_130m.json",
        "local_checkpoint_path": "/localdisk/ssrivas9/large-activations/runs/130m_res_LNS_lr1e-3/model_20001",
        "default_norm_type": "pre"
    },
}
# fmt: on


class CustomLlamaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        cfg = None, 
    ) -> None:
        
        # Get model config
        model_info = CUSTOM_LLAMA_MODELS[llm_backbone_id]
        
        # Determine normalization type: command line flags take precedence over registry
        if "NORM_TYPE" in os.environ:
            # Environment variable already set by command line flags (--use_lns or --use_pre)
            final_norm_type = os.environ["NORM_TYPE"]
            print(f"Using normalization type '{final_norm_type}' from command line override")
        else:
            # Fall back to registry default
            final_norm_type = model_info["default_norm_type"]
            os.environ["NORM_TYPE"] = final_norm_type
            print(f"Using normalization type '{final_norm_type}' from model registry default")
        
        # Log the final decision
        print(f"CustomLlamaLLMBackbone: Final NORM_TYPE = {final_norm_type}")
        
        # Initialize the LLMBackbone base class directly (skip HFCausalLLMBackbone)
        from prismatic.models.backbones.llm.base_llm import LLMBackbone
        LLMBackbone.__init__(self, llm_backbone_id)
        
        # Set the attributes that would normally be set by HFCausalLLMBackbone
        self.llm_family = model_info["llm_family"]
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode
        self.cfg = cfg
        
        if isinstance(self.cfg, dict):
            self.mitigation = self.cfg.get('mitigation', None)
            self.stage = self.cfg.get('stage', None)
            self.first_lora_after_warmup = self.cfg.get('first_lora_after_warmup', None)
        else:
            self.mitigation = getattr(self.cfg, 'mitigation', None)
            self.stage = getattr(self.cfg, 'stage', None)
            self.first_lora_after_warmup = getattr(self.cfg, 'first_lora_after_warmup', None)
            
        self.load_8bit = False
        if isinstance(self.cfg, dict) and 'load_8bit' in self.cfg:
            self.load_8bit = self.cfg.get('load_8bit', False)
        else:
            self.load_8bit = getattr(self.cfg, 'load_8bit', False)
        assert self.load_8bit is False or self.load_8bit is True, "load_8bit must be a boolean"
        
        # Determine checkpoint path: command line override takes precedence over registry default
        if self.cfg is not None:
            if isinstance(self.cfg, dict):
                override_checkpoint_path = self.cfg.get('llm_checkpoint_path', None)
            else:
                override_checkpoint_path = getattr(self.cfg, 'llm_checkpoint_path', None)
        else:
            override_checkpoint_path = None
            
        if override_checkpoint_path is not None:
            self.local_checkpoint_path = override_checkpoint_path
            print(f"Using checkpoint path override: {override_checkpoint_path}")
        else:
            self.local_checkpoint_path = model_info["local_checkpoint_path"]
            print(f"Using registry default checkpoint path: {self.local_checkpoint_path}")
            
        # Config path always comes from registry (model architecture is fixed)
        self.local_config_path = model_info["local_config_path"]
        
        # Load config from local path
        config = LlamaConfig.from_json_file(model_info["local_config_path"])
        
        # Load the model from local checkpoint (overriding the parent's llm attribute)
        print(f"Loading custom LLaMa model from {self.local_checkpoint_path}")
        self.llm = model_info["llm_cls"](config)
        
        # Load the state dict from the checkpoint if it exists
        state_dict_path = os.path.join(self.local_checkpoint_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            self.llm.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {state_dict_path}")
        else:
            print(f"No checkpoint found at {state_dict_path}, using randomly initialized weights")
        
        # Load tokenizer from LLaMa-2 (compatible with our models and supports required single tokens)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=llm_max_length)
        
        # Set up special tokens to match the config from checkpoint
        self.tokenizer.bos_token_id = config.bos_token_id
        self.tokenizer.eos_token_id = config.eos_token_id  
        self.tokenizer.pad_token_id = config.pad_token_id if config.pad_token_id != -1 else self.tokenizer.eos_token_id
        
        # Add pad token if needed (following LLaMa2LLMBackbone pattern)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id
        
        # Apply any mitigations
        self.llm = apply_mitigation(self.llm, cfg=cfg)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # Use pure prompt builder for custom models (no special chat formatting)
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Custom LLaMa models can be trained in BF16."""
        return torch.bfloat16

 