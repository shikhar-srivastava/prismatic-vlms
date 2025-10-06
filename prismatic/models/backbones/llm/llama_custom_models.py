"""
llama_custom_models.py

Class definition for custom LLMs with LNS and PRE norm support, derived from LlamaForCausalLM.
"""
from typing import Optional, Type, Callable, List
import os
import torch
from torch import nn as nn

# Dynamic import based on NORM_TYPE - this will be set properly in __init__
# We'll import the correct implementation after determining the norm type
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    PurePromptBuilder,
)
from prismatic.models.backbones.mitigation import apply_mitigation
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`  
overwatch = initialize_overwatch(__name__)

# Registry =>> Support Custom LLaMa Models with dynamic LNS/PRE norm support
# fmt: off
CUSTOM_LLAMA_MODELS = {
    # === Custom 130M LLaMa Models ===
    # Note: Normalization type (LNS vs PRE) is now controlled by command line flags
    # --use_lns or --use_pre, not by the model ID. This allows loading the same
    # checkpoint with different normalization schemes for MLLM training.
    "llama-130m": {
        "llm_family": "llama-custom", 
        "llm_cls": None,  # Will be determined dynamically based on NORM_TYPE 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "../large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "pre"  # Default if no command line override
    },
    
    # === Legacy support for backward compatibility ===
    # These entries are kept for existing scripts that reference them directly
    "llama-130m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": None,  # Will be determined dynamically based on NORM_TYPE 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "../large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "lns"
    },
    
    "llama-130m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": None,  # Will be determined dynamically based on NORM_TYPE 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "../large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "pre"
    },
    
    # === Custom 60M LLaMa Models ===
    "llama-60m": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/60m_res_lns_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/60m_res_lns_lr1e-3_llama/model_20001",
        "default_norm_type": "pre"
    },
    "llama-60m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/60m_res_lns_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/60m_res_lns_lr1e-3_llama/model_20001",
        "default_norm_type": "lns"
    },
    "llama-60m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/60m_res_pre_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/60m_res_pre_lr1e-3_llama/model_20001",
        "default_norm_type": "pre"
    },
    
    # === Custom 250M LLaMa Models ===
    "llama-250m": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/250m_res_lns_lr1e-3_llama/model_40001/config.json",
        "local_checkpoint_path": "../large-activations/250m_res_lns_lr1e-3_llama/model_40001",
        "default_norm_type": "pre"
    },
    "llama-250m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/250m_res_lns_lr1e-3_llama/model_40001/config.json",
        "local_checkpoint_path": "../large-activations/250m_res_lns_lr1e-3_llama/model_40001",
        "default_norm_type": "lns"
    },
    "llama-250m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/250m_res_pre_lr1e-3_llama/model_40001/config.json",
        "local_checkpoint_path": "../large-activations/250m_res_pre_lr1e-3_llama/model_40001",
        "default_norm_type": "pre"
    },
    
    # === Custom 350M LLaMa Models ===
    "llama-350m": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/350m_res_pre_lr5e-4_llama/model_60001/config.json",
        "local_checkpoint_path": "../large-activations/350m_res_pre_lr5e-4_llama/model_60001",
        "default_norm_type": "pre"
    },
    "llama-350m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/350m_res_lns_lr5e-4_llama/model_60001/config.json",
        "local_checkpoint_path": "../large-activations/350m_res_lns_lr5e-4_llama/model_60001",
        "default_norm_type": "lns"
    },
    "llama-350m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/350m_res_pre_lr5e-4_llama/model_60001/config.json",
        "local_checkpoint_path": "../large-activations/350m_res_pre_lr5e-4_llama/model_60001",
        "default_norm_type": "pre"
    },
    
    # === Custom 1B LLaMa Models ===
    "llama-1b": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/1b_res_lns_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/1b_res_lns_lr1e-3_llama/model_20001",
        "default_norm_type": "pre"
    },
    "llama-1b-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/1b_res_lns_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/1b_res_lns_lr1e-3_llama/model_20001",
        "default_norm_type": "lns"
    },
    "llama-1b-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": None,
        "hf_hub_path": None,
        "local_config_path": "../large-activations/1b_res_pre_lr1e-3_llama/model_20001/config.json",
        "local_checkpoint_path": "../large-activations/1b_res_pre_lr1e-3_llama/model_20001",
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
        
        # Dynamically import the correct LLaMA implementation based on NORM_TYPE
        if final_norm_type.lower() == "vision_lns":
            from prismatic.models.llama_custom.modeling_llama_vision_lns import LlamaForCausalLM, LlamaDecoderLayer
            print("Using Vision-LNS LLaMA implementation")
        elif final_norm_type.lower() == "lns":
            from prismatic.models.llama_custom.modeling_llama_lns import LlamaForCausalLM, LlamaDecoderLayer
            print("Using LNS LLaMA implementation")
        else:  # "pre" or any other value
            from prismatic.models.llama_custom.modeling_llama_advanced import LlamaForCausalLM, LlamaDecoderLayer
            print("Using PRE/Advanced LLaMA implementation")
        
        # Store the classes for later use
        self._llama_cls = LlamaForCausalLM
        self._decoder_layer_cls = LlamaDecoderLayer
        
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
        self.llm = self._llama_cls(config)
        
        # Load the state dict from the checkpoint if it exists
        state_dict_path = os.path.join(self.local_checkpoint_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            self.llm.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {state_dict_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {state_dict_path}. "
                f"The provided llm_checkpoint_path '{self.local_checkpoint_path}' does not contain a valid checkpoint. "
                f"Please verify the path is correct and the checkpoint exists."
            )
        
        # [CRITICAL FIX] Set use_cache = False for training (inherited from HFCausalLLMBackbone)
        # Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        # [CRITICAL FIX] Enable input require grads for gradient checkpointing compatibility
        # This was missing because we bypassed HFCausalLLMBackbone initialization
        # Without this, gradient checkpointing fails when LLM has no trainable parameters
        if not self.inference_mode:
            self.llm.enable_input_require_grads()
        
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
        
        # [CRITICAL FIX] Resize token embeddings to match tokenizer vocabulary size
        # This was missing and is essential after adding special tokens
        # All other LLM backbones call this method after modifying tokenizer tokens
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        
        # Ensure the model's generation config matches the tokenizer settings
        if hasattr(self.llm, 'generation_config') and self.llm.generation_config is not None:
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Apply any mitigations
        self.llm = apply_mitigation(self.llm, cfg=cfg)

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing with proper configuration.
        This was missing because we bypassed HFCausalLLMBackbone initialization.
        The custom models now default to use_reentrant=False.
        """
        # Enable gradient checkpointing - our custom models default to use_reentrant=False
        self.llm.gradient_checkpointing_enable()

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Return FSDP wrapping policy for the custom LLaMa model.
        This was missing because we bypassed HFCausalLLMBackbone initialization.
        """
        from functools import partial
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from peft.utils.other import fsdp_auto_wrap_policy
        
        if self.mitigation is None:
            overwatch.info(f"CustomLlamaLLMBackbone's FSDP Wrap Policy: [bold]STANDARD[/]", ctx_level=1)
            transformer_block_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
            )
        else:
            overwatch.info(f"CustomLlamaLLMBackbone's FSDP Wrap Policy: [bold]PEFT[/]", ctx_level=1)
            transformer_block_policy = fsdp_auto_wrap_policy(self.llm)
        return transformer_block_policy

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Embed input token IDs using the LLM's embedding layer."""
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vis_token_indices: Optional[tuple] = None,  # Support Vision-LNS
    ) -> CausalLMOutputWithPast:
        """Forward pass through the custom LLaMa model."""
        
        # Check if the current implementation supports vis_token_indices
        norm_type = os.environ.get("NORM_TYPE", "pre").lower()
        if norm_type == "vision_lns" and vis_token_indices is not None:
            # Pass vis_token_indices to Vision-LNS implementation
            output: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vis_token_indices=vis_token_indices,
            )
        else:
            # Standard forward pass for LNS/PRE implementations
            output: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return output

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # Use pure prompt builder for custom models (no special chat formatting)
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return self._decoder_layer_cls

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Custom LLaMa models can be trained in BF16."""
        return torch.bfloat16

 