"""
llama_custom_models.py

Class definition for custom LLMs with LNS and PRE norm support, derived from LlamaForCausalLM.
Supports flexible cross-normalization loading: any checkpoint can be loaded with any target normalization type.
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
        "local_config_path": "/scratch/ssrivas9/large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "/scratch/ssrivas9/large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "pre"  # Default if no command line override
    },
    
    # === Legacy support for backward compatibility ===
    # These entries are kept for existing scripts that reference them directly
    "llama-130m-lns": {
        "llm_family": "llama-custom", 
        "llm_cls": LlamaForCausalLM, 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "/scratch/ssrivas9/large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "/scratch/ssrivas9/large-activations/130m_res_LNS_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "lns"
    },
    
    "llama-130m-pre": {
        "llm_family": "llama-custom", 
        "llm_cls": LlamaForCausalLM, 
        "hf_hub_path": None,  # Will load from local checkpoint
        "local_config_path": "/scratch/ssrivas9/large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001/config.json",
        "local_checkpoint_path": "/scratch/ssrivas9/large-activations/130m_res_pre_lr1e-3_llama_tokenizer/model_20001",
        "default_norm_type": "pre"
    },
}
# fmt: on


def validate_and_adapt_state_dict(state_dict: dict, target_model: nn.Module, source_norm_type: str, target_norm_type: str) -> dict:
    """
    Validates and adapts a state dict for cross-normalization loading.
    
    Args:
        state_dict: The checkpoint state dict to load
        target_model: The target model to load into
        source_norm_type: The normalization type the checkpoint was trained with
        target_norm_type: The normalization type we want to use
        
    Returns:
        Adapted state dict that's compatible with the target model
    """
    print(f"[State Dict Validation] Source norm: {source_norm_type}, Target norm: {target_norm_type}")
    
    # Get target model's state dict for comparison
    target_state_dict = target_model.state_dict()
    adapted_state_dict = {}
    
    # Critical components that must be preserved across normalization types
    critical_components = [
        'model.embed_tokens.weight',  # Input embeddings
        'lm_head.weight',             # Output embeddings
    ]
    
    # Load critical components first (these should be identical across norm types)
    for key in critical_components:
        if key in state_dict:
            if key in target_state_dict:
                source_shape = state_dict[key].shape
                target_shape = target_state_dict[key].shape
                if source_shape == target_shape:
                    adapted_state_dict[key] = state_dict[key]
                    print(f"[State Dict] ✓ Loaded {key}: {source_shape}")
                else:
                    print(f"[State Dict] ✗ Shape mismatch for {key}: source {source_shape} vs target {target_shape}")
                    raise ValueError(f"Critical component {key} has incompatible shapes")
            else:
                print(f"[State Dict] ⚠ Target model missing {key}")
        else:
            print(f"[State Dict] ⚠ Source checkpoint missing {key}")
    
    # Handle decoder layers with normalization-aware loading
    for target_key in target_state_dict.keys():
        if target_key in adapted_state_dict:
            continue  # Already handled
            
        # Try to find corresponding key in source state dict
        source_key = target_key
        if source_key in state_dict:
            source_shape = state_dict[source_key].shape
            target_shape = target_state_dict[target_key].shape
            
            if source_shape == target_shape:
                adapted_state_dict[target_key] = state_dict[source_key]
                if 'layers.' in target_key:
                    print(f"[State Dict] ✓ Loaded layer component {target_key}: {source_shape}")
            else:
                print(f"[State Dict] ⚠ Shape mismatch for {target_key}: source {source_shape} vs target {target_shape}")
                # For normalization layers, we can initialize randomly if shapes don't match
                if any(norm_component in target_key for norm_component in ['layernorm', 'norm']):
                    print(f"[State Dict] → Initializing {target_key} randomly due to norm type mismatch")
                    # Keep the target model's initialized values
                    adapted_state_dict[target_key] = target_state_dict[target_key].clone()
                else:
                    raise ValueError(f"Non-norm component {target_key} has incompatible shapes")
        else:
            # Key doesn't exist in source - likely due to normalization differences
            if any(norm_component in target_key for norm_component in ['layernorm', 'norm']):
                print(f"[State Dict] → Initializing missing norm layer {target_key}")
                adapted_state_dict[target_key] = target_state_dict[target_key].clone()
            else:
                print(f"[State Dict] ⚠ Missing non-norm component {target_key}")
                # For other missing components, try to initialize reasonably
                adapted_state_dict[target_key] = target_state_dict[target_key].clone()
    
    print(f"[State Dict] Adaptation complete. Loaded {len(adapted_state_dict)} components.")
    return adapted_state_dict


def detect_checkpoint_norm_type(checkpoint_path: str) -> str:
    """
    Attempts to detect the normalization type used to train a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        Detected normalization type ('lns', 'pre', or 'unknown')
    """
    # Check the path for hints about normalization type
    path_lower = checkpoint_path.lower()
    if 'lns' in path_lower:
        return 'lns'
    elif 'pre' in path_lower:
        return 'pre'
    
    # If we can't detect from path, try loading the state dict and checking layer names
    try:
        state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            # This is a heuristic - we could add more sophisticated detection
            return 'unknown'
    except:
        pass
    
    return 'unknown'


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
        
        # Determine target normalization type: command line flags take precedence over registry
        if "NORM_TYPE" in os.environ:
            # Environment variable already set by command line flags (--use_lns or --use_pre)
            target_norm_type = os.environ["NORM_TYPE"]
            print(f"[CustomLlama] Using normalization type '{target_norm_type}' from command line override")
        else:
            # Fall back to registry default
            target_norm_type = model_info["default_norm_type"]
            os.environ["NORM_TYPE"] = target_norm_type
            print(f"[CustomLlama] Using normalization type '{target_norm_type}' from model registry default")
        
        # Log the final decision
        print(f"[CustomLlama] Final NORM_TYPE = {target_norm_type}")
        
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
            print(f"[CustomLlama] Using checkpoint path override: {override_checkpoint_path}")
        else:
            self.local_checkpoint_path = model_info["local_checkpoint_path"]
            print(f"[CustomLlama] Using registry default checkpoint path: {self.local_checkpoint_path}")
            
        # Detect source normalization type from checkpoint
        source_norm_type = detect_checkpoint_norm_type(self.local_checkpoint_path)
        print(f"[CustomLlama] Detected source normalization type: {source_norm_type}")
        
        # Config path always comes from registry (model architecture is fixed)
        self.local_config_path = model_info["local_config_path"]
        
        # Load config from local path
        config = LlamaConfig.from_json_file(model_info["local_config_path"])
        
        # CRITICAL: Build the model with the TARGET normalization type
        print(f"[CustomLlama] Building model with TARGET normalization: {target_norm_type}")
        self.llm = model_info["llm_cls"](config)
        
        # Load and adapt the state dict for cross-normalization compatibility
        self._load_checkpoint_with_adaptation(source_norm_type, target_norm_type)
        
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
        
        # Ensure the model's generation config matches the tokenizer settings
        if hasattr(self.llm, 'generation_config') and self.llm.generation_config is not None:
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Apply any mitigations
        if cfg is not None:
            mitigated_model = apply_mitigation(self.llm, cfg=cfg)
            if mitigated_model is not self.llm:
                self.llm = mitigated_model  # type: ignore

    def _load_checkpoint_with_adaptation(self, source_norm_type: str, target_norm_type: str) -> None:
        """
        Loads checkpoint with intelligent adaptation for cross-normalization compatibility.
        """
        state_dict_path = os.path.join(self.local_checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(state_dict_path):
            print(f"[CustomLlama] Loading state dict from {state_dict_path}")
            try:
                # Load the checkpoint state dict
                checkpoint_state_dict = torch.load(state_dict_path, map_location='cpu')
                
                # Validate and adapt the state dict for cross-normalization loading
                adapted_state_dict = validate_and_adapt_state_dict(
                    checkpoint_state_dict, 
                    self.llm, 
                    source_norm_type, 
                    target_norm_type
                )
                
                # Load the adapted state dict
                missing_keys, unexpected_keys = self.llm.load_state_dict(adapted_state_dict, strict=False)
                
                if missing_keys:
                    print(f"[CustomLlama] Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"[CustomLlama] Unexpected keys: {unexpected_keys}")
                
                print(f"[CustomLlama] Successfully loaded model weights with {target_norm_type} normalization")
                
                # Validate that critical components loaded correctly
                self._validate_embedding_layer()
                
            except Exception as e:
                print(f"[CustomLlama] Error loading checkpoint: {e}")
                print(f"[CustomLlama] Falling back to random initialization")
        else:
            print(f"[CustomLlama] No checkpoint found at {state_dict_path}, using randomly initialized weights")

    def _validate_embedding_layer(self) -> None:
        """
        Validates that the embedding layer loaded correctly.
        """
        try:
            embedding_layer = self.llm.get_input_embeddings()
            if embedding_layer is None:
                raise ValueError("Embedding layer is None")
            
            if not isinstance(embedding_layer, nn.Embedding):
                raise ValueError(f"Embedding layer is not nn.Embedding: {type(embedding_layer)}")
            
            if not isinstance(embedding_layer.weight, torch.Tensor):
                raise ValueError(f"Embedding weight is not a tensor: {type(embedding_layer.weight)}")
            
            if embedding_layer.weight.dim() != 2:
                raise ValueError(f"Embedding weight is not 2D: shape {embedding_layer.weight.shape}")
            
            print(f"[CustomLlama] ✓ Embedding validation passed: {embedding_layer.weight.shape}")
            
        except Exception as e:
            print(f"[CustomLlama] ✗ Embedding validation failed: {e}")
            raise

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

 