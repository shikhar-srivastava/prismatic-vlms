"""modeling_llama_lns.py

Local custom implementation of HuggingFace Llama that activates the *Layer-Norm
Scaling* (LNS) variant when `NORM_TYPE=lns`.

The philosophy is to keep a **thin diff** against upstream: we inherit from HF
classes and override only the pieces that differ.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Any

import torch
from torch import nn

import transformers.models.llama.modeling_llama as hf

# ---------------------------------------------------------------------------
# 1.  Dtype-safe RMSNorm
# ---------------------------------------------------------------------------
class LlamaRMSNorm(hf.LlamaRMSNorm):  # type: ignore[misc]
    """Identical to HF except we cast the output back to `weight.dtype`."""

    def forward(self, hidden_states: torch.Tensor):  # type: ignore[override]
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


# ---------------------------------------------------------------------------
# 2.  Decoder layer with LNS branch
# ---------------------------------------------------------------------------
class LlamaDecoderLayer(hf.LlamaDecoderLayer):  # type: ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Any,
    ):
        norm_type = os.getenv("NORM_TYPE", "pre").lower()
        if norm_type != "lns":
            # Defer to upstream implementation for all other modes.
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        # ======= LNS path =======
        scale = 1.0 / math.sqrt(self.layer_index + 1)

        # Self-attention (pre-norm + scaling)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states * scale
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Feed-forward (post-norm + scaling)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states * scale
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# 3.  Model wrapper that swaps in the custom Norm and Layer
# ---------------------------------------------------------------------------
class LlamaModel(hf.LlamaModel):  # type: ignore[misc]
    def __init__(self, config):  # noqa: D401 — matching HF signature
        super().__init__(config)
        # Replace RMSNorm in embedding/lns (the HF constructor already built those)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[assignment]
        # Rebuild decoder layers with our custom class
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # Re-run HF weight initialisation for new params
        self.post_init()


class LlamaPreTrainedModel(hf.LlamaPreTrainedModel):  # type: ignore[misc]
    """Extend HF's LlamaPreTrainedModel to support gradient checkpointing with use_reentrant=False."""
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing with proper use_reentrant parameter."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        # Store the checkpointing kwargs for use in forward pass
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
        # Enable gradient checkpointing using the standard HF method
        super().gradient_checkpointing_enable()
    
    def enable_input_require_grads(self):
        """Enable gradients on input embeddings for gradient checkpointing compatibility."""
        def _enable_input_require_grads(module):
            """Helper function to enable input require grads on embedding layers."""
            if hasattr(module, "require_grad_"):
                module.require_grad_(True)
        
        # Enable gradients on embedding layer
        if hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
            self.model.embed_tokens.require_grad_(True)
            # Register forward hook to enable gradients on input embeddings
            def _hook(module, input, output):
                if output.requires_grad:
                    return output
                else:
                    return output.requires_grad_(True)
            
            self.model.embed_tokens.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# 4.  Heads that rely on `LlamaModel`
# ---------------------------------------------------------------------------
class LlamaForCausalLM(LlamaPreTrainedModel, hf.LlamaForCausalLM):  # type: ignore[misc]
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)  # Get our gradient checkpointing support
        # super(hf.LlamaForCausalLM, self).__init__(config)  # initialise PreTrainedModel part
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


class LlamaForSequenceClassification(LlamaPreTrainedModel, hf.LlamaForSequenceClassification):  # type: ignore[misc]
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)  # Get our gradient checkpointing support
        # super(hf.LlamaForSequenceClassification, self).__init__(config)
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()


# ---------------------------------------------------------------------------
# 5.  Re-export config for convenience
# ---------------------------------------------------------------------------
LlamaConfig = hf.LlamaConfig  # Re-export unchanged

__all__ = [
    "LlamaConfig", 
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForCausalLM",
    "LlamaForSequenceClassification",
] 