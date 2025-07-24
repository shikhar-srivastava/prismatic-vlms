"""modeling_llama_vision_lns.py

Local custom implementation of HuggingFace Llama that activates the *Vision Layer-Norm
Scaling* (Vision-LNS) variant when `NORM_TYPE=vision_lns`.

This implementation applies LNS scaling only to visual token positions, leaving
text tokens with standard pre-normalization.

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
# 2.  Decoder layer with Vision-LNS branch
# ---------------------------------------------------------------------------
class LlamaDecoderLayer(hf.LlamaDecoderLayer):  # type: ignore[misc]
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_index = layer_idx  # Store for LNS scaling calculation (using consistent naming)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        vis_token_indices: Optional[Tuple[int, int]] = None,  # (start_idx, end_idx)
        **kwargs: Any,
    ):
        norm_type = os.getenv("NORM_TYPE", "pre").lower()
        if norm_type != "vision_lns":
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

        # ======= Vision-LNS path =======
        
        # If vis_token_indices not passed directly, try to get from parent model
        if vis_token_indices is None:
            # Try to find the root model that has _current_vis_indices
            current_module = self
            while current_module is not None:
                if hasattr(current_module, '_current_vis_indices'):
                    vis_token_indices = current_module._current_vis_indices
                    break
                # Go up one level to try to find the parent model
                current_module = getattr(current_module, '_parent', None)
                break  # For now, just try once
        
        scale = 1.0 / math.sqrt(self.layer_index + 1)

        # Self-attention with selective Vision-LNS scaling
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply LNS scaling ONLY to visual token positions if they exist
        if vis_token_indices is not None:
            vis_start, vis_end = vis_token_indices
            # Clone to avoid in-place operations that might affect gradients
            scaled_hidden_states = hidden_states.clone()
            # Apply LNS scaling only to visual tokens [batch, vis_start:vis_end, dim]
            scaled_hidden_states[:, vis_start:vis_end, :] = scale * hidden_states[:, vis_start:vis_end, :]
            hidden_states = scaled_hidden_states
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with selective Vision-LNS scaling
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply LNS scaling ONLY to visual token positions if they exist
        if vis_token_indices is not None:
            vis_start, vis_end = vis_token_indices
            # Clone to avoid in-place operations that might affect gradients
            scaled_hidden_states = hidden_states.clone()
            # Apply LNS scaling only to visual tokens [batch, vis_start:vis_end, dim]
            scaled_hidden_states[:, vis_start:vis_end, :] = scale * hidden_states[:, vis_start:vis_end, :]
            hidden_states = scaled_hidden_states
            
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vis_token_indices: Optional[Tuple[int, int]] = None,  # Vision-LNS support
    ):
        # Store vis_token_indices in a way that layers can access it
        # We'll temporarily store it as an attribute on the model
        old_vis_indices = getattr(self, '_current_vis_indices', None)
        self._current_vis_indices = vis_token_indices
        
        try:
            # Call the parent forward method
            result = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return result
        finally:
            # Restore the previous value
            if old_vis_indices is not None:
                self._current_vis_indices = old_vis_indices
            else:
                delattr(self, '_current_vis_indices')


# ---------------------------------------------------------------------------
# 4.  CausalLM wrapper
# ---------------------------------------------------------------------------
class LlamaForCausalLM(hf.LlamaForCausalLM):  # type: ignore[misc]
    def __init__(self, config):  # noqa: D401 — matching HF signature
        super().__init__(config)
        # Replace the model with our vision-LNS aware version
        self.model = LlamaModel(config)
        # Re-run weight initialization
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vis_token_indices: Optional[Tuple[int, int]] = None,  # Vision-LNS support
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vis_token_indices=vis_token_indices,  # Pass vision token indices
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Re-export the config and sequence classification for compatibility
LlamaConfig = hf.LlamaConfig
LlamaForSequenceClassification = hf.LlamaForSequenceClassification 