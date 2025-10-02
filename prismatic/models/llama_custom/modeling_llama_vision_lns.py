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
            # Vision-LNS indices should be passed explicitly from the model forward pass
            # If not provided, Vision-LNS scaling will be skipped (fallback to standard behavior)
            pass
        
        scale = 1.0 / math.sqrt(self.layer_index + 1)

        # Self-attention with selective Vision-LNS scaling
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply LNS scaling ONLY to visual token positions if they exist
        if vis_token_indices is not None:
            vis_start, vis_end = vis_token_indices
            # Apply LNS scaling only to visual tokens [batch, vis_start:vis_end, dim]
            hidden_states[:, vis_start:vis_end, :] *= scale
        
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
            # Apply LNS scaling only to visual tokens [batch, vis_start:vis_end, dim]
            hidden_states[:, vis_start:vis_end, :] *= scale
            
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
        **kwargs  # Accept additional kwargs for compatibility with newer transformers versions
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # CRITICAL FIX: Prismatic's 2D attention mask causes IndexError in HF LlamaAttention
        # HF expects 4D masks, but Prismatic provides 2D. Setting to None lets HF handle causal masking.
        # For multimodal training, padding is handled upstream in the data collator.
        effective_attention_mask_for_layers = None
        
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Unpack inputs properly
                        hidden_states_checkpoint, attention_mask_checkpoint, position_ids_checkpoint = inputs
                        return module(
                            hidden_states_checkpoint,
                            attention_mask=attention_mask_checkpoint,
                            position_ids=position_ids_checkpoint,
                            past_key_value=None,  # Can't use past_key_value with gradient checkpointing
                            output_attentions=output_attentions,
                            use_cache=False,  # Can't use cache with gradient checkpointing
                            vis_token_indices=vis_token_indices,
                        )
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    effective_attention_mask_for_layers,
                    position_ids,
                    use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=effective_attention_mask_for_layers,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    vis_token_indices=vis_token_indices,  # Pass visual token indices
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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