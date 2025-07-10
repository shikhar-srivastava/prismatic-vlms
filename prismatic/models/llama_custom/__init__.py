"""prismatic.models.llama_custom

Re-exports Llama classes. When `NORM_TYPE=lns` (case-insensitive) the custom
implementation with Layer-Norm Scaling is provided; otherwise we fall back to
HuggingFace's original implementation, so that upstream code can keep using the
same import path.
"""
from __future__ import annotations

import os

_norm_type = os.getenv("NORM_TYPE", "pre").lower()
if _norm_type == "lns":
    # Local fork with LNS behaviour.
    from .modeling_llama_lns import (
        LlamaConfig,
        LlamaModel,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaDecoderLayer,
    )
else:  # Defer to HuggingFace originals for most classes
    from transformers.models.llama.modeling_llama import (  # type: ignore
        LlamaConfig,  # noqa: F401
        LlamaForSequenceClassification,  # noqa: F401
    )
    # Use advanced custom implementation for model and decoder layer
    from .modeling_llama_advanced import (
        LlamaModel,
        LlamaForCausalLM,
        LlamaDecoderLayer,
    )

__all__ = [
    "LlamaConfig",
    "LlamaModel",
    "LlamaForCausalLM",
    "LlamaForSequenceClassification",
    "LlamaDecoderLayer",
] 