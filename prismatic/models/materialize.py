"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""
from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import LLaMa2LLMBackbone, LLMBackbone,\
    PythiaLLMBackbone, PhiLLMBackbone, Qwen2LLMBackbone, GemmaLLMBackbone, CustomLlamaLLMBackbone
from prismatic.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    VisionBackbone,
)
from prismatic.models.vlms import PrismaticVLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Pythia Backbones ===
    "pythia-160m": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-410m": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-1b": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-1p4b": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-1p4b-instruct": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-2p8b": {"cls": PythiaLLMBackbone, "kwargs": {}},
    "pythia-6p9b": {"cls": PythiaLLMBackbone, "kwargs": {}},

    # === Phi-2 Backbone ===
    "phi-2-3b": {"cls": PhiLLMBackbone, "kwargs": {}},
    "phi-1_5-1b": {"cls": PhiLLMBackbone, "kwargs": {}},
    
    # === Qwen-2 Backbones ===
    "qwen-1_5-0p5b": {"cls": Qwen2LLMBackbone, "kwargs": {}},
    "qwen-1_5-0p5b-instruct": {"cls": Qwen2LLMBackbone, "kwargs": {}},

    # ==== Gemma 2 Backbones ====
    "gemma-2b": {"cls": GemmaLLMBackbone, "kwargs": {}},
    "gemma-2b-it": {"cls": GemmaLLMBackbone, "kwargs": {}},
    
    # === Custom LLaMa Models with Dynamic LNS/PRE Norm Support ===
    # Use --use_lns or --use_pre flags to control normalization type
    "llama-130m": {"cls": CustomLlamaLLMBackbone, "kwargs": {}},
    # Legacy support for backward compatibility
    "llama-130m-lns": {"cls": CustomLlamaLLMBackbone, "kwargs": {}},
    "llama-130m-pre": {"cls": CustomLlamaLLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
    cfg = None
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            cfg=cfg,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
    llm_teacher: LLMBackbone = None,
    init_projector_path: str = None,
    scale_patch_embeddings: bool = False,
    pre_projection_layer_norm: bool = False
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        llm_teacher = llm_teacher,
        init_projector_path = init_projector_path,
        scale_patch_embeddings = scale_patch_embeddings,
        pre_projection_layer_norm = pre_projection_layer_norm
    )
