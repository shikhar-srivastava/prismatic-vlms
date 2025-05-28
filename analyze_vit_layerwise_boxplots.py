"""
Vision Transformer Activation & Parameter Study
===============================================
Collects per-layer statistics for a set of canonical ViT checkpoints using
`timm`.

This mirrors `analyze_layerwise_activations.py`, replacing LLMs with Vision
Transformers.  Statistics computed per transformer block:

* activation L2-norm / raw value
* **parameter** L2-norm / raw value
* box-and-whisker plot of activations per block

A JSON file with statistics and several plots are written to `viz/plots/act_analysis_vit_box/`.
A combined plot across all successful models is generated at the end.

Assumptions
-----------
* A single-GPU environment is used.
* Models are loaded via `timm` with pretrained weights.
* A single image (`test.png` at repo root) is used for inference.  The default
  image transform for each model is applied via `timm.data.create_transform`.
* Activations exiting each block have shape ``(1, num_patches, hidden_size)``.
* Forward hooks capture the outputs returned by each block's ``forward``
  method.  In timm models this is typically the tensor **after** residual
  connections are applied, so activations represent the post-residual state.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import timm
import torch
from PIL import Image, ImageOps

#   CONFIG
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_vit_box"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

DEVICE = "cuda:0"
IMAGE_PATH = BASE_DIR / "test.png"

# Popular Vision Transformers
MODELS: Dict[str, List[str]] = {
    "vit": [
        "vit_small_patch32_224",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
    ],
    "deit": [
        "deit_base_patch16_224",
    ],
    "swin": [
        "swin_base_patch4_window7_224",
        "swin_large_patch4_window12_224",
    ],
    "beit": [
        "beit_base_patch16_224",
    ],
    "dinov2": [
        "vit_small_patch14_dinov2.lvd142m",
        "vit_small_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
        "vit_giant_patch14_reg4_dinov2.lvd142m",
    ],
    # CLIP ViT models
    "clip": [
        "vit_base_patch16_clip_224.openai",
        "vit_large_patch14_clip_224.openai",
        "vit_large_patch14_clip_336.openai",
    ],
    # SigLIP ViT variants
    "siglip": [
        "vit_base_patch16_siglip_224",
        "vit_base_patch16_siglip_256",
        "vit_base_patch16_siglip_384",
        "vit_so400m_patch14_siglip_224",
        "vit_so400m_patch14_siglip_384",
    ],
}


def _get_hidden(out: torch.Tensor) -> torch.Tensor:
    """Return tensor regardless of model-specific wrappers."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Unknown output type: {type(out)}")


def _find_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate per-layer transformer blocks for various ViT implementations."""
    for path in ["blocks", "encoder.layers", "layers", "model.blocks"]:
        cur = model
        found = True
        for seg in path.split("."):
            if hasattr(cur, seg):
                cur = getattr(cur, seg)
            else:
                found = False
                break
        if found and isinstance(cur, (torch.nn.ModuleList, list)) and len(cur):
            blocks = []
            for b in cur:
                if hasattr(b, "blocks") and isinstance(b.blocks, (list, torch.nn.ModuleList)):
                    blocks.extend(list(b.blocks))
                else:
                    blocks.append(b)
            return blocks
    blocks = [m for m in model.modules() if hasattr(m, "attn") or hasattr(m, "self_attn")]
    return blocks if blocks else []


def analyse_model(model_id: str) -> Dict[str, List[float]]:
    print(f"\n=== Analysing {model_id} ===")
    model = timm.create_model(model_id, pretrained=True).to(DEVICE)
    model.eval()

    blocks = _find_transformer_blocks(model)
    if not blocks:
        print("  ! Could not locate transformer blocks  skipping")
        raise ValueError("no_blocks")

    data_cfg = timm.data.resolve_model_data_config(model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(DEVICE)
    img_h, img_w = pixel_values.shape[-2:]
    resized_img = image.resize((img_w, img_h))

    ps = getattr(model, "patch_size", None)
    if ps is None and hasattr(model, "patch_embed"):
        ps = getattr(model.patch_embed, "patch_size", None)
    if isinstance(ps, (tuple, list)):
        patch_h, patch_w = int(ps[0]), int(ps[1] if len(ps) > 1 else ps[0])
    else:
        patch_h = patch_w = int(ps) if ps is not None else 16
    grid_w = img_w // patch_w

    def patch_thumb(idx: int) -> Image.Image:
        if idx < 0:
            thumb = resized_img.resize((32, 32))
        else:
            r = idx // grid_w
            c = idx % grid_w
            box = (
                c * patch_w,
                r * patch_h,
                c * patch_w + patch_w,
                r * patch_h + patch_h,
            )
            thumb = resized_img.crop(box).resize((32, 32))
        # add small black border around thumbnail
        return ImageOps.expand(thumb, border=1, fill='black')

    # initialize statistics lists
    l2_mean, l2_std, raw_mean, raw_std = [], [], [], []
    p_l2_mean, p_l2_std, p_raw_mean, p_raw_std = [], [], [], []
    layer_acts, layer_outliers = [], []

    # collect parameter norms per block
    for b in blocks:
        norms = [p.detach().float().norm(2).item() for p in b.parameters()]
        p_l2_mean.append(float(np.mean(norms)))
        p_l2_std.append(float(np.std(norms)))
        flat = torch.cat([p.detach().view(-1) for p in b.parameters()])
        p_raw_mean.append(flat.mean().item())
        p_raw_std.append(flat.std().item())

    # hook to collect activations
    def hook(module, inp, output):
        h = _get_hidden(output).float()
        flat = h.flatten()
        layer_acts.append(h.squeeze(0).cpu().numpy())
        l2_mean.append(h.norm(2, dim=-1).mean().item())
        l2_std.append(h.norm(2, dim=-1).std().item())
        raw_mean.append(flat.mean().item())
        raw_std.append(flat.std().item())

    handles = [b.register_forward_hook(hook) for b in blocks]
    with torch.no_grad(): _ = model(pixel_values)
    for h in handles: h.remove()

    # detect outliers per layer
    for act in layer_acts:
        m, s = act.mean(), act.std()
        mask = (act > m + 4*s) | (act < m - 4*s)
        idxs = np.argwhere(mask)
        outs = []
        for tok, hid in idxs:
            val = float(act[tok, hid])
            offs = tok-1 if tok>0 else -1
            outs.append((val, patch_thumb(offs)))
        layer_outliers.append(outs)

    safe = model_id.replace('/', '__')
    stats = {"l2_mean":l2_mean, "l2_std":l2_std, "raw_mean":raw_mean, "raw_std":raw_std,
             "param_l2_mean":p_l2_mean, "param_l2_std":p_l2_std,
             "param_raw_mean":p_raw_mean, "param_raw_std":p_raw_std}
    json_path = OUT_DIR/f"{safe}.json"
    with open(json_path,'w') as f: json.dump(stats, f)
    print("Stats JSON", json_path)

    # plotting helpers omitted for brevity (unchanged)
    # ...
    print("\nAnalysis complete.")
