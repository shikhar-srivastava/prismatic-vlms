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

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
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
        return ImageOps.expand(thumb, border=1, fill="black")

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
    with torch.no_grad():
        _ = model(pixel_values)
    for h in handles:
        h.remove()

    # detect extremes per layer
    for act in layer_acts:
        seq_len, hid_size = act.shape
        flat = act.flatten()
        idxs = np.argsort(flat)
        sel = np.concatenate([idxs[:20], idxs[-20:]]) if idxs.size > 40 else idxs
        outs = []
        for i in sel:
            tok, hid = divmod(int(i), hid_size)
            val = float(act[tok, hid])
            outs.append((val, int(tok), int(hid)))
        layer_outliers.append(outs)

    safe = model_id.replace("/", "__")
    stats = {
        "l2_mean": l2_mean,
        "l2_std": l2_std,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "param_l2_mean": p_l2_mean,
        "param_l2_std": p_l2_std,
        "param_raw_mean": p_raw_mean,
        "param_raw_std": p_raw_std,
    }
    json_path = OUT_DIR / f"{safe}.json"
    with open(json_path, "w") as f:
        json.dump(stats, f)
    print("Stats JSON", json_path)

    def _plot(y_m, y_s, title, ylabel, tag):
        x = range(1, len(y_m) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        plt.fill_between(x, [m - s for m, s in zip(y_m, y_s)], [m + s for m, s in zip(y_m, y_s)], alpha=0.25)
        plt.plot(x, y_m, marker="o", lw=2)
        plt.title(title, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylabel)
        plt.grid(True, ls="--", lw=0.4)
        plt.tight_layout()
        p = OUT_DIR / f"{safe}_{tag}.png"
        plt.savefig(p)
        plt.close()
        print("    Plot ", p)

    def _plot_box(data: List[np.ndarray], outliers: List[List[tuple]]) -> None:
        if not data:
            return
        plt.figure(figsize=(12, 0.5 * len(data) + 3), dpi=300)
        plt.boxplot(
            data,
            vert=True,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "r", "alpha": 0.6},
        )
        for i, outs in enumerate(outliers):
            if not outs:
                continue
            xs = np.full(len(outs), i + 1)
            ys = [v for v, _, _ in outs]
            plt.scatter(xs, ys, c="red", marker="x", zorder=3)
            for x, y, patch, dim in zip(xs, ys, [p for _, p, _ in outs], [d for _, _, d in outs]):
                plt.text(x + 0.1, y, f"p{patch} [d{dim}]", fontsize=6, ha="left", va="center")
        plt.xlabel("Block")
        plt.ylabel("Activation value")
        plt.title(f"{model_id} activation distribution", weight="bold")
        plt.grid(True, ls="--", lw=0.4, axis="y")
        plt.tight_layout()
        p = OUT_DIR / f"{safe}_box.png"
        plt.savefig(p)
        plt.close()
        print("    Plot ", p)

    _plot(l2_mean, l2_std, f"{model_id} ||act||_2", "L2 norm", "l2")
    _plot(raw_mean, raw_std, f"{model_id} raw act", "Activation", "raw")
    _plot(p_l2_mean, p_l2_std, f"{model_id} ||theta||_2", "Parameter L2 norm", "param_l2")
    _plot(p_raw_mean, p_raw_std, f"{model_id} raw theta", "Parameter value", "param_raw")
    _plot_box([a.flatten() for a in layer_acts], layer_outliers)
    return stats


#   DRIVER
all_stats: Dict[str, Dict[str, List[float]]] = {}
for _fam, mids in MODELS.items():
    for mid in mids:
        try:
            all_stats[mid] = analyse_model(mid)
        except Exception as e:
            print(f"[WARN] Skipped {mid}: {e}")

if all_stats:

    def _combined_plot(metric: str, ylabel: str, fname: str, title: str) -> None:
        plt.figure(figsize=(12, 6), dpi=300)
        for mid, s in all_stats.items():
            plt.plot(range(1, len(s[metric]) + 1), s[metric], label=mid, lw=1.4)
        plt.title(title, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylabel)
        plt.grid(True, ls="--", lw=0.4)
        plt.legend(fontsize="x-small", ncol=2)
        plt.tight_layout()
        path = OUT_DIR / fname
        plt.savefig(path)
        plt.close()
        print("\nCombined plot ", path)

    _combined_plot("l2_mean", "L2 norm", "combined_l2.png", "Blockwise activation L2 (mean)")
    _combined_plot("raw_mean", "Activation", "combined_raw.png", "Blockwise raw activation (mean)")
    _combined_plot("param_l2_mean", "Parameter L2 norm", "combined_param_l2.png", "Blockwise parameter L2 (mean)")
    _combined_plot("param_raw_mean", "Parameter value", "combined_param_raw.png", "Blockwise raw parameter (mean)")
else:
    print("No successful runs; combined plot not generated.")

print("\n Analysis complete.")
