"""
Vision Transformer Activation & Parameter Study
===============================================
Collects per-layer statistics for a set of canonical ViT checkpoints using
`timm`.

This mirrors `analyze_layerwise_activations.py`, replacing LLMs with Vision
Transformers.  Statistics computed per transformer block:

* activation L2-norm / raw value
* **parameter** L2-norm / raw value
* histogram of activations per block

A JSON file with statistics and several plots are written to `viz/plots/act_analysis_vit/`.
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
import numpy as np
import timm
import torch
from PIL import Image

#   CONFIG
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_vit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


#   UTILITIES


def _get_hidden(out: torch.Tensor) -> torch.Tensor:
    """Return tensor regardless of model-specific wrappers."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Unknown output type: {type(out)}")


def _find_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate per-layer transformer blocks for a variety of ViT implementations."""
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
            # Flatten nested structures like Swin's layers
            blocks = []
            for b in cur:
                if hasattr(b, "blocks") and isinstance(b.blocks, (list, torch.nn.ModuleList)):
                    blocks.extend(list(b.blocks))
                else:
                    blocks.append(b)
            return blocks
    # fallback: modules with self-attention attribute
    blocks = [m for m in model.modules() if hasattr(m, "attn") or hasattr(m, "self_attn")]
    return blocks if blocks else []


#   MAIN ANALYSIS FUNC


def analyse_model(model_id: str) -> Dict[str, List[float]]:
    print(f"\n=== Analysing {model_id} ===")

    model = timm.create_model(model_id, pretrained=True).to(DEVICE)
    model.eval()

    blocks = _find_transformer_blocks(model)
    if not blocks:
        print("  ! Could not locate transformer blocks  skipping")
        raise ValueError("no_blocks")

    # Image transform from timm
    data_cfg = timm.data.resolve_model_data_config(model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(DEVICE)

    l2_mean, l2_std, raw_mean, raw_std = [], [], [], []
    p_l2_mean, p_l2_std, p_raw_mean, p_raw_std = [], [], [], []
    act_hist, hist_edges = [], None

    for b in blocks:
        l2s = [p.detach().float().norm(p=2).item() for p in b.parameters()]
        p_l2_mean.append(float(np.mean(l2s)))
        p_l2_std.append(float(np.std(l2s)))
        vals = torch.cat([p.detach().float().view(-1) for p in b.parameters()])
        p_raw_mean.append(vals.mean().item())
        p_raw_std.append(vals.std().item())

    def hook(_, __, output):
        h = _get_hidden(output).float()
        l2 = h.norm(p=2, dim=-1).flatten()
        l2_mean.append(l2.mean().item())
        l2_std.append(l2.std().item())
        flat = h.flatten()
        raw_mean.append(flat.mean().item())
        raw_std.append(flat.std().item())
        nonlocal hist_edges
        data = flat.cpu().numpy()
        if hist_edges is None:
            hist, hist_edges = np.histogram(data, bins=80)
        else:
            hist, _ = np.histogram(data, bins=hist_edges)
        act_hist.append(hist)

    # Register forward hooks on each block to capture the tensor returned by
    # ``forward``.  In ViT implementations from timm this output is the
    # activation **after** residual additions, so hooks record post-residual
    # activations.
    handles = [b.register_forward_hook(hook) for b in blocks]

    with torch.no_grad():
        _ = model(pixel_values)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    hist_matrix = np.vstack(act_hist) if act_hist else np.zeros((1, 1))

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
    print("    Stats JSON ", json_path)

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

    def _plot_hist(matrix, edges, *, log_scale: bool = False):
        if matrix.ndim != 2:
            return
        plt.figure(figsize=(10, 0.6 * matrix.shape[0] + 2), dpi=300)
        data = np.log1p(matrix) if log_scale else matrix
        plt.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[edges[0], edges[-1], 1, matrix.shape[0]],
            cmap="magma",
        )
        plt.colorbar(label="Log Count" if log_scale else "Count")
        plt.xlabel("Activation value")
        plt.ylabel("Block")
        suffix = "hist_log" if log_scale else "hist"
        title = f"{model_id} activation histogram" + (" (log count)" if log_scale else "")
        plt.title(title, weight="bold")
        plt.tight_layout()
        p = OUT_DIR / f"{safe}_{suffix}.png"
        plt.savefig(p)
        plt.close()
        print("    Plot ", p)

    _plot(l2_mean, l2_std, f"{model_id} ||act||_2", "L2 norm", "l2")
    _plot(raw_mean, raw_std, f"{model_id} raw act", "Activation", "raw")
    _plot(p_l2_mean, p_l2_std, f"{model_id} ||theta||_2", "Parameter L2 norm", "param_l2")
    _plot(p_raw_mean, p_raw_std, f"{model_id} raw theta", "Parameter value", "param_raw")
    _plot_hist(hist_matrix, hist_edges)
    _plot_hist(hist_matrix, hist_edges, log_scale=True)
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
