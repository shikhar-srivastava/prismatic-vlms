#!/usr/bin/env python3
# analyze_cnn_whisker_box.py
#
# Purpose:
#   Generate layer-wise activation & parameter statistics for multiple pretrained CNNs,
#   and create whisker box plots of the raw activations (with outlier annotations).
#
#   This script is inspired by the "Residual vs Dense vs Plain CNN – dual-depth study"
#   reference, ensuring a whisker box plot of raw activations is generated.
#
# Usage example:
#   python analyze_cnn_whisker_box.py
#
# Requirements:
#   - PyTorch
#   - torchvision
#   - matplotlib
#   - Pillow (for Image I/O)
#
# Note:
#   Make sure you have a valid test image "test.png" (224×224) in the working directory,
#   or specify your own path below in IMAGE_PATH.

from __future__ import annotations
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_cnn_box"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = BASE_DIR / "test.png"  # Provide your own image here if needed.

# Models to analyze
MODELS: Dict[str, str] = {
    "resnet":   "resnet18",     # Residual net
    "densenet": "densenet121",  # Dense net
    "cnn":      "vgg16_bn",     # Plain-ish CNN
}

# ───────────────────────── HELPERS ─────────────────────────
def _transform(size: int = 224) -> T.Compose:
    """
    Preprocessing pipeline for the input image.
    """
    return T.Compose([
        T.Resize(size + 32),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

def _get_blocks(model: torch.nn.Module, tag: str):
    """
    Extract the per-block or per-layer modules from each architecture.
    """
    if tag == "resnet":
        # For ResNet-18: the 4 major layers, each has multiple BasicBlocks.
        return [b for layer in [model.layer1, model.layer2, model.layer3, model.layer4]
                for b in layer]
    elif tag == "densenet":
        # For DenseNet-121: the 4 dense blocks, each with multiple DenseLayers.
        feats = model.features
        dbs = [feats.denseblock1, feats.denseblock2, feats.denseblock3, feats.denseblock4]
        return [b for db in dbs for b in db]  # Each b is a DenseLayer
    else:
        # For VGG16_bn: a sequential list with conv2d/batchnorm/relu layers.
        # We'll pick out only conv2d modules as the "blocks."
        return [m for m in model.features if isinstance(m, torch.nn.Conv2d)]

def _param_stats(m: torch.nn.Module) -> Tuple[float, float, float, float]:
    """
    Compute parameter-level stats:
       - param_l2_mean, param_l2_std : L2 norms of each parameter tensor
       - param_raw_mean, param_raw_std : raw param values across the entire block
    Returns (param_l2_mean, param_l2_std, param_raw_mean, param_raw_std).
    """
    vecs = [p.detach().float().reshape(-1) for p in m.parameters() if p.requires_grad]
    if not vecs:
        return 0.0, 0.0, 0.0, 0.0
    flat = torch.cat(vecs)
    l2s  = torch.stack([v.norm(2) for v in vecs])
    return (l2s.mean().item(), l2s.std().item(), flat.mean().item(), flat.std().item())

def _load_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """
    Load an ImageNet-pretrained torchvision model by ID, using either:
      - new Weights Enum API (if available),
      - or legacy pretrained=True approach.
    """
    fn = getattr(tvm, model_id, None)
    if fn is None:
        raise ValueError(f"Unknown model: {model_id}")

    # Attempt new-style "Weights" usage
    # e.g. for resnet18 -> "resnet18_weights"
    try:
        enum_name = f"{model_id}_weights".lower()
        enum = next((getattr(tvm, n) for n in dir(tvm) if n.lower() == enum_name), None)
        if enum is not None and hasattr(enum, "DEFAULT"):
            return fn(weights=enum.DEFAULT).to(device).eval()
    except Exception:
        pass

    # Fallback to old API:
    return fn(pretrained=True).to(device).eval()

# ───────────────────────── ANALYSIS ─────────────────────────
def analyse(tag: str, model_id: str, max_blocks: Optional[int]) -> Dict[str, List[float]]:
    """
    Analyze the specified model:
      1. Identify blocks.
      2. Gather parameter L2 stats.
      3. Forward on a test image, hooking each block's output to gather activation stats.
      4. Generate whisker box plot with outliers & store stats in JSON.

    Args:
        tag: A short label identifying the net (e.g. 'resnet', 'densenet', 'cnn').
        model_id: The torchvision constructor name (e.g. 'resnet18').
        max_blocks: If not None, limit analysis to the first N blocks.

    Returns: A dict with metric name -> list of per-block stats.
    """
    label = f"{model_id}_{'full' if max_blocks is None else f'top{max_blocks}'}"
    print(f"\n=== Analyzing {label} ===")

    # Load model
    try:
        model = _load_model(model_id, DEVICE)
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower():
            print("  [!] GPU OOM – attempting CPU …")
            model = _load_model(model_id, torch.device("cpu"))
        else:
            raise

    # Identify the blocks
    blocks = _get_blocks(model, tag)
    if not blocks:
        raise ValueError(f"No blocks found for {model_id}")
    if max_blocks is not None and len(blocks) > max_blocks:
        blocks = blocks[:max_blocks]

    # Prepare containers for stats
    act_L2_m, act_L2_s    = [], []
    act_raw_m, act_raw_s  = [], []
    par_L2_m, par_L2_s    = [], []
    par_raw_m, par_raw_s  = [], []
    activations           = []
    outliers              = []

    # Parameter stats
    for b in blocks:
        l2m, l2s, rm, rs = _param_stats(b)
        par_L2_m.append(l2m)
        par_L2_s.append(l2s)
        par_raw_m.append(rm)
        par_raw_s.append(rs)

    # Activation stats: forward hooks
    def _hook(_, __, out):
        h = out.detach().float()
        if h.ndim == 4:  # (B,C,H,W)
            # store raw activations
            # we assume B=1
            h = h.squeeze(0)
            activations.append(h.cpu().numpy())
            # L2 norms by channel
            l2vals = torch.norm(h.flatten(1), p=2, dim=1)
            act_L2_m.append(l2vals.mean().item())
            act_L2_s.append(l2vals.std().item())
            flat = h.flatten()
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())
        else:
            # fallback for shapes not 4D
            flat = h.flatten()
            activations.append(h.cpu().numpy())
            act_L2_m.append(h.norm(2).item())
            act_L2_s.append(0.0)
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())

    hdls = [b.register_forward_hook(_hook) for b in blocks]

    # Single image forward pass
    img = Image.open(IMAGE_PATH).convert("RGB")
    x = _transform(224)(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        _ = model(x)
    for h in hdls:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Outlier listing
    # For each activation map, find top/bottom values
    for a in activations:
        if a.ndim == 3:  # (C,H,W)
            C, H, W = a.shape
            flat = a.reshape(-1)
            if flat.size > 40:
                # pick top/bottom ~20
                idxs = np.argsort(flat)
                sel = np.concatenate([idxs[:20], idxs[-20:]])
            else:
                sel = np.arange(flat.size)
            outs = []
            for i in sel:
                val = float(flat[i])
                c = i // (H*W)
                r = (i % (H*W)) // W
                w = (i % (H*W)) % W
                outs.append((val, r*W + w, c))
            outliers.append(outs)
        else:
            # can't parse easily
            outliers.append([])

    # Prepare final stats dict
    stats = {
        "l2_mean":         act_L2_m,
        "l2_std":          act_L2_s,
        "raw_mean":        act_raw_m,
        "raw_std":         act_raw_s,
        "param_l2_mean":   par_L2_m,
        "param_l2_std":    par_L2_s,
        "param_raw_mean":  par_raw_m,
        "param_raw_std":   par_raw_s,
    }

    # Save to JSON
    safe_label = re.sub(r"[/\\\\]", "__", label)
    out_json = OUT_DIR / f"{safe_label}.json"
    out_json.write_text(json.dumps(stats))
    print(f"  [OK] Stats saved to {out_json}")

    # ───────── PLOTTING ─────────
    def _plot(vals, stds, ttl, ylab, suffix):
        xvals = range(1, len(vals) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        lower = [v - s for v, s in zip(vals, stds)]
        upper = [v + s for v, s in zip(vals, stds)]
        plt.fill_between(xvals, lower, upper, alpha=0.25)
        plt.plot(xvals, vals, marker="o")
        plt.title(ttl, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylab)
        plt.grid(ls="--", lw=0.4)
        plt.tight_layout()
        out_png = OUT_DIR / f"{safe_label}_{suffix}.png"
        plt.savefig(out_png)
        plt.close()
        print(f"  [OK] Plot saved to {out_png}")

    # Whisker box plot for raw activations
    def _plot_box(data, out_info):
        """
        data: list of 3D or 2D numpy arrays from each block's activation
        out_info: list of outliers in the form (value, paramIndex, channelIndex)
        """
        if not data:
            return

        # Flatten each block's activations
        block_vals = [d.flatten() for d in data]

        plt.figure(figsize=(12, 3 + 0.5 * len(block_vals)), dpi=300)
        bp = plt.boxplot(
            block_vals,
            vert=True,
            patch_artist=True,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "r",
                "alpha": 0.6,
            },
        )
        # Overplot specific outliers
        for i, outs in enumerate(out_info):
            if not outs:
                continue
            # x coordinate is i+1 for boxplot
            xs = np.full(len(outs), i+1)
            ys = [o[0] for o in outs]
            plt.scatter(xs, ys, color="red", marker="x", s=12)
            # add text if you want (comment out if too cluttered)
            for xpt, ypt, _, ch_id in zip(xs, ys, [o[1] for o in outs], [o[2] for o in outs]):
                plt.text(xpt + 0.2, ypt, f"d{ch_id}", fontsize=6, ha="left")

        plt.xlabel("Block")
        plt.ylabel("Activation values")
        plt.title(f"{label} raw activation distribution", weight="bold")
        plt.grid(True, ls="--", lw=0.4, axis="y")
        plt.tight_layout()
        out_png = OUT_DIR / f"{safe_label}_box.png"
        plt.savefig(out_png)
        plt.close()
        print(f"  [OK] Box plot saved to {out_png}")

    # Generate standard line plots
    _plot(act_L2_m, act_L2_s, f"{label} ||activation||₂", "L2 norm", "l2")
    _plot(act_raw_m, act_raw_s, f"{label} raw activation", "Activation", "raw")
    _plot(par_L2_m, par_L2_s, f"{label} ||params||₂", "Parameter L2", "param_l2")
    _plot(par_raw_m, par_raw_s, f"{label} raw params", "Parameter value", "param_raw")

    # Generate whisker box plot for raw activations with outliers
    _plot_box(activations, outliers)

    return stats


# ───────────────────────── DRIVER ─────────────────────────
if __name__ == "__main__":
    # Example usage: we do "top16" blocks or "full" depth. Adjust as needed:
    for depth_mode in ("top16", "full"):
        print(f"\n=== Depth mode: {depth_mode} ===")
        max_b = None if depth_mode == "full" else 16
        all_stats: Dict[str, Dict[str, List[float]]] = {}
        for net_tag, net_id in MODELS.items():
            try:
                s = analyse(net_tag, net_id, max_blocks=max_b)
                all_stats[net_id] = s
            except Exception as e:
                print(f"  [WARN] Skipped {net_id}: {e}")

        # Combine a line plot across all networks
        if not all_stats:
            print(f"No successful runs for {depth_mode} mode.")
            continue

        def _combine(metric: str, ylab: str, suffix: str, title: str):
            plt.figure(figsize=(12, 6), dpi=300)
            for mid, stats_dict in all_stats.items():
                vals = stats_dict[metric]
                xvals = range(1, len(vals) + 1)
                plt.plot(xvals, vals, label=mid, lw=1.4)
            plt.title(title, weight="bold")
            plt.xlabel("Block")
            plt.ylabel(ylab)
            plt.grid(ls="--", lw=0.4)
            plt.legend(fontsize="x-small")
            plt.tight_layout()
            out_png = OUT_DIR / f"combined_{suffix}_{depth_mode}.png"
            plt.savefig(out_png)
            plt.close()
            print(f"  [OK] Combined {suffix} plot saved to {out_png}")

        # Plot combined line charts of each metric
        _combine("l2_mean",        "L2 norm",          "l2",        "Blockwise Activation L2 (mean)")
        _combine("raw_mean",       "Activation",       "raw",       "Blockwise Raw Activation (mean)")
        _combine("param_l2_mean",  "Parameter L2",     "param_l2",  "Blockwise Parameter L2 (mean)")
        _combine("param_raw_mean", "Parameter value",  "param_raw", "Blockwise Raw Parameter (mean)")

    print("\nAnalysis complete.")
