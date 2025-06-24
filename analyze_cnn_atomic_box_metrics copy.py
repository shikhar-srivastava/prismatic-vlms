#!/usr/bin/env python3
# analyze_cnn_atomic_box_metrics.py
#
# PURPOSE  ▸ Layer‑wise activation & parameter statistics for multiple
#            pretrained CNNs (ResNet, DenseNet, VGG‑style).  **Atomic** means
#            every basic module (Conv, BN, Linear, …).  Input tensors are taken
#            exactly as the module receives them, so residual sums and dense
#            concatenations are naturally included.
#
# METRICS  ▸ |activation|   mean ± std
#          ▸ |activation|   Top‑1 %, 5 %, 10 %  (mean ± std across images)
#
# OUTPUT   ▸ *.json with numbers
#          ▸ *.png with beautiful plots (one per network)
#
# USAGE    ▸ python analyze_cnn_atomic_box_metrics.py
#
# DEPENDS  ▸ PyTorch ≥ 2.0, torchvision, matplotlib, Pillow, numpy
#
# NOTES    ▸ Place some *.jpg / *.png images in  ./test_images/
# ---------------------------------------------------------------------------

from __future__ import annotations
import gc, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR  = Path.cwd()
OUT_DIR   = BASE_DIR / "viz" / "plots" / "act_analysis_cnn_atomic_box_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        plt.style.use("default")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGES_DIR = BASE_DIR / "test_images"

MODELS: Dict[str, str] = dict(
    resnet="resnet18",
    densenet="densenet121",
    cnn="vgg16_bn",
)

# ───────────────────────── HELPERS ─────────────────────────
def _get_test_images(n: int = 8) -> List[Path]:
    paths = sorted(list(TEST_IMAGES_DIR.glob("*.png")) +
                   list(TEST_IMAGES_DIR.glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No images in {TEST_IMAGES_DIR}")
    return paths[:n]

def _transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# ───────────────────── LAYER DISCOVERY ─────────────────────
_ATOMIC_TYPES = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.Linear,
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.ReLU, torch.nn.ReLU6, torch.nn.GELU, torch.nn.SiLU,
    torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
    torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
    torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
    torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d,
)

def _layer_type(m: torch.nn.Module) -> str:
    """Human‑friendly type for annotation/legend."""
    t = type(m)
    if issubclass(t, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return "Conv"
    if issubclass(t, torch.nn.Linear):
        return "Linear"
    if issubclass(t, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        return "BN"
    if issubclass(t, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.GELU, torch.nn.SiLU)):
        return "ReLU"
    if issubclass(t, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
        return "MaxPool"
    if issubclass(t, (torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
                      torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d,
                      torch.nn.AdaptiveAvgPool3d)):
        return "AvgPool"
    if issubclass(t, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        return "Dropout"
    return t.__name__

def _get_atomic_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module, str]]:
    """
    Returns a list of tuples: (qualified_name, module, layer_type)
    Only modules of type _ATOMIC_TYPES are retained.
    """
    atomic = []
    for qname, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, _ATOMIC_TYPES):
            atomic.append((qname, module, _layer_type(module)))
    # Remove duplicates caused by shared modules (rare)
    seen = set()
    unique_atomic = []
    for qname, module, ltype in atomic:
        if module not in seen:
            unique_atomic.append((qname, module, ltype))
            seen.add(module)
    return unique_atomic

# ─────────────────── PARAMETER STATISTICS ──────────────────
def _param_stats(m: torch.nn.Module) -> Tuple[float, float]:
    """L2‑norm and std of raw weights (flattened)."""
    ps = list(m.parameters(recurse=False))
    if not ps:
        return 0.0, 0.0
    flat = torch.cat([p.detach().flatten().float() for p in ps])
    return flat.norm(p=2).item(), flat.std().item()

# ────────────────── ACTIVATION STATISTICS ──────────────────
def _compute_percentiles(abs_acts: List[np.ndarray]) -> Dict[str, float]:
    """
    abs_acts : list of 1‑D numpy arrays (already |activation|) for *one* layer,
               *one* image each element
    Returns mean & std of Top‑1 %, 5 %, 10 % values across the images.
    """
    if not abs_acts:
        return {k: 0.0 for k in
                ("top1_mean", "top1_std", "top5_mean", "top5_std",
                 "top10_mean", "top10_std")}
    perc_means, perc_stds = {}, {}
    for pct, key in zip((99, 95, 90), ("top1", "top5", "top10")):
        maxima = []
        for a in abs_acts:
            cutoff = np.percentile(a, pct)
            masked = a[a >= cutoff]
            maxima.append(masked.mean() if masked.size else 0.0)
        perc_means[f"{key}_mean"] = float(np.mean(maxima))
        perc_stds [f"{key}_std"]  = float(np.std (maxima))
    return {**perc_means, **perc_stds}

# ───────────────────────── PLOTTING ────────────────────────
def _plot(layer_types: List[str],
          mu: List[float], sigma: List[float],
          pct_stats: Dict[str, List[float]],
          title: str, safe_title: str) -> None:

    n = len(mu)
    x = np.arange(1, n + 1)

    colors = dict(top1="#D32F2F", top5="#FF8F00", top10="#FBC02D")
    markers = dict(top1="o",     top5="s",      top10="^")

    fig, ax = plt.subplots(figsize=(24, 10), dpi=300)

    # ‑‑ absolute mean ± std for *all* activations
    ax.plot(x, mu, color="#1976D2", marker=".", lw=2.5, ms=7,
            label="|activation| mean", alpha=0.9)
    ax.fill_between(x, np.array(mu)-np.array(sigma),
                       np.array(mu)+np.array(sigma),
                    color="#1976D2", alpha=0.15, linewidth=0)

    # ‑‑ Top‑k percent curves
    for tag in ("top1", "top5", "top10"):
        m = pct_stats[f"{tag}_mean"]
        s = pct_stats[f"{tag}_std"]
        ax.plot(x, m, color=colors[tag], marker=markers[tag], lw=2.5, ms=7,
                label=f"{tag.replace('top','Top ')}", alpha=0.9,
                markeredgecolor="white", markeredgewidth=1.3)
        ax.fill_between(x, np.array(m)-np.array(s), np.array(m)+np.array(s),
                        color=colors[tag], alpha=0.10, linewidth=0)

    # ‑‑ Style and decorations
    ax.set_xlabel("Atomic layer index", fontsize=15, fontweight="bold")
    ax.set_ylabel("Mean |activation|",  fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

    ax.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=13)

    # Colour‑coded layer‑type strip above x‑axis
    type_palette = {
        "Conv":"#3498DB", "BN":"#E74C3C", "ReLU":"#F39C12",
        "MaxPool":"#9B59B6", "AvgPool":"#1ABC9C", "Linear":"#27AE60",
        "Dropout":"#95A5A6"
    }
    ylim = ax.get_ylim()
    ymarker = ylim[1] + (ylim[1]-ylim[0])*0.05
    for i, lt in enumerate(layer_types):
        ax.scatter(x[i], ymarker, color=type_palette.get(lt, "#7F8C8D"),
                   marker="s", s=90, edgecolor="white", linewidth=1)
    ax.set_ylim(ylim[0], ymarker + (ylim[1]-ylim[0])*0.15)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Append small layer‑type legend
    for lt, col in type_palette.items():
        handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                  markerfacecolor=col, markersize=10,
                                  markeredgecolor="white", lw=0))
        labels.append(lt)
    ax.legend(handles, labels, ncol=2, fontsize=12, frameon=True,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    path = OUT_DIR / f"{safe_title}_abs_activation_metrics.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    ➜ Plot saved  ▸ {path}")

# ───────────────────────── ANALYSIS ────────────────────────
def analyse(tag: str, model_id: str,
            max_layers: Optional[int] = None) -> Dict[str, List[float]]:

    print(f"\n=== {model_id}  ({tag}) ===")

    # ── model
    model = tvm.get_model(model_id, weights="DEFAULT")
    model.eval().to(DEVICE)

    # ── atomic layers
    layers = _get_atomic_layers(model)
    if max_layers:
        layers = layers[:max_layers]
    print(f"    • {len(layers)} atomic layers")

    # ── parameter stats
    param_l2, param_std = zip(*(_param_stats(m) for _, m, _ in layers))

    # ── containers
    abs_means_all : List[List[float]] = []   # [img][layer]
    abs_vals_per_layer : List[List[np.ndarray]] = [ [] for _ in layers ]

    # ── forward hook
    def _hook(layer_idx: int):
        def fn(mod, inp, out):
            x = inp[0].detach()
            x = x.abs().flatten()           # |activation|
            mean = float(x.mean())
            abs_means_all[-1].append(mean)
            abs_vals_per_layer[layer_idx].append(x.cpu().numpy())
        return fn

    handles = []
    for idx, (_, m, _) in enumerate(layers):
        handles.append(m.register_forward_hook(_hook(idx)))

    # ── iterate images
    for img_p in _get_test_images():
        img = Image.open(img_p).convert("RGB")
        x   = _transform()(img).unsqueeze(0).to(DEVICE)
        abs_means_all.append([])                 # start new image row
        with torch.no_grad(): model(x)
        assert len(abs_means_all[-1]) == len(layers)

    # ── remove hooks
    for h in handles: h.remove()
    torch.cuda.empty_cache(); gc.collect()

    # ── aggregate mean/std over images per layer
    abs_means = np.array(abs_means_all)           # shape [Nimg, Nlayer]
    mu  = abs_means.mean(axis=0).tolist()
    std = abs_means.std (axis=0).tolist()

    # ── percentile stats per layer
    pct_combined : Dict[str, List[float]] = {
        "top1_mean":[], "top1_std":[],
        "top5_mean":[], "top5_std":[],
        "top10_mean":[], "top10_std":[],
    }
    for abs_list in abs_vals_per_layer:
        layer_stats = _compute_percentiles(abs_list)
        for k in pct_combined:
            pct_combined[k].append(layer_stats[k])

    # ── serialise and plot
    stats = dict(
        abs_mean=mu, abs_std=std,
        param_l2=param_l2, param_std=param_std,
        **pct_combined,
    )
    safe_lbl = re.sub(r"[\\/]", "__", model_id)
    (OUT_DIR / f"{safe_lbl}_metrics.json").write_text(json.dumps(stats, indent=2))
    print(f"    ➜ Numbers saved ▸ {safe_lbl}_metrics.json")

    _plot(
        layer_types=[lt for _, _, lt in layers],
        mu=mu, sigma=std, pct_stats=pct_combined,
        title=f"{model_id} — absolute activations",
        safe_title=safe_lbl,
    )
    return stats

# ───────────────────────── MAIN ────────────────────────────
def main():
    print(f"Output directory: {OUT_DIR}")
    print(f"Device: {DEVICE}\n")

    results = {}
    for tag, model_id in MODELS.items():
        try:
            stats = analyse(tag, model_id)
            results[model_id] = stats
        except Exception as e:
            print(f"[ERROR] {model_id}: {e}")

    (OUT_DIR / "all_atomic_layer_metrics.json").write_text(
        json.dumps(results, indent=2))
    print("\n✔ All done!")

if __name__ == "__main__":
    main()
