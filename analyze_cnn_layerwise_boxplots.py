"""
Residual vs Dense vs Plain CNN – dual-depth study
=================================================
Generates layer-wise activation & parameter statistics for three ImageNet-
pretrained networks **twice** each:
  • TOP-16 blocks – equalises depth across models
  • FULL depth    – uses *every* block present

Outputs go to  viz/plots/act_analysis_cnn_box/
"""

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
IMAGE_PATH = BASE_DIR / "test.png"  # 224×224 RGB

MODELS: Dict[str, str] = {
    "resnet": "resnet18",
    "densenet": "densenet121",
    "cnn": "vgg16_bn",
}


# ─────────────────────── HELPERS ────────────────────────
def _transform(size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(size + 32),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _get_blocks(model: torch.nn.Module, tag: str):
    if tag == "resnet":
        return [b for layer in [model.layer1, model.layer2, model.layer3, model.layer4] for b in layer]
    if tag == "densenet":
        feats = model.features
        dbs = [feats.denseblock1, feats.denseblock2, feats.denseblock3, feats.denseblock4]
        return [b for db in dbs for b in db]  # DenseLayer only
    return [m for m in model.features if isinstance(m, torch.nn.Conv2d)]  # VGG convs


def _param_stats(m: torch.nn.Module) -> Tuple[float, float, float, float]:
    vecs = [p.detach().float().view(-1) for p in m.parameters() if p.requires_grad]
    if not vecs:
        return 0, 0, 0, 0
    flat = torch.cat(vecs)
    l2s = torch.stack([v.norm(2) for v in vecs])
    return l2s.mean().item(), l2s.std().item(), flat.mean().item(), flat.std().item()


def _load_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """
    Return an ImageNet-pre-trained torchvision model as nn.Module,
    regardless of torchvision version.

    1.  Try new-style API with the model-specific *Weights* enum.
    2.  Fallback to legacy `pretrained=True`.
    """
    fn = getattr(tvm, model_id)  # constructor

    # ➊ new API – enum is e.g. tvm.DenseNet121_Weights.DEFAULT
    enum_name = f"{model_id}_weights".lower()
    enum = next((getattr(tvm, n) for n in dir(tvm) if n.lower() == enum_name), None)
    if enum is not None and hasattr(enum, "DEFAULT"):
        try:
            return fn(weights=enum.DEFAULT).to(device).eval()
        except Exception:
            pass  # fall through

    # ➋ old API
    return fn(pretrained=True).to(device).eval()


# ─────────────────── ANALYSIS CORE ────────────────────
def analyse(tag: str, model_id: str, max_blocks: Optional[int]) -> Dict[str, List[float]]:
    label = f"{model_id}_{'full' if max_blocks is None else f'top{max_blocks}'}"
    print(f"\n=== Analysing {label} ===")

    # build & place on GPU/CPU
    try:
        model = _load_model(model_id, DEVICE)
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower():
            print("  > GPU OOM – using CPU …")
            model = _load_model(model_id, DEVICE).cpu().eval()
        else:
            raise
    param_dev = next(model.parameters()).device

    blocks = _get_blocks(model, tag)
    if not blocks:
        raise ValueError("No blocks found")

    if max_blocks is not None and len(blocks) > max_blocks:
        blocks = blocks[:max_blocks]

    # === data containers ===
    act_L2_m, act_L2_s, act_raw_m, act_raw_s = [], [], [], []
    par_L2_m, par_L2_s, par_raw_m, par_raw_s = [], [], [], []
    activations, outliers = [], []

    for b in blocks:  # parameter stats
        m1, s1, m2, s2 = _param_stats(b)
        par_L2_m.append(m1)
        par_L2_s.append(s1)
        par_raw_m.append(m2)
        par_raw_s.append(s2)

    # image tensor
    img = _transform()(Image.open(IMAGE_PATH).convert("RGB"))
    x = img.unsqueeze(0).to(param_dev)

    # forward hooks
    def _hook(_, __, out):
        h = out.detach().float()
        if h.ndim == 4:  # (B,C,H,W)
            h = h.squeeze(0)
            activations.append(h.cpu().numpy())
            L2 = h.flatten(1).norm(2, dim=0)
            act_L2_m.append(L2.mean().item())
            act_L2_s.append(L2.std().item())
            flat = h.flatten()
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())
        else:  # generic fallback
            flat = h.flatten()
            activations.append(h.squeeze(0).cpu().numpy())
            act_L2_m.append(h.norm(2, dim=1).mean().item())
            act_L2_s.append(h.norm(2, dim=1).std().item())
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())

    hdls = [b.register_forward_hook(_hook) for b in blocks]
    with torch.no_grad():
        _ = model(x)
    for h in hdls:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # outlier tagging
    for a in activations:
        if a.ndim == 3:  # (C,H,W)
            C, H, W = a.shape
            flat = a.flatten()
            idxs = np.argsort(flat)
            sel = np.concatenate([idxs[:20], idxs[-20:]]) if idxs.size > 40 else idxs
            outs = []
            for i in sel:
                c, r, w = np.unravel_index(int(i), (C, H, W))
                outs.append((float(a[c, r, w]), r * W + w, c))
            outliers.append(outs)
        else:
            outliers.append([])

    # === dump & plots ===
    safe = re.sub(r"[/\\\\]", "__", label)
    stats = {
        "l2_mean": act_L2_m,
        "l2_std": act_L2_s,
        "raw_mean": act_raw_m,
        "raw_std": act_raw_s,
        "param_l2_mean": par_L2_m,
        "param_l2_std": par_L2_s,
        "param_raw_mean": par_raw_m,
        "param_raw_std": par_raw_s,
    }
    (OUT_DIR / f"{safe}.json").write_text(json.dumps(stats))

    def _plot(vals, stds, ttl, ylab, tag):
        x = range(1, len(vals) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        plt.fill_between(x, [v - s for v, s in zip(vals, stds)], [v + s for v, s in zip(vals, stds)], alpha=0.25)
        plt.plot(x, vals, marker="o")
        plt.title(ttl, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylab)
        plt.grid(True, ls="--", lw=0.4)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{safe}_{tag}.png")
        plt.close()

    def _plot_box(data, outs):
        if not data:
            return
        plt.figure(figsize=(12, 0.5 * len(data) + 3), dpi=300)
        plt.boxplot(
            [d.flatten() for d in data],
            vert=True,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "r", "alpha": 0.6},
        )
        for i, o in enumerate(outs):
            if not o:
                continue
            xs = np.full(len(o), i + 1)
            ys = [v for v, _, _ in o]
            plt.scatter(xs, ys, c="red", marker="x")
            for x, y, pid, ch in zip(xs, ys, [p for _, p, _ in o], [c for _, _, c in o]):
                plt.text(x + 0.1, y, f"p{pid} [d{ch}]", fontsize=6, ha="left")
        plt.xlabel("Block")
        plt.ylabel("Activation value")
        plt.title(f"{label} activation distribution", weight="bold")
        plt.grid(True, ls="--", lw=0.4, axis="y")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{safe}_box.png")
        plt.close()

    _plot(act_L2_m, act_L2_s, f"{label} ||act||₂", "L2 norm", "l2")
    _plot(act_raw_m, act_raw_s, f"{label} raw act", "Activation", "raw")
    _plot(par_L2_m, par_L2_s, f"{label} ||θ||₂", "Parameter L2", "param_l2")
    _plot(par_raw_m, par_raw_s, f"{label} raw θ", "Parameter value", "param_raw")
    _plot_box(activations, outliers)

    return stats


# ─────────────────────── DRIVER ────────────────────────
for depth in ("top16", "full"):
    all_stats: Dict[str, Dict[str, List[float]]] = {}
    for tag, mid in MODELS.items():
        try:
            all_stats[mid] = analyse(tag, mid, None if depth == "full" else 16)
        except Exception as e:
            print(f"[WARN] {mid} ({depth}) skipped: {e}")

    if not all_stats:
        print(f"No successful runs for {depth}")
        continue

    def _combine(metric, ylab, fname, title):
        plt.figure(figsize=(12, 6), dpi=300)
        for mid, s in all_stats.items():
            plt.plot(range(1, len(s[metric]) + 1), s[metric], label=mid, lw=1.4)
        plt.title(title, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylab)
        plt.grid(True, ls="--", lw=0.4)
        plt.legend(fontsize="x-small")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{fname}_{depth}.png")
        plt.close()

    _combine("l2_mean", "L2 norm", "combined_l2", "Blockwise activation L2 (mean)")
    _combine("raw_mean", "Activation", "combined_raw", "Blockwise raw activation (mean)")
    _combine("param_l2_mean", "Parameter L2", "combined_param_l2", "Blockwise parameter L2 (mean)")
    _combine("param_raw_mean", "Parameter value", "combined_param_raw", "Blockwise raw parameter (mean)")

print("\nAnalysis complete.")
