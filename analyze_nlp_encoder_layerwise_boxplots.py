"""
Encoder Transformer Activation & Parameter Study
================================================
Collects per-layer statistics for common encoder-based NLP models using
`transformers`.

This mirrors `analyze_layerwise_boxplots.py`, replacing decoder-only LLMs with
encoder-only architectures. Statistics computed per encoder block:

* activation L2-norm / raw value
* **parameter** L2-norm / raw value
* box-and-whisker plot of activations per block

A JSON file with statistics and several plots are written to
`viz/plots/act_analysis_encoder_box/`. A combined plot across all successful
models is generated at the end.

Assumptions
-----------
* A single-GPU environment is used.
* Models are loaded via `AutoModel` with pretrained weights.
* A single text prompt is used for inference.
* Activations exiting each block have shape ``(1, seq_len, hidden_size)``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

#   CONFIG
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_encoder_box"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

DEVICE = "cuda:0"
PROMPT = "Give me a short introduction to large language models."

MODELS: Dict[str, List[str]] = {
    "bert": ["bert-base-uncased"],
    "roberta": ["roberta-base"],
    "deberta": ["microsoft/deberta-v3-base"],
    "electra": ["google/electra-base-discriminator"],
}


#   UTILITIES


def _get_hidden(out: torch.Tensor) -> torch.Tensor:
    """Return tensor regardless of model-specific output wrappers."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    raise TypeError(f"Unknown output type: {type(out)}")


def _find_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate per-layer transformer blocks for BERT-like models."""
    for path in [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "deberta.encoder.layer",
        "electra.encoder.layer",
        "model.encoder.layer",
        "transformer.encoder.layer",
    ]:
        cur = model
        found = True
        for seg in path.split("."):
            if hasattr(cur, seg):
                cur = getattr(cur, seg)
            else:
                found = False
                break
        if found and isinstance(cur, (torch.nn.ModuleList, list)) and len(cur):
            return list(cur)
    # fallback: modules with self-attention attribute
    blocks = [m for m in model.modules() if hasattr(m, "attention")]
    return blocks if blocks else []


#   MAIN ANALYSIS FUNC


def analyse_model(model_id: str) -> Dict[str, List[float]]:
    print(f"\n=== Analysing {model_id} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()

    blocks = _find_transformer_blocks(model)
    if not blocks:
        print("  ! Could not locate transformer blocks  skipping")
        raise ValueError("no_blocks")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    token_ids = inputs["input_ids"][0].tolist()
    decoded_tokens = [tokenizer.decode(tid) for tid in token_ids]

    l2_mean, l2_std, raw_mean, raw_std = [], [], [], []
    p_l2_mean, p_l2_std, p_raw_mean, p_raw_std = [], [], [], []
    layer_acts: List[np.ndarray] = []
    layer_outliers: List[List[tuple]] = []

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
        layer_acts.append(flat.cpu().numpy())

    handles = [b.register_forward_hook(hook) for b in blocks]

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    seq_len = len(token_ids)
    for arr in layer_acts:
        mean = arr.mean()
        std = arr.std()
        hid_size = arr.size // seq_len
        mask = (arr > mean + 4 * std) | (arr < mean - 4 * std)
        idxs = np.where(mask)[0]
        outs: List[tuple] = []
        for i in idxs:
            tok_idx = int(i) // hid_size
            hid_idx = int(i) % hid_size
            val = float(arr[i])
            outs.append((val, decoded_tokens[tok_idx], hid_idx))
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
    print("    Stats JSON ", json_path)

    def _plot(y_m, y_s, title, ylabel, tag):
        x = range(1, len(y_m) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        plt.fill_between(x, [m - s for m, s in zip(y_m, y_s)], [m + s for m, s in zip(y_m, y_s)], alpha=0.25)
        plt.plot(x, y_m, marker="o", lw=2)
        plt.title(title, weight="bold")
        plt.xlabel("Layer")
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
            for x, y, tok, dim in zip(xs, ys, [t for _, t, _ in outs], [d for _, _, d in outs]):
                plt.text(x + 0.1, y, f"{tok} [d{dim}]", fontsize=6, ha="left", va="center")
        plt.xlabel("Layer")
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
    _plot_box(layer_acts, layer_outliers)
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
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.grid(True, ls="--", lw=0.4)
        plt.legend(fontsize="x-small", ncol=2)
        plt.tight_layout()
        path = OUT_DIR / fname
        plt.savefig(path)
        plt.close()
        print("\nCombined plot ", path)

    _combined_plot("l2_mean", "L2 norm", "combined_l2.png", "Layerwise activation L2 (mean)")
    _combined_plot("raw_mean", "Activation", "combined_raw.png", "Layerwise raw activation (mean)")
    _combined_plot("param_l2_mean", "Parameter L2 norm", "combined_param_l2.png", "Layerwise parameter L2 (mean)")
    _combined_plot("param_raw_mean", "Parameter value", "combined_param_raw.png", "Layerwise raw parameter (mean)")
else:
    print("No successful runs; combined plot not generated.")

print("\n Analysis complete.")
