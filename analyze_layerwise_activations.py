"""
Activation & Parameter Distribution Study
========================================
Collects per-layer statistics for a suite of open-weight LLMs:

* activation L2-norm / raw value
* **parameter** L2-norm / raw value
* histogram of activations per transformer block

The script writes `<model>.json` (persistent stats) and saves several
`<model>_*.png` plots.  A combined plot across all successful models is
generated as well.

Assumptions
-----------
The analysis is run in a single-GPU environment.  For inference with a single
token (position id of 1), the hidden states at each transformer block have
shape ``(1, 1, hidden_size)`` throughout the forward pass.  The histogram uses
all activations of the generated prompt; it checks for spikes along the
position dimension by aggregating activations for every position.

This revision adds parameter statistics and activation histograms, and clarifies
the expected tensor shapes during inference.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#   CONFIG
BASE_DIR = Path.cwd()  # absolute project root
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"  # single-GPU env assumed
PROMPT = "Give me a short introduction to large language models."
AUTO_INT8 = (
    False  # quantization disabled - load models in default precision                     # quantise >=6.9B checkpoints
)
CPU_OFFLOAD = True  # offload if GPU OOM

# Model families -> HF IDs <=~14B params
MODELS: Dict[str, List[str]] = {
    "deepseek_v2_lite": [
        "deepseek-ai/DeepSeek-V2-Lite",  # 2 B
    ],
    "gemma3": [  # Google Gemma3 family
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
    ],  # 12 B skipped by default (OOM)
    "qwen3": [  # Qwen3 Base checkpoints
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
        #   "Qwen/Qwen3-14B-Base",  # uncomment if you have >48 GB or CPU offload
    ],
    "qwen2": [
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2-7B",
    ],
    "qwen2_5": [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
    ],
    "meta_llama": [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b",
    ],
    "pythia": [  # Pythia scaling suite
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-12b",
    ],
}

#   UTILITIES


def _get_hidden(out):
    """Return tensor regardless of model-specific output wrappers."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    raise TypeError(f"Unknown output type: {type(out)}")


def _find_transformer_blocks(model: torch.nn.Module):
    """Return a list of per-layer transformer blocks for *any* decoder-only LLM."""
    # common attribute names by family
    for path in [
        "model.layers",
        "transformer.h",
        "transformer.layers",
        "gpt_neox.layers",
        "layers",
        "decoder.layers",
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
    # fallback: heuristic  modules with a selfattn attribute
    blocks = [m for m in model.modules() if hasattr(m, "self_attn") or hasattr(m, "self_attention")]
    return blocks if blocks else []


#   MAIN ANALYSIS FUNC


def analyse_model(model_id: str) -> Dict[str, List[float]]:
    print(f"\n=== Analysing {model_id} ===")

    # decide quantisation / offload strategy
    big_checkpoint = any(k in model_id.lower() for k in ["14b", "12b", "8b", "6.9b"])
    use_int8 = AUTO_INT8 and big_checkpoint

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if use_int8:
        load_kwargs.update(
            {
                "load_in_8bit": True,
                "quantization_config": BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True),
            }
        )
        load_kwargs["device_map"] = "auto" if CPU_OFFLOAD else {"": 0}
    else:
        load_kwargs["device_map"] = {"": 0}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    except RuntimeError as oom:
        if CPU_OFFLOAD and "out of memory" in str(oom):
            print("  > GPU OOM  retrying with full CPU offload ...")
            load_kwargs["device_map"] = "cpu"
            load_kwargs.pop("load_in_8bit", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
        else:
            raise

    # locate blocks generically
    blocks = _find_transformer_blocks(model)
    if not blocks:
        print("  ! Could not locate transformer blocks  skipping")
        raise ValueError("no_blocks")

    l2_mean, l2_std, raw_mean, raw_std = [], [], [], []
    p_l2_mean, p_l2_std, p_raw_mean, p_raw_std = [], [], [], []
    act_hist, hist_edges = [], None

    # parameter statistics per transformer block
    for b in blocks:
        l2s = [p.detach().float().norm(p=2).item() for p in b.parameters()]
        p_l2_mean.append(float(np.mean(l2s)))
        p_l2_std.append(float(np.std(l2s)))
        vals = torch.cat([p.detach().float().view(-1) for p in b.parameters()])
        p_raw_mean.append(vals.mean().item())
        p_raw_std.append(vals.std().item())

    # During inference with a single position id (seq_len=1), the activations
    # entering and exiting each transformer block have shape:
    #   (batch=1, seq_len=1, hidden_size)
    # Histogram is computed from the flattened activations of each block; this
    # aggregates all positions so spikes along the position dimension become
    # visible as heavy tails in the distribution.
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

    handles = [b.register_forward_hook(hook) for b in blocks]

    with torch.no_grad():
        _ = model(**tokenizer(PROMPT, return_tensors="pt").to(model.device), use_cache=False)

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
        plt.xlabel("Layer")
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
        plt.ylabel("Layer")
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
