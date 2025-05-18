"""
Activation L2‑Norm + Raw Activation Study
========================================
Collects per‑layer activation statistics (L2 norm & raw value) for a suite of open‑weight LLMs and writes:
• `<model>.json` ‑ persistent stats  • `<model>_l2.png`  • `<model>_raw.png`
Generates an overlay plot across all successfully‑analysed models.

This revision fixes path errors, adds Qwen3‑*‑Base and Pythia models, replaces DeepSeek IDs with **DeepSeek‑V2‑Lite**, and makes the layer‑finder generic so Gemma models work. It also falls back to CPU/offload when VRAM is tight.
"""
from __future__ import annotations
import gc, json, os
from pathlib import Path
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)

# ─────────────────────────────────────────────────────────  CONFIG  ─────────
BASE_DIR = Path.cwd()                   # absolute project root
OUT_DIR  = BASE_DIR / "viz" / "plots" / "act_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda:0"                 # single‑GPU env assumed
PROMPT      = "Give me a short introduction to large language models."
AUTO_INT8 = False  # quantization disabled – load models in default precision                     # quantise ≥6.9B checkpoints
CPU_OFFLOAD = True                     # offload if GPU OOM

# Model families → HF IDs ≤≈14B params
MODELS: Dict[str, List[str]] = {
    "deepseek_v2_lite": [
        "deepseek-ai/DeepSeek-V2-Lite",          # ≈2 B
    ],
    "gemma3": [                                  # Google Gemma‑3 family
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
    ],                                            # 12 B skipped by default (OOM)
    "qwen3": [                                   # Qwen3 Base checkpoints
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
        #   "Qwen/Qwen3-14B-Base",  # uncomment if you have >48 GB or CPU offload
    ],
    "pythia": [                                  # Pythia scaling suite
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

# ───────────────────────────────────────────────  UTILITIES  ───────────────

def _get_hidden(out):
    """Return tensor regardless of model‑specific output wrappers."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    raise TypeError(f"Unknown output type: {type(out)}")


def _find_transformer_blocks(model: torch.nn.Module):
    """Return a list of per‑layer transformer blocks for *any* decoder‑only LLM."""
    # common attribute names by family
    for path in [
        "model.layers", "transformer.h", "transformer.layers",
        "gpt_neox.layers", "layers", "decoder.layers",
    ]:
        cur = model
        found = True
        for seg in path.split('.'):
            if hasattr(cur, seg):
                cur = getattr(cur, seg)
            else:
                found = False; break
        if found and isinstance(cur, (torch.nn.ModuleList, list)) and len(cur):
            return list(cur)
    # fallback: heuristic – modules with a self‑attn attribute
    blocks = [m for m in model.modules() if hasattr(m, "self_attn") or hasattr(m,"self_attention")]
    return blocks if blocks else []

# ──────────────────────────────────────────  MAIN ANALYSIS FUNC  ────────────

def analyse_model(model_id: str) -> Dict[str, List[float]]:
    print(f"\n=== Analysing {model_id} ===")

    # decide quantisation / offload strategy
    big_checkpoint = any(k in model_id.lower() for k in ["14b", "12b", "8b", "6.9b"])
    use_int8  = AUTO_INT8 and big_checkpoint

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if use_int8:
        load_kwargs.update({
            "load_in_8bit": True,
            "quantization_config": BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True),
        })
        load_kwargs["device_map"] = "auto" if CPU_OFFLOAD else {"": 0}
    else:
        load_kwargs["device_map"] = {"": 0}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    except RuntimeError as oom:
        if CPU_OFFLOAD and "out of memory" in str(oom):
            print("  > GPU OOM – retrying with full CPU offload …")
            load_kwargs["device_map"] = "cpu"
            load_kwargs.pop("load_in_8bit", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
        else:
            raise

    # locate blocks generically
    blocks = _find_transformer_blocks(model)
    if not blocks:
        print("  ! Could not locate transformer blocks – skipping")
        raise ValueError("no_blocks")

    l2_mean, l2_std, raw_mean, raw_std = [], [], [], []

    def hook(_, __, output):
        h = _get_hidden(output).float()
        l2 = h.norm(p=2, dim=-1).flatten()
        l2_mean.append(l2.mean().item()); l2_std.append(l2.std().item())
        flat = h.flatten()
        raw_mean.append(flat.mean().item()); raw_std.append(flat.std().item())

    handles = [b.register_forward_hook(hook) for b in blocks]

    with torch.no_grad():
        _ = model(**tokenizer(PROMPT, return_tensors="pt").to(model.device), use_cache=False)

    for h in handles: h.remove()
    del model; torch.cuda.empty_cache(); gc.collect()

    safe = model_id.replace("/", "__")
    stats = {"l2_mean": l2_mean, "l2_std": l2_std, "raw_mean": raw_mean, "raw_std": raw_std}
    json_path = OUT_DIR / f"{safe}.json"
    with open(json_path, "w") as f: json.dump(stats, f)
    print("    Stats JSON →", json_path)

    def _plot(y_m, y_s, title, ylabel, tag):
        x = range(1, len(y_m)+1)
        plt.figure(figsize=(10,4), dpi=140)
        plt.fill_between(x, [m-s for m,s in zip(y_m,y_s)], [m+s for m,s in zip(y_m,y_s)], alpha=.25)
        plt.plot(x, y_m, marker='o', lw=2)
        plt.title(title, weight='bold'); plt.xlabel('Layer'); plt.ylabel(ylabel)
        plt.grid(True, ls='--', lw=.4); plt.tight_layout()
        p = OUT_DIR / f"{safe}_{tag}.png"; plt.savefig(p); plt.close(); print("    Plot →", p)

    _plot(l2_mean, l2_std, f"{model_id} ‖act‖₂", "L2 norm", "l2")
    _plot(raw_mean, raw_std, f"{model_id} raw act", "Activation", "raw")
    return stats

# ───────────────────────────────────────────────  DRIVER  ───────────────────
all_stats: Dict[str, Dict[str, List[float]]] = {}
for fam, mids in MODELS.items():
    for mid in mids:
        try:
            all_stats[mid] = analyse_model(mid)
        except Exception as e:
            print(f"[WARN] Skipped {mid}: {e}")

if all_stats:
    plt.figure(figsize=(12,6), dpi=150)
    for mid,s in all_stats.items():
        plt.plot(range(1,len(s["l2_mean"])+1), s["l2_mean"], label=mid, lw=1.4)
    plt.title("Layer‑wise ‖activation‖₂ (mean)", weight='bold'); plt.xlabel('Layer'); plt.ylabel('L2 norm')
    plt.grid(True, ls='--', lw=.4); plt.legend(fontsize='x-small', ncol=2); plt.tight_layout()
    combo = OUT_DIR / "combined_l2_raw.png"; plt.savefig(combo); plt.close(); print("\nCombined plot →", combo)
else:
    print("No successful runs; combined plot not generated.")

print("\n✓ Analysis complete.")
