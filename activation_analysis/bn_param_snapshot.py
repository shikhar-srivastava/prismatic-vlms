#!/usr/bin/env python3
# bn_param_snapshot.py
#
# Quick-and-dirty extractor for BatchNorm γ (weight) and β (bias)
# from the CIFAR-100 checkpoints produced by analyze_custom_resnet_densenet.py
# ----------------------------------------------------------------------

import argparse, json, numpy as np, torch
from pathlib import Path

# pull utilities straight from the main analysis script
from analyze_custom_resnet_densenet import FACTORY, _find_ckpt, CHK_DIR

def collect_bn_params(model):
    gammas, betas = [], []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            gammas.append(m.weight.detach().cpu().numpy())
            betas.append(m.bias.detach().cpu().numpy())
    return np.concatenate(gammas), np.concatenate(betas)

def summarise(tag, γ, β):
    return {
        "gamma": {
            "mean":   float(γ.mean()),
            "std":    float(γ.std()),
            ">1_cnt": int((γ > 1.0).sum()),
            "total":  int(γ.size)
        },
        "beta": {
            "mean": float(β.mean()),
            "std":  float(β.std())
        }
    }

def main(tags):
    all_stats = {}
    for tag in tags:
        ck = _find_ckpt(tag)
        if ck is None:
            print(f"[!] checkpoint for {tag} not found in {CHK_DIR}")
            continue
        model = FACTORY[tag]()          # fresh model skeleton
        model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)

        γ, β = collect_bn_params(model)
        stats = summarise(tag, γ, β)
        all_stats[tag] = stats

        g = stats["gamma"]
        print(f"{tag:11s} ─ γ μ={g['mean']:.3f}  σ={g['std']:.3f}  "
              f"{g['>1_cnt']}/{g['total']} (>{1.0})")

    # optional: dump everything to a JSON file for later plots
    out = Path("bn_stats.json")
    out.write_text(json.dumps(all_stats, indent=2))
    print(f"\n✓ stats written to {out.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=[
        "resnet14","resnet18","resnet34",
        "densenet14","densenet18","densenet34"
    ], help="model tags to inspect")
    args = ap.parse_args()
    main(args.tags)
