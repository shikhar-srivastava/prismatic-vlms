#!/usr/bin/env python3
# bn_effective_gain.py
#
# Compute γ / sqrt(var+eps) for every BatchNorm2d layer
# and show how many channels exceed common thresholds.

import argparse, json, torch, numpy as np
from pathlib import Path
from analyze_custom_resnet_densenet import FACTORY, _find_ckpt, CHK_DIR

def bn_gain_stats(m: torch.nn.BatchNorm2d):
    g   = m.weight.detach().cpu().numpy()
    var = m.running_var.detach().cpu().numpy()
    gain = g / np.sqrt(var + m.eps)
    return {
        "mean":   float(gain.mean()),
        "std":    float(gain.std()),
        ">1":     int((gain > 1.0).sum()),
        ">2":     int((gain > 2.0).sum()),
        "total":  int(gain.size)
    }

def main(tags):
    all_stats = {}
    for tag in tags:
        ck = _find_ckpt(tag)
        if ck is None:
            print(f"[!] missing checkpoint for {tag}")
            continue

        net = FACTORY[tag]()
        net.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)

        gains = [bn_gain_stats(m) for m in net.modules()
                                   if isinstance(m, torch.nn.BatchNorm2d)]

        # aggregate
        agg = {
            "avg_gain":   float(np.mean([g["mean"] for g in gains])),
            ">1_ratio":   sum(g[">1"] for g in gains) / sum(g["total"] for g in gains),
            ">2_ratio":   sum(g[">2"] for g in gains) / sum(g["total"] for g in gains),
        }
        all_stats[tag] = agg
        print(f"{tag:11s}  gain μ={agg['avg_gain']:.2f}  "
              f">1:{agg['>1_ratio']*100:5.1f}%  >2:{agg['>2_ratio']*100:4.1f}%")

    Path("bn_gain_stats.json").write_text(json.dumps(all_stats, indent=2))
    print("\n✓ detailed stats → bn_gain_stats.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=[
        "resnet14","resnet18","resnet34",
        "densenet14","densenet18","densenet34"])
    args = ap.parse_args()
    main(args.tags)
