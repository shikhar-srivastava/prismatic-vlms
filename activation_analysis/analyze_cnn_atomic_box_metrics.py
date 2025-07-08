#!/usr/bin/env python3
# analyze_cnn_atomic_layer_metrics.py
#
# Atomic-layer forward-activation *and* backward-gradient analysis
# with precision-fitness badges and cross-model comparison plots.
#
# Author : <your-name>
# ----------------------------------------------------------------------

from __future__ import annotations
import gc, json, random, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR  = Path.cwd()
OUT_DIR   = BASE_DIR / "viz" / "plots" / "act_analysis_atomic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:                       plt.style.use("seaborn-v0_8-whitegrid")
except:                    plt.style.use("default")

DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGES_DIR = BASE_DIR / "test_images"

MODELS: Dict[str,str] = {
    "resnet"  : "resnet18",
    "densenet": "densenet121",
    "cnn"     : "vgg16_bn"
}

N_FWD_IMAGES  = 8
N_BW_BATCHES  = 10
BATCH_SIZE    = 64
FP16_LIMIT, BF16_LIMIT = 4.0, 0.5          # dotted-line thresholds

MODEL_COLOUR = {"resnet":"#D32F2F", "densenet":"#1976D2", "cnn":"#388E3C"}

# ───────────────────────── HELPERS ─────────────────────────
def _get_test_images()->List[Path]:
    imgs=list(TEST_IMAGES_DIR.glob("*.png"))+list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not imgs:
        raise ValueError(f"No test images found in {TEST_IMAGES_DIR}")
    return sorted(imgs)[:N_FWD_IMAGES]

def _transform(sz:int=224)->T.Compose:
    return T.Compose([
        T.Resize((sz,sz)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])

ATOMIC_TYPES = (
    nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU,
    nn.Sigmoid, nn.SiLU,
    nn.Linear,
    nn.MaxPool2d, nn.AvgPool2d,
    nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    nn.Dropout, nn.Identity
)

def _atomic_layers(model:nn.Module)->List[Tuple[str,nn.Module]]:
    """Return (qualified_name,module) for each *leaf* atomic layer."""
    return [(n,m) for n,m in model.named_modules()
            if len(list(m.children()))==0 and isinstance(m,ATOMIC_TYPES)]

def _layer_type(m:nn.Module)->str:
    return {
        nn.Conv2d:"Conv", nn.BatchNorm2d:"BN", nn.ReLU:"ReLU",
        nn.LeakyReLU:"ReLU", nn.Sigmoid:"Act", nn.SiLU:"Act",
        nn.Linear:"Linear",
        nn.MaxPool2d:"MaxPool", nn.AvgPool2d:"AvgPool",
        nn.AdaptiveAvgPool2d:"AdaptivePool",
        nn.AdaptiveMaxPool2d:"AdaptivePool",
        nn.Dropout:"Dropout", nn.Identity:"Identity"
    }[type(m)]

# ---------- stat helpers -----------------------------------------------------
def _param_stats(m:nn.Module)->Tuple[float,float,float,float]:
    ps=list(m.parameters())
    if not ps:
        return 0.0,0.0,0.0,0.0
    flat=torch.cat([p.detach().float().flatten() for p in ps])
    return flat.norm().item(),0.0,flat.mean().item(),flat.std().item()

def _percentile_stats(flat_items:List[List[np.ndarray]])->Dict[str,List[float]]:
    """
    flat_items : List[item][layer] where each element is a 1-D numpy array.
    Returns per-layer means/stds for top 1 / 5 / 10 % and all values.
    """
    if not flat_items:
        return {}
    n_layers=len(flat_items[0])

    def _agg(p:int):
        mu,sd=[],[]
        for li in range(n_layers):
            vecs=[it[li] for it in flat_items if li<len(it)]
            if not vecs:
                mu.append(0.0); sd.append(0.0); continue
            tops=[]
            for v in vecs:
                c=np.percentile(v,100-p)
                tops.append(v[v>=c].mean() if v[v>=c].size else 0.0)
            mu.append(float(np.mean(tops)))
            sd.append(float(np.std(tops)))
        return mu,sd

    t1m,t1s=_agg(1); t5m,t5s=_agg(5); t10m,t10s=_agg(10)
    return {
        "top1_mean":t1m, "top1_std":t1s,
        "top5_mean":t5m, "top5_std":t5s,
        "top10_mean":t10m, "top10_std":t10s
    }

def _precision_fitness(vecs:List[np.ndarray])->Tuple[float,float]:
    """Return percentage of |value| ≤ 0.5 (BF16) and ≤ 4 (FP16)."""
    concat=np.concatenate(vecs)
    total=concat.size
    bf16=(np.abs(concat)<=BF16_LIMIT).sum()/total*100
    fp16=(np.abs(concat)<=FP16_LIMIT).sum()/total*100
    return bf16,fp16

# ---------- hook helper to merge multiple calls of the same module ----------
def _collect_calls_to_layers(hooks_layers):
    """
    Return (accumulator_list, hook_fn).
    The accumulator holds a list per layer; the hook concatenates all calls.
    """
    layer_acc=[[] for _ in hooks_layers]
    id2idx={id(m):i for i,(_,m) in enumerate(hooks_layers)}

    def hook(mod, inp, _out):
        idx=id2idx[id(mod)]
        vec=torch.abs(inp[0].detach().squeeze(0)).flatten().cpu().numpy()
        layer_acc[idx].append(vec)
    return layer_acc, hook

# ---------- plotting ---------------------------------------------------------
def _annotate_thresholds(ax):
    for y,lab in ((FP16_LIMIT,"FP16"), (BF16_LIMIT,"BF16")):
        ax.axhline(y,ls=":",lw=1.8,color="#607D8B")
        ax.text(0.99, y*1.02 if y>1 else y*1.15,
                f"{lab}  •  Error ≤½-ULP (≈2⁻⁹)",
                ha="right", va="bottom", fontsize=9,
                color="#455A64", style="italic",
                transform=ax.get_yaxis_transform())

def _plot(stats:Dict[str,List[float]],
          layer_meta:List[Tuple[str,str]],
          tag:str,
          bf16_pct:float,
          fp16_pct:float,
          kind:str):            # 'act' or 'grad'
    x=np.arange(1,len(stats["abs_mean"])+1)

    fig,ax=plt.subplots(figsize=(26,10),dpi=300)

    series={
        "All |x|":("#1976D2","o","abs_mean","abs_std"),
        "Top 1 %":("#D32F2F","s","top1_mean","top1_std"),
        "Top 5 %":("#FFA000","^","top5_mean","top5_std"),
        "Top 10 %":("#FBC02D","D","top10_mean","top10_std")
    }
    for lbl,(col,mark,μk,σk) in series.items():
        μ,σ=stats[μk],stats[σk]
        ax.plot(x,μ,label=lbl,marker=mark,ms=6,lw=2,color=col,alpha=.9)
        ax.fill_between(x,np.array(μ)-np.array(σ),
                          np.array(μ)+np.array(σ),
                          color=col,alpha=.15,lw=0)

    _annotate_thresholds(ax)

    # call-out “hot” layers (top1_mean > 2×global mean)
    thr=2*np.mean(stats["top1_mean"])
    y_min,y_max=ax.get_ylim(); y_rng=y_max-y_min
    marker_y=y_max+y_rng*0.08

    TYPE_COLOR={"Conv":"#42A5F5","BN":"#EF5350","ReLU":"#FFB300",
                "Linear":"#66BB6A","MaxPool":"#8E24AA",
                "AvgPool":"#26A69A","AdaptivePool":"#26C6DA",
                "Dropout":"#BDBDBD","Identity":"#78909C","Act":"#FB8C00"}

    for idx,(lt,name) in enumerate(layer_meta,1):
        ax.scatter(idx,marker_y,
                   color=TYPE_COLOR.get(lt,"#9E9E9E"),
                   s=16,edgecolors="white",lw=0.6)
        if stats["top1_mean"][idx-1] > thr:
            ax.text(idx,marker_y - y_rng*0.04,
                    name.split(".")[-1],
                    rotation=45, ha="center", va="top",
                    fontsize=8, color="#37474F")

    # labels & title
    ax.set_xlabel("Layer index", fontsize=15, fontweight="bold")
    ylabel = "Absolute Input Activation Value" \
             if kind=="act" else "Absolute Parameter-Gradient Value"
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ttl = "Forward Activations" if kind=="act" else "Backward Gradients"
    ax.set_title(f"{tag}: {ttl} (mean ± std)",
                 fontsize=18, fontweight="bold")

    ax.grid(ls="--",alpha=.3)

    # main (series) legend – upper left
    series_legend = ax.legend(fontsize=11, loc="upper left")

    # layer-type legend – centre-right outside plot
    handles=[plt.Line2D([0],[0],marker='s',color='w',
             markerfacecolor=TYPE_COLOR[t],markersize=8,label=t)
             for t in sorted({lt for lt,_ in layer_meta})]
    layer_legend = ax.legend(handles=handles,
                             title="Layer type", fontsize=8,
                             title_fontsize=9, frameon=True,
                             loc="center left", bbox_to_anchor=(1.02,0.5),
                             borderpad=0.4)
    ax.add_artist(series_legend)   # keep both legends

    # precision-fitness badge (top-right inside axes)
    ax.text(0.99,0.98,
            f"≤{BF16_LIMIT:g} : {bf16_pct:5.1f}%\n"
            f"≤{FP16_LIMIT:g} : {fp16_pct:5.1f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="#ECEFF1",
                      edgecolor="#B0BEC5",
                      boxstyle="round,pad=0.3"))

    ax.set_ylim(y_min, marker_y + y_rng*0.15)
    plt.tight_layout()

    suffix="_act" if kind=="act" else "_grad"
    out=OUT_DIR/f"{tag}{suffix}_plot.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] plot → {out}")

# ---------- combined plots ---------------------------------------------------
def _combined_plot(all_stats:Dict[str,Dict[str,List[float]]],
                   metric_key:str,
                   fname:str,
                   title:str):
    fig,ax=plt.subplots(figsize=(20,9),dpi=300)
    for mdl,data in all_stats.items():
        μ=data[metric_key]
        ax.plot(np.arange(1,len(μ)+1), μ,
                label=mdl, lw=2, color=MODEL_COLOUR[mdl], alpha=.9)
    _annotate_thresholds(ax)
    ylabel = "Absolute Top 1 % mean" if "top1" in metric_key \
             else "Absolute mean of all values"
    ax.set_xlabel("Layer index", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=17)
    ax.grid(ls="--",alpha=.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR/fname, bbox_inches="tight", facecolor="white")
    plt.close()

def _make_combined_plots(act_stats, grad_stats):
    _combined_plot(act_stats,"top1_mean",
                   "combined_top1_percentile_act.png",
                   "Top 1 % Activations – model comparison")
    _combined_plot(act_stats,"abs_mean",
                   "combined_abs_mean_act.png",
                   "All |x| Activations – model comparison")
    _combined_plot(grad_stats,"top1_mean",
                   "combined_top1_percentile_grad.png",
                   "Top 1 % Gradients – model comparison")
    _combined_plot(grad_stats,"abs_mean",
                   "combined_abs_mean_grad.png",
                   "All |x| Gradients – model comparison")

# ---------- CIFAR-100 loader for backward pass -------------------------------
def _cifar_loader()->DataLoader:
    tf=T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.5071,0.4865,0.4409],
                    [0.2673,0.2564,0.2761])
    ])
    ds=dsets.CIFAR100(root=str(BASE_DIR/"data"),
                      train=True, download=True,
                      transform=tf)
    idxs=random.sample(range(len(ds)), k=N_BW_BATCHES*BATCH_SIZE)
    subset=torch.utils.data.Subset(ds, idxs)
    return DataLoader(subset, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2, pin_memory=True)

# ---------- model loader -----------------------------------------------------
def _load(model_id:str)->nn.Module:
    fn=getattr(tvm, model_id)
    try:    model=fn(weights=getattr(tvm,f"{model_id.upper()}_Weights").DEFAULT)
    except Exception:
        try:        model=fn(weights="DEFAULT")
        except Exception:model=fn(pretrained=True)
    return model.to(DEVICE).train(False)

# ---------- safe JSON helper -------------------------------------------------
def _py(obj):
    """Recursively cast NumPy scalars to Python scalars for JSON."""
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):      return int(obj)
    if isinstance(obj, list):  return [_py(x) for x in obj]
    if isinstance(obj, dict):  return {k:_py(v) for k,v in obj.items()}
    return obj

# ───────────────────────── ANALYSIS ─────────────────────────
def analyse(tag:str, model_id:str):
    print(f"\n=== {model_id} ===")
    model=_load(model_id)
    layers=_atomic_layers(model)
    print(f"  [INFO] {len(layers)} atomic layers")

    layer_meta=[(_layer_type(m),n) for n,m in layers]

    # ---------- forward activations ------------------------------------------
    act_flat_items=[]
    for img_path in _get_test_images():
        x=_transform()(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        layer_acc, hk_fn = _collect_calls_to_layers(layers)
        hdls=[m.register_forward_hook(hk_fn) for _,m in layers]
        with torch.no_grad(): _ = model(x)
        for h in hdls: h.remove()

        # concatenate multiple calls per module
        act_flat_items.append([
            np.concatenate(v) if v else np.array([0.0], np.float32)
            for v in layer_acc
        ])

    concat_layer_vecs=[np.concatenate([it[i] for it in act_flat_items])
                       for i in range(len(layers))]
    bf16_act, fp16_act = _precision_fitness(concat_layer_vecs)
    act_stats=_percentile_stats(act_flat_items)
    act_stats.update({
        "abs_mean":[float(v.mean()) for v in concat_layer_vecs],
        "abs_std" :[float(v.std())  for v in concat_layer_vecs]
    })
    _plot(act_stats, layer_meta,
          f"{model_id}_atomic",
          bf16_act, fp16_act, "act")

    # ---------- backward gradients ------------------------------------------
    grad_flat_items=[]
    loader=_cifar_loader()
    loss_fn=nn.CrossEntropyLoss().to(DEVICE)

    for b_ix,(x,y) in enumerate(loader):
        if b_ix>=N_BW_BATCHES: break
        x,y=x.to(DEVICE),y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        out=model(x)
        loss=loss_fn(out,y)
        loss.backward()

        per_layer=[]
        for _,m in layers:
            vecs=[p.grad.detach().abs().flatten().cpu()
                  for p in m.parameters(recurse=False)
                  if p.grad is not None]
            if vecs:
                per_layer.append(torch.cat(vecs).numpy())
            else:
                per_layer.append(np.array([0.0], np.float32))
        grad_flat_items.append(per_layer)

    concat_grad_vecs=[np.concatenate([it[i] for it in grad_flat_items])
                      for i in range(len(layers))]
    bf16_grad, fp16_grad=_precision_fitness(concat_grad_vecs)
    grad_stats=_percentile_stats(grad_flat_items)
    grad_stats.update({
        "abs_mean":[float(v.mean()) for v in concat_grad_vecs],
        "abs_std" :[float(v.std())  for v in concat_grad_vecs]
    })
    _plot(grad_stats, layer_meta,
          f"{model_id}_atomic",
          bf16_grad, fp16_grad, "grad")

    # ---------- return for combined plotting ---------------------------------
    return act_stats, grad_stats

# ───────────────────────── MAIN ─────────────────────────
if __name__=="__main__":
    print(f"Device: {DEVICE} | output: {OUT_DIR}")
    act_all, grad_all = {},{}
    for tag, model_id in MODELS.items():
        try:
            act, grad = analyse(tag, model_id)
            act_all[tag]  = act
            grad_all[tag] = grad
        except Exception as e:
            print(f"  [ERR] {tag}: {e}")

    _make_combined_plots(act_all, grad_all)

    # save all metrics to JSON
    (OUT_DIR/"all_atomic_layer_metrics.json").write_text(
        json.dumps(_py({"activations":act_all,
                        "gradients":grad_all}), indent=2)
    )
    print("\nAnalysis complete – see", OUT_DIR)
