#!/usr/bin/env python3
# analyze_cnn_atomic_layer_metrics.py
#
# High‑resolution, atomic‑layer activation analysis (absolute values)
# Author : <your‑name>

from __future__ import annotations
import gc, json, re, glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR  = Path.cwd()
OUT_DIR   = BASE_DIR / "viz" / "plots" / "act_analysis_atomic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:                       plt.style.use("seaborn-v0_8-whitegrid")
except:                    plt.style.use("default")

DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGES_DIR  = BASE_DIR / "test_images"
MODELS: Dict[str,str] = {"resnet":"resnet18",
                         "densenet":"densenet121",
                         "cnn":"vgg16_bn"}

# ───────────────────────── HELPERS ─────────────────────────
def _get_test_images(n_max:int=8)->List[Path]:
    imgs = list(TEST_IMAGES_DIR.glob("*.png"))+list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not imgs: raise ValueError(f"No test images in {TEST_IMAGES_DIR}")
    return sorted(imgs)[:n_max]

def _transform(sz:int=224)->T.Compose:
    return T.Compose([T.Resize((sz,sz)),T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ---------- layer discovery ----------
_ATOMIC_TYPES = (nn.Conv2d,nn.BatchNorm2d,nn.ReLU,nn.LeakyReLU,
                 nn.Sigmoid,nn.SiLU,nn.Linear,nn.MaxPool2d,nn.AvgPool2d,
                 nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d,
                 nn.Dropout,nn.Identity)

def _atomic_layers(model:nn.Module)->List[Tuple[str,nn.Module]]:
    """Return (qualified_name,module) for every leaf atomic layer."""
    layers=[]
    for name,mod in model.named_modules():
        if len(list(mod.children()))==0 and isinstance(mod,_ATOMIC_TYPES):
            layers.append((name,mod))
    return layers

def _layer_type(m:nn.Module)->str:
    return {nn.Conv2d:"Conv", nn.BatchNorm2d:"BN", nn.ReLU:"ReLU",
            nn.LeakyReLU:"ReLU", nn.Sigmoid:"Act", nn.SiLU:"Act",
            nn.Linear:"Linear", nn.MaxPool2d:"MaxPool",
            nn.AvgPool2d:"AvgPool", nn.AdaptiveAvgPool2d:"AdaptivePool",
            nn.AdaptiveMaxPool2d:"AdaptivePool",
            nn.Dropout:"Dropout", nn.Identity:"Identity"}[type(m)]

# ---------- stats helpers ----------
def _param_stats(m:nn.Module)->Tuple[float,float,float,float]:
    ps=list(m.parameters())
    if not ps: return (0.0,0.0,0.0,0.0)
    flat=torch.cat([p.detach().float().flatten() for p in ps])
    l2=flat.norm(p=2).item()
    return (l2,0.0,flat.mean().item(),flat.std().item())

def _percentile_stats(per_image_flat:List[List[np.ndarray]])->Dict[str,List[float]]:
    """List[image][layer] -> percentile stats (abs values)."""
    if not per_image_flat: return {}
    n_layers=len(per_image_flat[0])

    def agg(percentile:float):
        means,stds=[],[]
        for li in range(n_layers):
            vals=[img[li] for img in per_image_flat if li<len(img)]
            if not vals: means.append(0.0); stds.append(0.0); continue
            tops=[]
            for v in vals:
                cutoff=np.percentile(v,100-percentile)
                tops.append(v[v>=cutoff].mean() if v[v>=cutoff].size else 0.0)
            means.append(float(np.mean(tops))); stds.append(float(np.std(tops)))
        return means,stds

    top1m,top1s  = agg(1)
    top5m,top5s  = agg(5)
    top10m,top10s= agg(10)

    return {"top1_mean":top1m,"top1_std":top1s,
            "top5_mean":top5m,"top5_std":top5s,
            "top10_mean":top10m,"top10_std":top10s}

# ---------- plotting ----------
def _plot(stats:Dict[str,List[float]],
          layers_meta:List[Tuple[str,str]],  # (type,name)
          tag:str):
    if not stats.get("abs_mean"): return
    x=np.arange(1,len(stats["abs_mean"])+1)

    fig,ax=plt.subplots(figsize=(26,10),dpi=300)
    base={"abs":("#1976D2","o"),"top1":("#D32F2F","s"),
          "top5":("#FFA000","^"),"top10":("#FBC02D","D")}
    mapping={"abs":("abs_mean","abs_std"),
             "top1":("top1_mean","top1_std"),
             "top5":("top5_mean","top5_std"),
             "top10":("top10_mean","top10_std")}
    for key,(col,mark) in base.items():
        mu,std = stats[mapping[key][0]], stats[mapping[key][1]]
        ax.plot(x,mu,label=key.replace("abs","All |x|"),marker=mark,
                markersize=6,linewidth=2,color=col,alpha=.9)
        ax.fill_between(x,np.array(mu)-np.array(std),
                          np.array(mu)+np.array(std),
                          color=col,alpha=.15,linewidth=0)

    ax.set_xlabel("Layer index",fontsize=15,fontweight="bold")
    ax.set_ylabel("Absolute Input Activation Value",fontsize=15,fontweight="bold")
    ax.set_title(f"{tag}: Absolute Activation Distributions (mean ± std)",
                 fontsize=18,fontweight="bold")
    ax.grid(True,ls="--",alpha=.3)
    ax.legend(fontsize=12)
    # annotate a small colour dot signalling layer type
    y_max=ax.get_ylim()[1]
    y_annot=y_max*1.05
    type_col={"Conv":"#42A5F5","BN":"#EF5350","ReLU":"#FFB300","Linear":"#66BB6A",
              "MaxPool":"#8E24AA","AvgPool":"#26A69A","AdaptivePool":"#26C6DA",
              "Dropout":"#BDBDBD","Identity":"#78909C","Act":"#FB8C00"}
    for idx,(lt,_) in enumerate(layers_meta,1):
        ax.scatter(idx,y_annot,color=type_col.get(lt,"#9E9E9E"),s=15)
    ax.set_ylim(ax.get_ylim()[0],y_annot*1.08)
    plt.tight_layout()
    out=OUT_DIR/f"{tag}_abs_activation_plot.png"
    fig.savefig(out,bbox_inches="tight",facecolor="white")
    plt.close()
    print(f"  [OK] plot → {out}")

# ---------- model loading ----------
def _load(model_id:str,device)->nn.Module:
    fn=getattr(tvm,model_id)
    try:
        w_enum=getattr(tvm,f"{model_id.upper()}_Weights").DEFAULT
        model=fn(weights=w_enum)
    except Exception:               model=fn(weights="DEFAULT")
    except Exception:               model=fn(pretrained=True)
    model.to(device).eval()
    return model

# ───────────────────────── ANALYSIS ─────────────────────────
def analyse(tag:str,model_id:str,max_layers:Optional[int]=None)->Dict[str,List[float]]:
    print(f"\n=== {model_id} (atomic) ===")
    model=_load(model_id,DEVICE)
    layers=_atomic_layers(model)
    if max_layers: layers=layers[:max_layers]
    print(f"  [INFO] {len(layers)} atomic layers found")

    # parameter stats
    par_l2_m,par_l2_s,par_raw_m,par_raw_s=[],[],[],[]
    for _,m in layers:
        l2,_,rm,rs=_param_stats(m)
        par_l2_m.append(l2); par_l2_s.append(0.0)
        par_raw_m.append(rm);par_raw_s.append(rs)

    per_image_flat:List[List[np.ndarray]]=[]
    abs_mean,abs_std=[],[]

    for img_ix,img_path in enumerate(_get_test_images()):
        img   = Image.open(img_path).convert("RGB")
        tensor= _transform()(img).unsqueeze(0).to(DEVICE)

        flat_list=[]; img_abs_mu=[]; img_abs_sd=[]
        def _hook(mod,in_t,_out):
            t = in_t[0].detach()
            t_abs=torch.abs(t).squeeze(0)          # remove batch
            v   = t_abs.flatten().cpu().numpy()
            flat_list.append(v)
            img_abs_mu.append(float(v.mean()))
            img_abs_sd.append(float(v.std()))
        hdls=[m.register_forward_hook(_hook) for _,m in layers]
        with torch.no_grad(): _ = model(tensor)
        for h in hdls: h.remove()

        # aggregate absolute μ/σ per layer (running average)
        if img_ix==0:
            abs_mean,abs_std=img_abs_mu,img_abs_sd
        else:
            for i in range(len(img_abs_mu)):
                abs_mean[i]=(abs_mean[i]*img_ix+img_abs_mu[i])/(img_ix+1)
                abs_std[i] =(abs_std[i] *img_ix+img_abs_sd[i]) /(img_ix+1)

        per_image_flat.append(flat_list)

    torch.cuda.empty_cache(); gc.collect()

    pct=_percentile_stats(per_image_flat)
    stats={"abs_mean":abs_mean,"abs_std":abs_std,
           "param_l2_mean":par_l2_m,"param_l2_std":par_l2_s,
           "param_raw_mean":par_raw_m,"param_raw_std":par_raw_s,**pct}

    safe=re.sub(r"[/\\\\]", "__", f"{model_id}_atomic")
    (OUT_DIR/f"{safe}_metrics.json").write_text(json.dumps(stats,indent=2))
    layer_meta=[(_layer_type(m),n) for n,m in layers]
    _plot(stats,layer_meta,safe)
    return stats

# ───────────────────────── MAIN ─────────────────────────
if __name__=="__main__":
    print(f"Device: {DEVICE} | output: {OUT_DIR}")
    all_results={}
    for tag,mid in MODELS.items():
        try:
            res=analyse(tag,mid)
            all_results[f"{tag}_atomic"]=res
        except Exception as e:
            print(f"  [ERR] {tag}: {e}")
    (OUT_DIR/"all_atomic_layer_metrics.json").write_text(json.dumps(all_results,indent=2))
    print("\nAnalysis complete.")
