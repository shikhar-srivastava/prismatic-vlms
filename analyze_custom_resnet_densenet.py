#!/usr/bin/env python3
# analyze_custom_resnet_densenet.py
#
# – Loads / (optionally trains) ResNet‑, Plain‑ and DenseNet‑variants that are
#   bit‑for‑bit identical to the “Outdated Script”.
# – Performs forward‑activation *and* backward‑gradient analysis at atomic layer
#   level, then produces depth‑specific comparison plots.
# ------------------------------------------------------------------------------

from __future__ import annotations
import argparse, json, random, gc, re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from PIL import Image

# ═════════════════ CONFIG ═════════════════
BASE       = Path.cwd()
CHK_DIR    = BASE / "checkpoints";  CHK_DIR.mkdir(exist_ok=True)
PLOT_DIR   = BASE / "viz/plots/act_analysis_atomic_custom_resnet_densenet"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR   = BASE / "datasets";     DATA_DIR.mkdir(exist_ok=True)
TEST_DIR   = BASE / "test_images"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

FP16_LIMIT, BF16_LIMIT = 4.0, 0.5

MODEL_COLOUR = {
    "resnet"  : "#D32F2F",
    "plain"   : "#7B1FA2",
    "densenet": "#1976D2",
}

# ═════════════ CLI ═════════════
ap = argparse.ArgumentParser()
ap.add_argument("--train", action="store_true",
                help="re‑train any model whose checkpoint is missing")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch",  type=int, default=128)
ap.add_argument("--grad-batches", type=int, default=10)
args = ap.parse_args()

# ═══════════ Architecture definitions (EXACT copy of ‘Outdated Script’) ═══════════
# ---------- Residual / Plain blocks ----------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.downsample: nn.Module|None = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self,x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)

class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c:int, out_c:int, stride:int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,3,stride,1,bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,1,bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

    def forward(self,x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out)

# ---------- ResNet / PlainNet ----------------------------------------------------
class _ResNetBase(nn.Module):
    def __init__(self, block, layers: List[int], num_classes:int=100):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.act   = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(block, 64,  layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_c, blocks, stride):
        seq = [block(self.in_c, out_c, stride)]
        self.in_c = out_c * block.expansion
        for _ in range(1, blocks):
            seq.append(block(self.in_c, out_c))
        return nn.Sequential(*seq)

    def forward(self,x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet14() -> nn.Module:  return _ResNetBase(BasicBlock, [3,2,2,2])
def resnet18() -> nn.Module:  return _ResNetBase(BasicBlock, [2,2,2,2])
def resnet34() -> nn.Module:  return _ResNetBase(BasicBlock, [3,4,6,3])

def plain14()  -> nn.Module:  return _ResNetBase(PlainBlock, [3,2,2,2])
def plain18()  -> nn.Module:  return _ResNetBase(PlainBlock, [2,2,2,2])
def plain34()  -> nn.Module:  return _ResNetBase(PlainBlock, [3,4,6,3])

# ---------- DenseNet (identical growth‑rate / compression settings) --------------
class _DenseLayer(nn.Module):
    def __init__(self,in_f,gr,bn_size=4,drop=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_f)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_f, bn_size*gr, 1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size*gr)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*gr, gr, 3,1,1,bias=False)

        self.drop = drop
    def forward(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.cat(x,1)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop>0: out = F.dropout(out,self.drop,self.training)
        return out

class _DenseBlock(nn.Module):
    def __init__(self,n,in_f,gr,bn_size=4,drop=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(in_f+i*gr,gr,bn_size,drop) for i in range(n)
        ])
    def forward(self,x):
        feats=[x]
        for l in self.layers:
            y=l(feats)
            feats.append(y)
        return torch.cat(feats,1)

class _Transition(nn.Module):
    def __init__(self,in_f,out_f):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_f)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_f,out_f,1,bias=False)
        self.pool = nn.AvgPool2d(2,2)
    def forward(self,x): return self.pool(self.conv(self.relu(self.norm(x))))

class DenseNet(nn.Module):
    def __init__(self,cfg:tuple, gr:int, comp:float, num_classes:int=100):
        super().__init__()
        init_f = 64
        self.features = nn.Sequential(
            nn.Conv2d(3,init_f,7,2,3,bias=False),
            nn.BatchNorm2d(init_f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        n_f = init_f
        for i,n in enumerate(cfg):
            blk=_DenseBlock(n,n_f,gr)
            self.features.add_module(f'db{i+1}',blk)
            n_f += n*gr
            if i != len(cfg)-1:
                out=int(n_f*comp)
                tr=_Transition(n_f,out)
                self.features.add_module(f'tr{i+1}',tr)
                n_f = out
        self.features.add_module('norm_final', nn.BatchNorm2d(n_f))
        self.classifier = nn.Linear(n_f, num_classes)

    def forward(self,x):
        x = self.features(x)
        x = F.relu(x,inplace=True)
        x = F.adaptive_avg_pool2d(x,1).flatten(1)
        return self.classifier(x)

def densenet14() -> nn.Module: return DenseNet((3,4,4,3), 64, 0.8)
def densenet18() -> nn.Module: return DenseNet((4,4,4,6), 64, 0.8)
def densenet34() -> nn.Module: return DenseNet((6,8,12,8), 64, 0.75)

# ---------- factory --------------------------------------------------------------
FACTORY = {
    "resnet14":resnet14, "resnet18":resnet18, "resnet34":resnet34,
    "plain14" :plain14,  "plain18" :plain18,  "plain34" :plain34,
    "densenet14":densenet14, "densenet18":densenet18, "densenet34":densenet34
}

# ═══════════ checkpoint helpers ═══════════
def _ckpt_variants(tag:str):
    return [CHK_DIR/f"{tag}.pth",
            CHK_DIR/f"{tag}.pt",
            CHK_DIR/f"cifar100_{tag}.pth",
            CHK_DIR/f"cifar100_{tag}.pt"]

def _find_ckpt(tag)->Path|None:
    for p in _ckpt_variants(tag):
        if p.exists(): return p
    return None

# ---------- CIFAR‑100 loaders ----------------------------------------------------
def _cifar_loader(train:bool,batch:int):
    mean,std=[0.5071,0.4865,0.4409],[0.2673,0.2564,0.2761]
    tf_train=T.Compose([
        T.RandomResizedCrop(224),T.RandomHorizontalFlip(),
        T.ToTensor(),T.Normalize(mean,std)])
    tf_val=T.Compose([T.Resize(224),T.ToTensor(),T.Normalize(mean,std)])
    ds=dsets.CIFAR100(DATA_DIR,train=train,download=True,
                      transform=tf_train if train else tf_val)
    return DataLoader(ds,batch_size=batch,shuffle=train,
                      num_workers=2,pin_memory=True)

# ---------- (re‑)training --------------------------------------------------------
def train_if_needed(tag:str):
    ck=_find_ckpt(tag)
    if ck and not args.train:
        print(f"[✓] using existing checkpoint {ck.name}")
        return ck
    if ck and args.train: print(f"[i] retraining will overwrite {ck.name}")
    if not args.train:
        raise FileNotFoundError(f"{tag}: checkpoint missing – run with --train")

    print(f"[⋯] Training {tag} for {args.epochs} epochs")
    model=FACTORY[tag]().to(DEVICE)
    opt=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[args.epochs//2,args.epochs*3//4])
    loss=nn.CrossEntropyLoss()
    loader=_cifar_loader(True,args.batch)
    model.train()
    for ep in range(1,args.epochs+1):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            loss(model(x),y).backward()
            opt.step()
        sched.step()
        if ep%5==0 or ep==args.epochs:
            print(f"  epoch {ep}/{args.epochs}")
    ck=_ckpt_variants(tag)[0]  # prefer .pth
    torch.save(model.state_dict(),ck)
    print(f"[✓] saved → {ck}")
    return ck

# ═══════════ analysis utilities (unchanged) ═══════════
ATOMIC_TYPES=(nn.Conv2d,nn.BatchNorm2d,nn.ReLU,nn.LeakyReLU,
              nn.Sigmoid,nn.SiLU,nn.Linear,nn.MaxPool2d,nn.AvgPool2d,
              nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d,nn.Dropout,nn.Identity)

def _atomic_layers(m): return [(n,l) for n,l in m.named_modules()
                               if len(list(l.children()))==0 and isinstance(l,ATOMIC_TYPES)]

def _test_imgs(n=8):
    imgs=list(TEST_DIR.glob("*.png"))+list(TEST_DIR.glob("*.jpg"))
    if not imgs: raise RuntimeError(f"No images in {TEST_DIR}")
    return sorted(imgs)[:n]

def _precision_fitness(vecs):
    a=np.concatenate(vecs); tot=a.size
    return (np.abs(a)<=BF16_LIMIT).sum()/tot*100,(np.abs(a)<=FP16_LIMIT).sum()/tot*100

def _collect(layers):
    acc=[[] for _ in layers]; id2={id(m):i for i,(_,m) in enumerate(layers)}
    def hk(mod,inp,_out):
        v=torch.abs(inp[0].detach().squeeze(0)).flatten().cpu().numpy()
        acc[id2[id(mod)]].append(v)
    return acc,hk

def _pct(items):
    n=len(items[0])
    def agg(p):
        μ,σ=[],[]
        for i in range(n):
            vecs=[it[i] for it in items]
            tops=[(v[v>=np.percentile(v,100-p)].mean() if v.size else 0.0)
                  for v in vecs]
            μ.append(float(np.mean(tops))); σ.append(float(np.std(tops)))
        return μ,σ
    t1m,t1s=agg(1); t5m,t5s=agg(5); t10m,t10s=agg(10)
    return {"top1_mean":t1m,"top1_std":t1s,
            "top5_mean":t5m,"top5_std":t5s,
            "top10_mean":t10m,"top10_std":t10s}

# ---------- plotting helpers (unchanged – colours extended) ----------------------
def _plot(stats,meta,tag,bf16,fp16,kind):
    x=np.arange(1,len(stats["abs_mean"])+1)
    fig,ax=plt.subplots(figsize=(26,10),dpi=300)
    series={"All |x|":("#1976D2","o","abs_mean","abs_std"),
            "Top 1 %":("#D32F2F","s","top1_mean","top1_std"),
            "Top 5 %":("#FFA000","^","top5_mean","top5_std"),
            "Top 10 %":("#FBC02D","D","top10_mean","top10_std")}
    for lbl,(c,m,μk,σk) in series.items():
        μ,σ=stats[μk],stats[σk]
        ax.plot(x,μ,label=lbl,marker=m,lw=2,ms=6,color=c)
        ax.fill_between(x,np.array(μ)-np.array(σ),np.array(μ)+np.array(σ),
                        color=c,alpha=.15)
    for y,lab in ((FP16_LIMIT,"FP16"),(BF16_LIMIT,"BF16")):
        ax.axhline(y,ls=":",lw=1.8,color="#607D8B")
        ax.text(0.99,y*(1.02 if y>1 else 1.15),
                f"{lab} • Error ≤½‑ULP (≈2⁻⁹)",
                transform=ax.get_yaxis_transform(),
                ha="right",va="bottom",fontsize=9,color="#455A64")
    thr=2*np.mean(stats["top1_mean"])
    y_min,y_max=ax.get_ylim(); y_rng=y_max-y_min; mark_y=y_max+y_rng*0.08
    TYPE_COL={"Conv":"#42A5F5","BN":"#EF5350","ReLU":"#FFB300",
              "Linear":"#66BB6A","MaxPool":"#8E24AA","AvgPool":"#26A69A",
              "AdaptivePool":"#26C6DA","Dropout":"#BDBDBD","Identity":"#78909C","Act":"#FB8C00"}
    for i,(lt,name) in enumerate(meta,1):
        ax.scatter(i,mark_y,color=TYPE_COL.get(lt,"#9E9E9E"),
                   s=16,edgecolors="white",lw=0.6)
        if stats["top1_mean"][i-1]>thr:
            ax.text(i,mark_y-y_rng*0.04,name.split(".")[-1],
                    rotation=45,ha="center",va="top",fontsize=8,color="#37474F")
    ax.set_xlabel("Layer index",fontsize=15,fontweight="bold")
    ax.set_ylabel("Absolute Input Activation Value" if kind=="act"
                  else "Absolute Parameter‑Gradient Value",
                  fontsize=15,fontweight="bold")
    ax.set_title(f"{tag}: {'Activations' if kind=='act' else 'Gradients'} (mean ± std)",
                 fontsize=18,fontweight="bold")
    ax.grid(ls="--",alpha=.3)
    series_leg=ax.legend(fontsize=11,loc="upper left")
    handles=[plt.Line2D([0],[0],marker='s',color='w',
             markerfacecolor=TYPE_COL[t],markersize=8,label=t)
             for t in sorted({lt for lt,_ in meta})]
    ax.legend(handles=handles,title="Layer type",fontsize=8,
              title_fontsize=9,loc="center left",bbox_to_anchor=(1.02,0.5))
    ax.add_artist(series_leg)
    ax.text(0.99,0.98,f"≤{BF16_LIMIT}: {bf16:5.1f}%\n≤{FP16_LIMIT}: {fp16:5.1f}%",
            transform=ax.transAxes,ha="right",va="top",
            fontsize=10,fontweight="bold",
            bbox=dict(facecolor="#ECEFF1",edgecolor="#B0BEC5",
                      boxstyle="round,pad=0.3"))
    ax.set_ylim(y_min,mark_y+y_rng*0.15)
    plt.tight_layout()
    out=PLOT_DIR/f"{tag}_{kind}.png"
    fig.savefig(out,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"[✓] plot → {out}")

# ═══════════ single‑model pipeline ═══════════
def analyse(tag):
    ck=train_if_needed(tag)
    model=FACTORY[tag]().to(DEVICE)
    state=torch.load(ck,map_location="cpu")
    model.load_state_dict(state, strict=False)      # identical key‑set now
    model.eval()

    layers=_atomic_layers(model)
    meta=[(("Conv" if isinstance(m,nn.Conv2d) else
            "BN" if isinstance(m,nn.BatchNorm2d) else
            "ReLU" if isinstance(m,(nn.ReLU,nn.LeakyReLU)) else
            "Linear" if isinstance(m,nn.Linear) else
            "MaxPool" if isinstance(m,nn.MaxPool2d) else
            "AvgPool" if isinstance(m,nn.AvgPool2d) else
            "AdaptivePool" if isinstance(m,(nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d)) else
            "Dropout" if isinstance(m,nn.Dropout) else "Identity"), n)
          for n,m in layers]

    tf=T.Compose([T.Resize(224),T.ToTensor(),
                  T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    fwd=[]
    for p in _test_imgs():
        x=tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        acc,hk=_collect(layers); hds=[m.register_forward_hook(hk) for _,m in layers]
        with torch.no_grad(): model(x)
        [h.remove() for h in hds]
        fwd.append([np.concatenate(a) if a else np.array([0.0],np.float32)
                    for a in acc])

    concat=[np.concatenate([it[i] for it in fwd]) for i in range(len(layers))]
    bf16,fp16=_precision_fitness(concat)
    act=_pct(fwd); act.update({"abs_mean":[float(v.mean()) for v in concat],
                               "abs_std":[float(v.std()) for v in concat]})
    _plot(act,meta,tag,bf16,fp16,"act")

    # backward‑gradients ----------------------------------------------------
    loader=_cifar_loader(False,64)
    crit=nn.CrossEntropyLoss()
    gitems=[]
    for b,(x,y) in enumerate(loader):
        if b>=args.grad_batches: break
        x,y=x.to(DEVICE),y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        crit(model(x),y).backward()
        per=[]
        for _,m in layers:
            vec=[p.grad.detach().abs().flatten().cpu()
                 for p in m.parameters(recurse=False) if p.grad is not None]
            per.append(torch.cat(vec).numpy() if vec else np.array([0.0],np.float32))
        gitems.append(per)

    concat_g=[np.concatenate([it[i] for it in gitems]) for i in range(len(layers))]
    bf16g,fp16g=_precision_fitness(concat_g)
    grad=_pct(gitems); grad.update({"abs_mean":[float(v.mean()) for v in concat_g],
                                    "abs_std":[float(v.std()) for v in concat_g]})
    _plot(grad,meta,tag,bf16g,fp16g,"grad")
    return act,grad

# ═══════════ combined plots (overall and per‑depth) ═══════════
def _combined(stats,metric,fname,title,subset:List[str]|None=None):
    fig,ax=plt.subplots(figsize=(20,9),dpi=300)
    for tag,data in stats.items():
        if subset and tag not in subset: continue
        μ=data[metric]; x=np.arange(1,len(μ)+1)
        if tag.startswith("resnet"): col=MODEL_COLOUR["resnet"]
        elif tag.startswith("plain"): col=MODEL_COLOUR["plain"]
        else:                         col=MODEL_COLOUR["densenet"]
        ax.plot(x,μ,label=tag,lw=2,color=col,alpha=.9)
    for y in (FP16_LIMIT,BF16_LIMIT): ax.axhline(y,ls=":",lw=1.8,color="#607D8B")
    ax.set_xlabel("Layer index",fontsize=14,fontweight="bold")
    ax.set_ylabel("Abs. Top‑1 % mean" if "top1" in metric
                  else "Abs. mean of all values",
                  fontsize=14,fontweight="bold")
    ax.set_title(title,fontsize=17)
    ax.grid(ls="--",alpha=.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out=PLOT_DIR/fname; fig.savefig(out,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"[✓] plot → {out}")

def depth_of(tag:str)->int: return int(re.search(r"(\d+)$",tag)[0])

def make_all_combined(act,grad):
    # overall (all depths mixed)
    _combined(act,"top1_mean","combined_top1_act_all.png",
              "Top‑1 % Activations – all depths")
    _combined(act,"abs_mean","combined_abs_act_all.png",
              "All‑|x| Activations – all depths")
    _combined(grad,"top1_mean","combined_top1_grad_all.png",
              "Top‑1 % Gradients – all depths")
    _combined(grad,"abs_mean","combined_abs_grad_all.png",
              "All‑|x| Gradients – all depths")
    # per‑depth
    for d in (14,18,34):
        subset=[t for t in act if depth_of(t)==d]
        if not subset: continue
        _combined(act,"top1_mean",f"combined_top1_act_{d}.png",
                  f"Top‑1 % Activations – {d}-layer models",subset)
        _combined(act,"abs_mean",f"combined_abs_act_{d}.png",
                  f"All‑|x| Activations – {d}-layer models",subset)
        _combined(grad,"top1_mean",f"combined_top1_grad_{d}.png",
                  f"Top‑1 % Gradients – {d}-layer models",subset)
        _combined(grad,"abs_mean",f"combined_abs_grad_{d}.png",
                  f"All‑|x| Gradients – {d}-layer models",subset)

# ═══════════ driver ═══════════
if __name__=="__main__":
    print(f"Device: {DEVICE} | plots → {PLOT_DIR}")

    tags = ["resnet14","resnet18","resnet34",
            "plain14","plain18","plain34",
            "densenet14","densenet18","densenet34"]

    act_all,grad_all={},{}
    for t in tags:
        try:
            a,g=analyse(t)
            act_all[t],grad_all[t]=a,g
        except Exception as e:
            print(f"[ERR] {t}: {e}")

    make_all_combined(act_all,grad_all)

    # save JSON
    (PLOT_DIR/"all_metrics.json").write_text(
        json.dumps({"activations":act_all,"gradients":grad_all},indent=2))
    print("[✓] Finished – see plots & JSON in",PLOT_DIR)
