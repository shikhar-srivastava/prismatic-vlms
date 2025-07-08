#!/usr/bin/env python3
# analyze_custom_resnet_densenet.py
#
# Architectural-parity ResNet / PlainNet / DenseNet analyser
# – forward activation & backward-gradient magnitude statistics
# – CIFAR-100 trained weights (retrains if absent)
# ----------------------------------------------------------------------

from __future__ import annotations
import argparse, json, re, random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D         # for custom legend handles
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from PIL import Image

# ═════════════════════════════ CONFIG ══════════════════════════════
BASE      = Path.cwd()
CHK_DIR   = BASE / "checkpoints";  CHK_DIR.mkdir(exist_ok=True)
PLOT_DIR  = BASE / "viz/plots/act_analysis_atomic_custom_resnet_densenet"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR  = BASE / "datasets";     DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# thresholds for half-precision error bands (≈½-ULP)
FP16_LIMIT, BF16_LIMIT = 4.0, 0.5

MODEL_COLOUR = {"resnet":"#D32F2F",
                "plain":"#7B1FA2",
                "densenet":"#1976D2"}

# identical learning schedule to the original training script
DEPTH_CFG: dict[int,Tuple[int,float,List[int]]] = {
    14:(100,0.1,[50,75]),
    18:(150,0.1,[75,110]),
    34:(200,0.1,[100,150])
}

# ═════════════════════════════ CLI ════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--train",   action="store_true",
                help="train if checkpoint missing")
ap.add_argument("--retrain", action="store_true",
                help="force retraining even if checkpoint exists")
ap.add_argument("--batch", type=int, default=128,
                help="training batch size")
ap.add_argument("--grad-batches", type=int, default=10,
                help="CIFAR-100 batches for gradient statistics")
ap.add_argument("--tags", nargs="+", default=[],
                help="subset of model tags to analyse (default: all)")
args = ap.parse_args()

# ═════════════════════════ ARCHITECTURE DEFINITIONS ═══════════════
# ---------- Residual / Plain blocks --------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c:int, out_c:int, stride:int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.downsample: nn.Module | None = None
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
        return self.act(out + identity)

class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c:int, out_c:int, stride:int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
    def forward(self,x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out)

# ---------- ResNet / PlainNet -------------------------------------
class _ResNetBase(nn.Module):
    def __init__(self, block, layers: List[int], num_classes:int=100):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.act   = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(block, 64,  layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_c, blocks, stride):
        seq = [block(self.in_c, out_c, stride)]
        self.in_c = out_c * block.expansion
        for _ in range(1, blocks):
            seq.append(block(self.in_c, out_c, 1))
        return nn.Sequential(*seq)

    def forward(self,x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet14(): return _ResNetBase(BasicBlock,[3,2,2,2])
def resnet18(): return _ResNetBase(BasicBlock,[2,2,2,2])
def resnet34(): return _ResNetBase(BasicBlock,[3,4,6,3])

def plain14():  return _ResNetBase(PlainBlock,[3,2,2,2])
def plain18():  return _ResNetBase(PlainBlock,[2,2,2,2])
def plain34():  return _ResNetBase(PlainBlock,[3,4,6,3])

# ---------- DenseNet (original settings) --------------------------
from collections import OrderedDict
class _DenseLayer(nn.Module):
    def __init__(self,in_f,gr,bn_size=4,drop=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_f)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_f, bn_size*gr, 1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size*gr)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*gr, gr, 3,1,1,bias=False)
        self.drop  = drop
    def forward(self,x):
        concat = x if isinstance(x,torch.Tensor) else torch.cat(x,1)
        out = self.conv1(self.relu1(self.norm1(concat)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop>0:
            out = F.dropout(out, self.drop, self.training)
        return out

class _DenseBlock(nn.Module):
    def __init__(self,n,in_f,gr):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DenseLayer(in_f+i*gr, gr) for i in range(n)]
        )
    def forward(self,x):
        feats=[x]
        for l in self.layers:
            y = l(feats)
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
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3,init_f,7,2,3,bias=False)),
            ("norm0", nn.BatchNorm2d(init_f)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3,2,1)),
        ]))
        n_f = init_f
        for i,n in enumerate(cfg):
            blk = _DenseBlock(n, n_f, gr)
            self.features.add_module(f"denseblock{i+1}", blk)
            n_f += n*gr
            if i != len(cfg)-1:
                out = int(n_f * comp)
                tr = _Transition(n_f, out)
                self.features.add_module(f"transition{i+1}", tr)
                n_f = out
        self.features.add_module("norm_final", nn.BatchNorm2d(n_f))
        self.classifier = nn.Linear(n_f, num_classes)
    def forward(self,x):
        x = self.features(x)
        x = F.relu(x,inplace=True)
        x = F.adaptive_avg_pool2d(x,1).flatten(1)
        return self.classifier(x)

def densenet14(): return DenseNet((3,4,4,3), 64, 0.8)
def densenet18(): return DenseNet((4,4,4,6), 64, 0.8)
def densenet34(): return DenseNet((6,8,12,8),64,0.75)

# ---------- factory table ------------------------------------------------------
FACTORY: Dict[str, callable] = {
    "resnet14":resnet14,"resnet18":resnet18,"resnet34":resnet34,
    "plain14":plain14,  "plain18":plain18,  "plain34":plain34,
    "densenet14":densenet14,"densenet18":densenet18,"densenet34":densenet34
}

# ════════════════════════ TRAINING / CKPT UTILITIES ══════════════
def depth_of(tag:str)->int: return int(re.search(r"(\d+)$",tag)[0])

def _ckpt_variants(tag:str):
    return [CHK_DIR/f"{tag}.pth",
            CHK_DIR/f"{tag}.pt",
            CHK_DIR/f"cifar100_{tag}.pth",
            CHK_DIR/f"cifar100_{tag}.pt"]

def _find_ckpt(tag:str)->Path|None:
    for p in _ckpt_variants(tag):
        if p.exists(): return p
    return None

# ---------- CIFAR-100 loaders ---------------------------------------------------
CIFAR_MEAN, CIFAR_STD = [0.5071,0.4865,0.4409],[0.2673,0.2564,0.2761]
def _cifar_loader(train:bool, batch:int):
    tf = (T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN,CIFAR_STD)
          ]) if train else
          T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN,CIFAR_STD)
          ]))
    ds = dsets.CIFAR100(DATA_DIR, train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=train,
                      num_workers=4, pin_memory=True, persistent_workers=True)

# ---------- training loop ------------------------------------------------------
def train_if_needed(tag:str)->Path:
    ck = _find_ckpt(tag)
    if ck and not (args.retrain or (args.train and not ck.name.startswith("cifar100_"))):
        print(f"[✓] using checkpoint {ck.name}")
        return ck
    if not args.train and not args.retrain and ck is None:
        raise FileNotFoundError(f"{tag}: checkpoint missing – run with --train")

    depth = depth_of(tag)
    epochs, base_lr, milestones = DEPTH_CFG[depth]
    print(f"[⋯] Training {tag}: {epochs} epochs")

    model = FACTORY[tag]().to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), base_lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, 0.1)
    crit  = nn.CrossEntropyLoss()
    loader = _cifar_loader(True, args.batch)

    model.train()
    for ep in range(1, epochs+1):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}")

    ck = CHK_DIR / f"cifar100_{tag}.pth"
    torch.save(model.state_dict(), ck)
    print(f"[✓] saved → {ck}")
    return ck

# ═══════════════════════ ANALYSIS UTILITIES ═══════════════════════
ATOMIC_TYPES = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU,
                nn.Sigmoid, nn.SiLU, nn.Linear,
                nn.MaxPool2d, nn.AvgPool2d,
                nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                nn.Dropout, nn.Identity)

def _atomic_layers(net: nn.Module):
    return [(n,m) for n,m in net.named_modules()
            if len(list(m.children()))==0 and isinstance(m,ATOMIC_TYPES)]

def _precision_fitness(vecs: List[np.ndarray]):
    flat = np.concatenate(vecs)
    total = flat.size
    return ((np.abs(flat)<=BF16_LIMIT).sum()/total*100,
            (np.abs(flat)<=FP16_LIMIT).sum()/total*100)

def _collect(layers):
    acc=[[] for _ in layers]
    id2={id(m):i for i,(_,m) in enumerate(layers)}
    def hk(mod,inp,_out):
        v = torch.abs(inp[0]).detach().squeeze(0).flatten().cpu().numpy()
        acc[id2[id(mod)]].append(v)
    return acc, hk

def _pct(items):
    n=len(items[0])
    def agg(p):
        mu,sd=[],[]
        for i in range(n):
            vecs=[it[i] for it in items]
            tops=[(v[v>=np.percentile(v,100-p)].mean() if v.size else 0.0)
                  for v in vecs]
            mu.append(float(np.mean(tops)))
            sd.append(float(np.std(tops)))
        return mu,sd
    t1m,t1s = agg(1); t5m,t5s = agg(5); t10m,t10s = agg(10)
    return {"top1_mean":t1m,"top1_std":t1s,
            "top5_mean":t5m,"top5_std":t5s,
            "top10_mean":t10m,"top10_std":t10s}

# ---------- 20 random CIFAR-100 validation images ------------------------------
def _sample_test_images(n:int=20)->List[torch.Tensor]:
    ds = dsets.CIFAR100(DATA_DIR, train=False, download=True,
            transform=T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN,CIFAR_STD)
            ]))
    idx = random.sample(range(len(ds)), n)
    return [ds[i][0] for i in idx]

# ═══════════════════════ PLOTTING HELPER ══════════════════════════
def _plot(stats, meta, tag, bf16, fp16, kind):
    x = np.arange(1, len(stats["abs_mean"])+1)
    fig, ax = plt.subplots(figsize=(26,10), dpi=300)

    # main series
    series = {"All |x|":("#1976D2","o","abs_mean","abs_std"),
              "Top 1 %":("#D32F2F","s","top1_mean","top1_std"),
              "Top 5 %":("#FFA000","^","top5_mean","top5_std"),
              "Top 10 %":("#FBC02D","D","top10_mean","top10_std")}
    for lbl,(c,m,mu_k,sd_k) in series.items():
        mu, sd = stats[mu_k], stats[sd_k]
        ax.plot(x, mu, label=lbl, marker=m, lw=2, ms=6, color=c)
        ax.fill_between(x, np.array(mu)-np.array(sd), np.array(mu)+np.array(sd),
                        color=c, alpha=.15)

    # precision thresholds
    for y_val, lab in ((FP16_LIMIT,"FP16"), (BF16_LIMIT,"BF16")):
        ax.axhline(y_val, ls=":", lw=1.6, color="#607D8B")
        ax.text(0.99, y_val*(1.02 if y_val>1 else 1.15),
                f"{lab} • ≤½-ULP", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, color="#455A64")

    # per-layer markers (layer-type colour)
    TYPE_COL = {"Conv":"#42A5F5","BN":"#EF5350","ReLU":"#FFB300",
                "Linear":"#66BB6A","MaxPool":"#8E24AA","AvgPool":"#26A69A",
                "AdaptivePool":"#26C6DA","Dropout":"#BDBDBD","Identity":"#78909C"}
    thr = 2*np.mean(stats["top1_mean"])
    y_min,y_max = ax.get_ylim(); y_rng = y_max-y_min; mark_y = y_max + y_rng*0.08
    used_types=set()
    for i,(lt,name) in enumerate(meta,1):
        col = TYPE_COL.get(lt,"#9E9E9E")
        ax.scatter(i, mark_y, color=col, s=16,
                   edgecolors="white", lw=0.6, zorder=3)
        used_types.add(lt)
        if stats["top1_mean"][i-1] > thr:
            ax.text(i, mark_y - y_rng*0.04, name.split(".")[-1],
                    rotation=45, ha="center", va="top",
                    fontsize=8, color="#37474F")

    # labels & grid
    ax.set_xlabel("Layer index", fontsize=15, fontweight="bold")
    ax.set_ylabel("Activation |x|" if kind=="act"
                  else "Gradient |∂L/∂θ|",
                  fontsize=15, fontweight="bold")
    ax.set_title(f"{tag}: {'Activations' if kind=='act' else 'Gradients'}",
                 fontsize=18, fontweight="bold")
    ax.grid(ls="--", alpha=.3)

    # legends
    series_leg = ax.legend(fontsize=11, loc="upper left")

    handles = [Line2D([0],[0], marker='s', linestyle='',
                      markerfacecolor=TYPE_COL.get(t,"#9E9E9E"),
                      markeredgecolor="white", markeredgewidth=0.6,
                      markersize=10, label=t)
               for t in sorted(used_types)]
    type_leg = ax.legend(handles=handles, title="Layer type",
                         fontsize=10, title_fontsize=11,
                         loc="center left", bbox_to_anchor=(1.02,0.5),
                         borderaxespad=0.8, framealpha=0.95)
    ax.add_artist(series_leg)

    # numeric box
    ax.text(0.99, 0.98, f"≤{BF16_LIMIT}: {bf16:4.1f}%\n"
                        f"≤{FP16_LIMIT}: {fp16:4.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#ECEFF1", edgecolor="#B0BEC5"))

    ax.set_ylim(y_min, mark_y + y_rng*0.15)
    plt.tight_layout()
    out = PLOT_DIR / f"{tag}_{kind}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"[✓] plot → {out}")

# ═════════════════════ SINGLE-MODEL ANALYSIS ═════════════════════
def analyse(tag:str):
    ck = train_if_needed(tag)
    model = FACTORY[tag]().to(DEVICE)
    state = torch.load(ck, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    layers = _atomic_layers(model)
    meta=[(("Conv" if isinstance(m,nn.Conv2d) else
            "BN" if isinstance(m,nn.BatchNorm2d) else
            "ReLU" if isinstance(m,(nn.ReLU,nn.LeakyReLU)) else
            "Linear" if isinstance(m,nn.Linear) else
            "MaxPool" if isinstance(m,nn.MaxPool2d) else
            "AvgPool" if isinstance(m,nn.AvgPool2d) else
            "AdaptivePool" if isinstance(m,(nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d)) else
            "Dropout" if isinstance(m,nn.Dropout) else
            "Identity"), n) for n,m in layers]

    # forward activations (20 random CIFAR-100 val images)
    fwd=[]
    for img in _sample_test_images():
        xb = img.unsqueeze(0).to(DEVICE)
        acc,hk = _collect(layers)
        hooks=[m.register_forward_hook(hk) for _,m in layers]
        with torch.no_grad(): model(xb)
        for h in hooks: h.remove()
        fwd.append([np.concatenate(a) if a else np.array([0.0],np.float32)
                    for a in acc])
    concat=[np.concatenate([it[i] for it in fwd]) for i in range(len(layers))]
    bf16,fp16=_precision_fitness(concat)
    act=_pct(fwd)
    act.update({"abs_mean":[float(v.mean()) for v in concat],
                "abs_std" :[float(v.std())  for v in concat]})
    _plot(act,meta,tag,bf16,fp16,"act")

    # backward gradients -------------------------------------------------------
    loader=_cifar_loader(False,64)
    crit=nn.CrossEntropyLoss()
    gitems=[]
    for b,(xb,yb) in enumerate(loader):
        if b >= args.grad_batches: break
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        model.zero_grad(set_to_none=True)
        crit(model(xb),yb).backward()
        per=[]
        for _,m in layers:
            vec=[p.grad.detach().abs().flatten().cpu()
                 for p in m.parameters(recurse=False) if p.grad is not None]
            per.append(torch.cat(vec).numpy() if vec else np.array([0.0],np.float32))
        gitems.append(per)
    concat_g=[np.concatenate([it[i] for it in gitems]) for i in range(len(layers))]
    bf16g,fp16g=_precision_fitness(concat_g)
    grad=_pct(gitems)
    grad.update({"abs_mean":[float(v.mean()) for v in concat_g],
                 "abs_std" :[float(v.std())  for v in concat_g]})
    _plot(grad,meta,tag,bf16g,fp16g,"grad")
    return act,grad

# ═════════════════════ COMBINED COMPARISONS ══════════════════════
def _combined(stats:Dict[str,dict], metric:str,
              fname:str, title:str, subset:List[str]|None=None):
    fig,ax=plt.subplots(figsize=(20,9), dpi=300)
    for tag,data in stats.items():
        if subset and tag not in subset: continue
        col=(MODEL_COLOUR["resnet"]  if tag.startswith("resnet") else
             MODEL_COLOUR["plain"]   if tag.startswith("plain")  else
             MODEL_COLOUR["densenet"])
        mu=data[metric]
        ax.plot(np.arange(1,len(mu)+1), mu, label=tag, lw=2, color=col, alpha=.9)
    for y in (FP16_LIMIT,BF16_LIMIT): ax.axhline(y,ls=":",lw=1.6,color="#607D8B")
    ax.set_xlabel("Layer index",fontsize=14,fontweight="bold")
    ax.set_ylabel("Abs. Top-1 % mean" if "top1" in metric else "Abs. mean",
                  fontsize=14,fontweight="bold")
    ax.set_title(title,fontsize=17)
    ax.grid(ls="--",alpha=.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out=PLOT_DIR/fname; fig.savefig(out,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"[✓] plot → {out}")

def make_all_combined(act,grad):
    _combined(act,"top1_mean","combined_top1_act_all.png",
              "Top-1 % Activations – all depths")
    _combined(act,"abs_mean","combined_abs_act_all.png",
              "All-|x| Activations – all depths")
    _combined(grad,"top1_mean","combined_top1_grad_all.png",
              "Top-1 % Gradients – all depths")
    _combined(grad,"abs_mean","combined_abs_grad_all.png",
              "All-|x| Gradients – all depths")
    for d in (14,18,34):
        subset=[t for t in act if depth_of(t)==d]
        if not subset: continue
        _combined(act,"top1_mean",f"combined_top1_act_{d}.png",
                  f"Top-1 % Activations – {d}-layer models",subset)
        _combined(act,"abs_mean",f"combined_abs_act_{d}.png",
                  f"All-|x| Activations – {d}-layer models",subset)
        _combined(grad,"top1_mean",f"combined_top1_grad_{d}.png",
                  f"Top-1 % Gradients – {d}-layer models",subset)
        _combined(grad,"abs_mean",f"combined_abs_grad_{d}.png",
                  f"All-|x| Gradients – {d}-layer models",subset)

# ═════════════════════════════ DRIVER ════════════════════════════
if __name__ == "__main__":
    print(f"Device: {DEVICE} | plots → {PLOT_DIR}")

    default_tags = ["resnet14","resnet18","resnet34",
                    "plain14","plain18","plain34",
                    "densenet14","densenet18","densenet34"]
    tags = args.tags if args.tags else default_tags

    act_all, grad_all = {}, {}
    for tag in tags:
        try:
            a,g = analyse(tag)
            act_all[tag], grad_all[tag] = a,g
        except Exception as e:
            print(f"[ERR] {tag}: {e}")

    if act_all and grad_all:
        make_all_combined(act_all, grad_all)
        (PLOT_DIR/"all_metrics.json").write_text(
            json.dumps({"activations":act_all,"gradients":grad_all}, indent=2))

    print("✓ Finished – see plots & JSON in", PLOT_DIR)
