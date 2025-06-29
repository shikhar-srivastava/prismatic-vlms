#!/usr/bin/env python3
# analyze_laplace_resnet14.py
#
# ▸ Activation sweep: ReLU | ReLU² | Lap4-PW-C¹ | Rational-4
# ▸ Precision sweep  : FP32 | FP16 | FP8-e4m3fn | FP8-e5m2
# ▸ Dataset          : CIFAR-100 @ 224 px
#
# Output:
#   checkpoints_lap/cifar100_r14_<act>_<prec>.pth
#   viz/laplace_r14/<act>_<prec>_curves.png
#   viz/laplace_r14/<act>_<prec>_metrics.json
#   viz/laplace_r14/comparative_val_acc_cifar100.png
#   viz/laplace_r14/summary_table_cifar100.png
#   viz/laplace_r14/summary_cifar100.json
#
# Requires CUDA GPU (L40S or Hopper/Ada recommended for FP8) and
# transformer-engine ≥ 0.11 for native FP8 kernels.
# ----------------------------------------------------------------------

from __future__ import annotations
import argparse, json, os, warnings, re
from pathlib import Path
from typing import Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, Subset

# ───────────────────────── ENV & DEPENDENCIES ──────────────────────────
assert torch.cuda.is_available(), "CUDA GPU required."
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

try:
    import transformer_engine.pytorch as te  # type: ignore
    _HAS_TE = True
except ModuleNotFoundError:
    _HAS_TE = False
    warnings.warn("Transformer-Engine not installed → FP8 runs will be skipped.")

# ───────────────────────── ACTIVATION FUNCTIONS ────────────────────────
class ReLU2(nn.Module):
    def forward(self, x): 
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return torch.square(torch.clamp_min(x, zero)).to(x.dtype)

class Lap4PWC1(nn.Module):
    """C¹-continuous squared-ReLU up to T then Laplace tail (A=4, T=1.9)."""
    A, T, mu, sig, sqrt2 = 4.0, 1.9, 1.665, 0.1813, 1.4142135623730951
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure all constants have the same dtype and device as input
        device, dtype = x.device, x.dtype
        A = torch.tensor(self.A, device=device, dtype=dtype)
        T = torch.tensor(self.T, device=device, dtype=dtype)
        mu = torch.tensor(self.mu, device=device, dtype=dtype)
        sig = torch.tensor(self.sig, device=device, dtype=dtype)
        sqrt2 = torch.tensor(self.sqrt2, device=device, dtype=dtype)
        half = torch.tensor(0.5, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        
        y = torch.zeros_like(x)
        pos_sq  = (x > 0) & (x <= T)
        pos_lp  = x > T
        if pos_sq.any():
            y[pos_sq] = torch.square(x[pos_sq]).to(dtype)
        if pos_lp.any():
            z = (x[pos_lp] - mu) / (sig * sqrt2)
            y[pos_lp] = (half * A * (one + torch.erf(z))).to(dtype)
        return y

class Rational4(nn.Module):
    A = 4.0
    def forward(self, x): 
        # Ensure constant has the same dtype and device as input
        A = torch.tensor(self.A, device=x.device, dtype=x.dtype)
        x_sq = torch.square(x).to(x.dtype)
        return (A * x_sq / (x_sq + A)).to(x.dtype)

class LaplaceActivation(nn.Module):
    """Pure Laplace activation function based on error function.
    
    f_laplace(x; μ, σ) = 0.5 × [1 + erf((x - μ)/(σ√2))]
    
    where μ = √(1/2) and σ = √(1/4π) to approximate f_relu².
    """
    # Precomputed constants for maximum precision
    mu = 0.7071067811865476      # √(1/2)
    sigma = 0.28209479177387814  # √(1/4π)
    sqrt2 = 1.4142135623730951   # √2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure all constants have the same dtype and device as input
        device, dtype = x.device, x.dtype
        mu = torch.tensor(self.mu, device=device, dtype=dtype)
        sigma = torch.tensor(self.sigma, device=device, dtype=dtype)
        sqrt2 = torch.tensor(self.sqrt2, device=device, dtype=dtype)
        half = torch.tensor(0.5, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)
        
        # Compute z = (x - μ) / (σ√2)
        z = (x - mu) / (sigma * sqrt2)
        
        # f_laplace(x) = 0.5 × [1 + erf(z)]
        result = half * (one + torch.erf(z))
        
        return result.to(dtype)

ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "relu"    : nn.ReLU,
    "relu2"   : ReLU2,
    "lap4pw"  : Lap4PWC1,
    "rat4"    : Rational4,
    "laplace" : LaplaceActivation,
}

# ───────────────────────── PRECISION CONTEXTS ──────────────────────────
class PrecisionCtx:
    def __init__(self, mode: str):
        self.mode = mode
    def __enter__(self):
        if self.mode == "fp32":
            self.ctx = torch.cuda.amp.autocast(False)
        elif self.mode == "fp16":
            self.ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif self.mode.startswith("fp8"):
            if not _HAS_TE:
                raise RuntimeError("Transformer-Engine required for FP8.")
            if "e4m3" in self.mode:
                # Pure E4M3 format with DelayedScaling
                recipe = te.fp8.DelayedScaling(fp8_format=te.fp8.Format.E4M3)
            elif "e5m2" in self.mode:
                # Pure E5M2 format with Float8CurrentScaling (DelayedScaling doesn't support pure E5M2)
                recipe = te.fp8.Float8CurrentScaling()
                recipe.fp8_format = te.fp8.Format.E5M2
            else:
                recipe = te.fp8.DelayedScaling(fp8_format=te.fp8.Format.E4M3)
            self.ctx = te.fp8.fp8_autocast(enabled=True, fp8_recipe=recipe)
        else:
            raise ValueError(self.mode)
        return self.ctx.__enter__()
    def __exit__(self, et, ev, tb): return self.ctx.__exit__(et, ev, tb)

def convert_to_te(model: nn.Module) -> nn.Module:
    if not _HAS_TE: return model
    for name, mod in model.named_children():
        if isinstance(mod, nn.Linear):
            # TE FP8 Linear requires in_features (last dim) % 16 == 0 and
            # out_features (prod of all other dims) % 8 == 0.
            if (mod.in_features % 16 == 0) and (mod.out_features % 8 == 0):
                te_lin = te.Linear(mod.in_features, mod.out_features, True)
                te_lin.weight.data.copy_(mod.weight.data)
                te_lin.bias.data.copy_(mod.bias.data)
                setattr(model, name, te_lin)
            else:
                # Dimensions not compatible with TE FP8 kernel – keep original layer.
                convert_to_te(mod)
        else:
            convert_to_te(mod)
    return model

# ───────────────────────── RESNET-14 CORE ──────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inC, outC, act_cls, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inC, outC, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(outC)
        self.act1  = act_cls()
        self.conv2 = nn.Conv2d(outC, outC, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(outC)
        self.down  = (None if stride==1 and inC==outC else
                      nn.Sequential(nn.Conv2d(inC,outC,1,stride,bias=False),
                                    nn.BatchNorm2d(outC)))
        self.act2  = act_cls()
    def forward(self,x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = x if self.down is None else self.down(x)
        return self.act2(x+y)

class ResNet14(nn.Module):
    def __init__(self, act_cls, num_classes=100):
        super().__init__()
        self.inC = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            act_cls(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer1 = self._make_layer(64, 3, act_cls, 1)
        self.layer2 = self._make_layer(128,2, act_cls, 2)
        self.layer3 = self._make_layer(256,2, act_cls, 2)
        self.layer4 = self._make_layer(512,2, act_cls, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()
    def _make_layer(self, outC, blocks, act_cls, stride):
        layers=[BasicBlock(self.inC,outC,act_cls,stride)]
        self.inC=outC
        layers.extend(BasicBlock(self.inC,outC,act_cls) for _ in range(1,blocks))
        return nn.Sequential(*layers)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.stem(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x=self.avgpool(x).flatten(1)
        return self.fc(x)

# ───────────────────────── DATA PIPELINE ───────────────────────────────
IMG_RES, NUM_CLASSES = 224, 100
DATA_DIR = Path.cwd()/ "datasets"; DATA_DIR.mkdir(exist_ok=True)

def make_loaders(batch: int) -> Tuple[DataLoader, DataLoader]:
    mean,std = [0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]
    train_tf = T.Compose([T.RandomHorizontalFlip(),
                          T.RandomCrop(32,4),
                          T.Resize(IMG_RES),
                          T.ToTensor(), T.Normalize(mean,std)])
    test_tf  = T.Compose([T.Resize(IMG_RES),
                          T.ToTensor(), T.Normalize(mean,std)])
    tr = datasets.CIFAR100(DATA_DIR,True, download=True, transform=train_tf)
    te = datasets.CIFAR100(DATA_DIR,False,download=True, transform=test_tf)
    dl = lambda ds,bs,sh: DataLoader(ds,bs,sh,num_workers=8,pin_memory=True,persistent_workers=True)
    return dl(tr,batch,True), dl(te,256,False)

# ───────────────────────── TRAINING UTILS ──────────────────────────────
DEPTH_CFG = {14:(100,0.1,[50,75])}        # epochs, baseLR, milestones
CHK_DIR = Path.cwd()/ "checkpoints_lap"; CHK_DIR.mkdir(exist_ok=True)
LOG_DIR = Path.cwd()/ "viz"/"laplace_r14"; LOG_DIR.mkdir(parents=True, exist_ok=True)

def train(model, act_name, prec_name, tr_loader, te_loader):
    tag = f"{act_name}_{prec_name}"
    ck  = CHK_DIR / f"cifar100_r14_{tag}.pth"
    if ck.exists():
        print(f"[skip] {tag} (checkpoint exists).")
        return

    epochs, base_lr, milestones = DEPTH_CFG[14]
    model.to(DEVICE, memory_format=torch.channels_last)
    if prec_name.startswith("fp8"): model = convert_to_te(model)

    scaler = torch.cuda.amp.GradScaler(enabled=(prec_name!="fp32"))
    opt = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    crit = nn.CrossEntropyLoss()

    tr_loss, tr_acc, te_loss, te_acc, lrs = [],[],[],[],[]

    for ep in range(1, epochs+1):
        model.train(); tl, correct = 0.0, 0
        with PrecisionCtx(prec_name):
            for xb,yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with PrecisionCtx(prec_name):
                    out = model(xb); loss = crit(out,yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tl += loss.item()*xb.size(0)
                correct += (out.argmax(1)==yb).sum().item()
        tr_loss.append(tl/len(tr_loader.dataset))
        tr_acc.append(correct/len(tr_loader.dataset))

        # validation
        model.eval(); vl, vc = 0.0, 0
        with torch.no_grad(), PrecisionCtx(prec_name):
            for xb,yb in te_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb); l = crit(out,yb)
                vl += l.item()*xb.size(0)
                vc += (out.argmax(1)==yb).sum().item()
        te_loss.append(vl/len(te_loader.dataset))
        te_acc.append(vc/len(te_loader.dataset))
        lrs.append(opt.param_groups[0]['lr'])

        print(f"{tag:18s} | ep {ep:3}/{epochs} | "
              f"train {tr_loss[-1]:.3f}/{tr_acc[-1]*100:5.1f}% | "
              f"val {te_loss[-1]:.3f}/{te_acc[-1]*100:5.1f}%")

        torch.save(model.state_dict(), ck)
        sch.step()

    _save_curves_and_metrics(tag, tr_loss, te_loss, tr_acc, te_acc, lrs, epochs)

def _save_curves_and_metrics(tag, tl, vl, ta, va, lrs, epochs):
    ep = range(1,epochs+1)
    plt.figure(figsize=(12,4), dpi=300)
    plt.subplot(1,3,1); plt.plot(ep, tl,'b-', label="train"); plt.plot(ep, vl,'r-', label="val"); plt.title("Loss"); plt.legend(); plt.grid(alpha=.3)
    plt.subplot(1,3,2); plt.plot(ep, np.array(ta)*100,'b-', label="train"); plt.plot(ep, np.array(va)*100,'r-', label="val"); plt.title("Accuracy"); plt.grid(alpha=.3)
    plt.subplot(1,3,3); plt.semilogy(ep,lrs,'g-'); plt.title("LR"); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(LOG_DIR/f"{tag}_curves.png", dpi=300, bbox_inches='tight'); plt.close()

    json.dump({"epochs":list(ep),
               "train_loss":tl,"val_loss":vl,
               "train_acc":ta,"val_acc":va,
               "lrs":lrs,
               "best_val_acc":max(va),
               "best_val_acc_epoch":va.index(max(va))+1,
               "final_val_acc":va[-1]},
              open(LOG_DIR/f"{tag}_metrics.json","w"), indent=2)

# ───────────────────────── AGGREGATED DASHBOARD ────────────────────────
def parse_depth_from_tag(tag:str)->int:
    m=re.search(r"(\d+)",tag)
    return int(m.group(1)) if m else 14

def create_training_summary_table(all_metrics: Dict[str,dict]):
    rows=[]
    for name,m in all_metrics.items():
        rows.append({"Model":name.upper(),
                     "Best Val Acc (%)":f"{m['best_val_acc']*100:.2f}",
                     "Best Epoch":m['best_val_acc_epoch'],
                     "Final Val Acc (%)":f"{m['final_val_acc']*100:.2f}"})
    rows.sort(key=lambda r:r['Model'])
    headers=list(rows[0].keys())
    fig,ax=plt.subplots(figsize=(12,0.6*len(rows)+2), dpi=300)
    ax.axis('off')
    tbl=ax.table(cellText=[list(r.values()) for r in rows],
                 colLabels=headers, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2,1.8)
    plt.title('ResNet-14 Activation/Precision Sweep – CIFAR-100', pad=20, fontsize=14, weight='bold')
    out=LOG_DIR/"summary_table_cifar100.png"; plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    json.dump(rows, open(LOG_DIR/"summary_cifar100.json","w"), indent=2)
    print(f"[INFO] Summary table saved → {out}")

def create_comparative_training_plots():
    metrics_files=list(LOG_DIR.glob("*_metrics.json"))
    if not metrics_files:
        print("[WARN] No metrics found for comparative plot.")
        return
    all_metrics={}
    for j in metrics_files:
        tag=j.stem.replace("_metrics","")
        all_metrics[tag]=json.load(open(j))
    # Validation accuracy comparison
    colors={'relu':'#1f77b4','relu2':'#ff7f0e',
            'lap4pw':'#2ca02c','rat4':'#d62728'}
    linestyles={'fp32':'-','fp16':'--','fp8_e4m3fn':'-.','fp8_e5m2':':'}
    plt.figure(figsize=(14,8), dpi=300)
    for tag,m in all_metrics.items():
        act,prec=tag.split("_",1)
        ep=m['epochs']; acc=np.array(m['val_acc'])*100
        plt.plot(ep, acc,
                 color=colors[act], linestyle=linestyles.get(prec,'-'),
                 linewidth=2, label=f"{act.upper()} /{prec.upper()}",
                 alpha=0.85)
    plt.title("Validation Accuracy – ResNet-14 (CIFAR-100)", fontsize=16, weight='bold', pad=20)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.grid(alpha=.3); plt.legend(ncol=2, fontsize=9)
    out=LOG_DIR/"comparative_val_acc_cifar100.png"
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[INFO] Comparative plot saved → {out}")
    # summary table
    create_training_summary_table(all_metrics)

# ───────────────────────── MAIN ────────────────────────────────────────
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--no_train", action="store_true")
    args=parser.parse_args()

    train_loader, test_loader = make_loaders(args.batch)

    PRECISIONS = ["fp32", "fp16"] + ([] if not _HAS_TE else ["fp8_e4m3","fp8_e5m2"])
    if not args.no_train:
        for act_name, act_cls in ACTIVATIONS.items():
            for prec in PRECISIONS:
                model = ResNet14(act_cls, NUM_CLASSES)
                train(model, act_name, prec, train_loader, test_loader)

    # ───── Aggregated comparative dashboard ─────
    create_comparative_training_plots()
    print("\n✓ All tasks complete.  Check visualisations in:", LOG_DIR)
