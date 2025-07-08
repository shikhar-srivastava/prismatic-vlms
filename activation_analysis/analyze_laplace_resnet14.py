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
        # Add numerical stability for FP16
        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-10.0, max=10.0)  # Prevent overflow
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
        # Add numerical stability for FP16
        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-10.0, max=10.0)  # Prevent overflow
        x_sq = torch.square(x).to(x.dtype)
        # Add epsilon to prevent division by zero
        eps = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
        return (A * x_sq / (x_sq + A + eps)).to(x.dtype)

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
        
        # Add numerical stability for FP16
        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-5.0, max=5.0)  # Prevent overflow in erf
        
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

# FP16-specific configurations
FP16_CONFIG = {
    "lr_scale": 0.1,  # Reduce learning rate by 10x for FP16
    "grad_clip": 1.0,  # Gradient clipping threshold
    "scaler_init_scale": 2**16,  # Higher initial scale for better stability
    "scaler_growth_factor": 2.0,
    "scaler_backoff_factor": 0.5,
    "scaler_growth_interval": 2000
}

def train(model, act_name, prec_name, tr_loader, te_loader):
    tag = f"{act_name}_{prec_name}"
    ck  = CHK_DIR / f"cifar100_r14_{tag}.pth"
    if ck.exists():
        print(f"[skip] {tag} (checkpoint exists).")
        return

    epochs, base_lr, milestones = DEPTH_CFG[14]
    
    # FP16-specific learning rate adjustment
    if prec_name == "fp16":
        base_lr *= FP16_CONFIG["lr_scale"]
        print(f"[INFO] FP16 detected - scaling LR from {DEPTH_CFG[14][1]} to {base_lr:.4f}")
    
    model.to(DEVICE, memory_format=torch.channels_last)
    if prec_name.startswith("fp8"): model = convert_to_te(model)

    # Enhanced scaler configuration for FP16
    if prec_name == "fp16":
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
            init_scale=FP16_CONFIG["scaler_init_scale"],
            growth_factor=FP16_CONFIG["scaler_growth_factor"],
            backoff_factor=FP16_CONFIG["scaler_backoff_factor"],
            growth_interval=FP16_CONFIG["scaler_growth_interval"]
        )
    else:
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
                
                # Enhanced gradient handling for FP16
                if prec_name == "fp16":
                    scaler.scale(loss).backward()
                    # Gradient clipping for FP16 stability
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), FP16_CONFIG["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                
                # NaN detection for FP16 stability
                if prec_name == "fp16" and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"[WARN] {tag} - NaN/Inf detected in loss at epoch {ep}, stopping training")
                    return
                
                tl += loss.item()*xb.size(0)
                correct += (out.argmax(1)==yb).sum().item()
        
        # Check for NaN in accumulated loss
        if prec_name == "fp16" and (np.isnan(tl) or np.isinf(tl)):
            print(f"[WARN] {tag} - NaN/Inf detected in accumulated loss at epoch {ep}, stopping training")
            return
            
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

def create_performance_heatmap(all_metrics: Dict[str,dict]):
    """Create a professional heatmap showing best validation accuracy for each activation/precision combo."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    # Extract unique activations and precisions
    activations = sorted(set(tag.split("_")[0] for tag in all_metrics.keys()))
    precisions = sorted(set(tag.split("_",1)[1] for tag in all_metrics.keys()))
    
    # Create performance matrix
    perf_matrix = np.zeros((len(activations), len(precisions)))
    
    for i, act in enumerate(activations):
        for j, prec in enumerate(precisions):
            tag = f"{act}_{prec}"
            if tag in all_metrics:
                perf_matrix[i, j] = all_metrics[tag]['best_val_acc'] * 100
    
    # Create beautiful heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Custom colormap (white to dark blue)
    colors_map = ['#ffffff', '#e8f4f8', '#d1e7f0', '#85c1e5', '#3498db', '#2874a6']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors_map, N=n_bins)
    
    im = ax.imshow(perf_matrix, cmap=cmap, aspect='auto', vmin=perf_matrix.min(), vmax=perf_matrix.max())
    
    # Add text annotations
    for i in range(len(activations)):
        for j in range(len(precisions)):
            text = ax.text(j, i, f'{perf_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black" if perf_matrix[i, j] < perf_matrix.max()*0.7 else "white",
                          fontsize=11, weight='bold')
    
    # Customize axes
    ax.set_xticks(np.arange(len(precisions)))
    ax.set_yticks(np.arange(len(activations)))
    ax.set_xticklabels([p.upper().replace('_', '-') for p in precisions], fontsize=12)
    ax.set_yticklabels([a.upper() for a in activations], fontsize=12)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title("Best Validation Accuracy Heatmap (%)\nResNet-14 on CIFAR-100", 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Precision", fontsize=14, weight='bold')
    ax.set_ylabel("Activation Function", fontsize=14, weight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Validation Accuracy (%)", rotation=-90, va="bottom", fontsize=12, weight='bold')
    
    plt.tight_layout()
    out_heatmap = LOG_DIR / "performance_heatmap_cifar100.png"
    plt.savefig(out_heatmap, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[INFO] Performance heatmap saved → {out_heatmap}")

def create_training_summary_table(all_metrics: Dict[str,dict]):
    rows=[]
    for name,m in all_metrics.items():
        act, prec = name.split('_', 1)
        rows.append({
            "Activation": act.upper(),
            "Precision": prec.upper().replace('_', '-'),
            "Best Val Acc (%)": f"{m['best_val_acc']*100:.2f}",
            "Best Epoch": m['best_val_acc_epoch'],
            "Final Val Acc (%)": f"{m['final_val_acc']*100:.2f}",
            "Improvement (%)": f"{(m['best_val_acc'] - m['val_acc'][0])*100:.2f}" if len(m['val_acc']) > 0 else "N/A"
        })
    
    # Sort by best validation accuracy (descending)
    rows.sort(key=lambda r: float(r['Best Val Acc (%)'].rstrip('%')), reverse=True)
    
    headers=list(rows[0].keys())
    
    # Create professional table
    fig, ax = plt.subplots(figsize=(16, 0.7*len(rows)+3), dpi=300)
    ax.axis('off')
    
    # Create table with enhanced styling
    tbl = ax.table(cellText=[list(r.values()) for r in rows],
                 colLabels=headers, cellLoc='center', loc='center')
    
    # Enhanced table styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)
    
    # Color coding - best performers get highlighted
    best_acc = max(float(r['Best Val Acc (%)'].rstrip('%')) for r in rows)
    
    for i, row in enumerate(rows):
        acc = float(row['Best Val Acc (%)'].rstrip('%'))
        
        # Header styling
        for j in range(len(headers)):
            cell = tbl[(0, j)]
            cell.set_facecolor('#34495E')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.1)
        
        # Data row styling
        for j in range(len(headers)):
            cell = tbl[(i+1, j)]
            
            # Highlight top performers
            if acc >= best_acc - 1.0:  # Within 1% of best
                cell.set_facecolor('#E8F8F5')
            elif acc >= best_acc - 2.0:  # Within 2% of best
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('white')
            
            cell.set_height(0.08)
            
            # Bold the best accuracy values
            if j == 2 and acc == best_acc:  # Best Val Acc column
                cell.set_text_props(weight='bold', color='#27AE60')
    
    plt.title('ResNet-14 Activation Function & Precision Performance Summary\nCIFAR-100 Classification Results', 
              pad=25, fontsize=16, weight='bold')
    
    # Add subtitle with key insights
    plt.figtext(0.5, 0.02, f'Best Performance: {best_acc:.2f}% | Total Configurations: {len(rows)}', 
                ha='center', fontsize=11, style='italic', color='#666666')
    
    out = LOG_DIR / "summary_table_cifar100.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Enhanced JSON output
    summary_data = {
        "experiment_info": {
            "model": "ResNet-14",
            "dataset": "CIFAR-100",
            "total_configurations": len(rows),
            "best_accuracy": best_acc
        },
        "results": rows,
        "top_3_performers": rows[:3]
    }
    
    json.dump(summary_data, open(LOG_DIR/"summary_cifar100.json","w"), indent=2)
    print(f"[INFO] Enhanced summary table saved → {out}")

def create_comparative_training_plots():
    metrics_files=list(LOG_DIR.glob("*_metrics.json"))
    if not metrics_files:
        print("[WARN] No metrics found for comparative plot.")
        return
    all_metrics={}
    for j in metrics_files:
        tag=j.stem.replace("_metrics","")
        all_metrics[tag]=json.load(open(j))
    
    # Enhanced color palette and styling
    colors={'relu':'#2E86C1','relu2':'#E67E22',
            'lap4pw':'#27AE60','rat4':'#E74C3C','laplace':'#8E44AD'}
    linestyles={'fp32':'-','fp16':'--','fp8_e4m3':'-.','fp8_e5m2':':'}
    
    # Set matplotlib style for professional appearance
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    })
    
    # 1. Combined plot with all activation functions and precisions
    fig, ax = plt.subplots(figsize=(16,10), dpi=300)
    for tag,m in all_metrics.items():
        act,prec=tag.split("_",1)
        ep=m['epochs']; acc=np.array(m['val_acc'])*100
        ax.plot(ep, acc,
                 color=colors[act], linestyle=linestyles.get(prec,'-'),
                linewidth=2.5, label=f"{act.upper()} / {prec.upper()}",
                alpha=0.9, marker='o' if len(ep) <= 20 else None, markersize=3)
    
    ax.set_title("Validation Accuracy Comparison – ResNet-14 on CIFAR-100", 
                fontsize=18, weight='bold', pad=25)
    ax.set_xlabel("Epoch", fontsize=14, weight='bold')
    ax.set_ylabel("Validation Accuracy (%)", fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(ncol=4, fontsize=10, loc='lower right', 
              bbox_to_anchor=(1.0, 0.02), framealpha=0.95)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    out=LOG_DIR/"comparative_val_acc_cifar100.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[INFO] Combined comparative plot saved → {out}")
    
    # 2. Separate plots for each precision
    precisions = list(set(tag.split("_",1)[1] for tag in all_metrics.keys()))
    
    for prec in sorted(precisions):
        fig, ax = plt.subplots(figsize=(12,8), dpi=300)
        
        prec_metrics = {tag: m for tag, m in all_metrics.items() if tag.endswith(f"_{prec}")}
        
        for tag, m in prec_metrics.items():
            act = tag.split("_")[0]
            ep = m['epochs']
            acc = np.array(m['val_acc']) * 100
            
            ax.plot(ep, acc, color=colors[act], linewidth=3.0, 
                   label=f"{act.upper()}", alpha=0.9,
                   marker='o', markersize=4, markevery=max(1, len(ep)//10))
        
        # Beautify the precision-specific plot
        prec_title = {
            'fp32': 'FP32 (Single Precision)',
            'fp16': 'FP16 (Half Precision)', 
            'fp8_e4m3': 'FP8-E4M3 (8-bit Floating Point)',
            'fp8_e5m2': 'FP8-E5M2 (8-bit Floating Point)'
        }.get(prec, prec.upper())
        
        ax.set_title(f"Activation Function Comparison – {prec_title}\nResNet-14 on CIFAR-100", 
                    fontsize=16, weight='bold', pad=20)
        ax.set_xlabel("Epoch", fontsize=13, weight='bold')
        ax.set_ylabel("Validation Accuracy (%)", fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
        ax.set_ylim(bottom=0)
        
        # Add subtle background color
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        out_prec = LOG_DIR / f"activation_comparison_{prec}_cifar100.png"
        plt.savefig(out_prec, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[INFO] {prec.upper()} precision plot saved → {out_prec}")
    
    # 3. Enhanced summary heatmap
    create_performance_heatmap(all_metrics)
    
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
