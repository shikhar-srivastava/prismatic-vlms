"""
ResNet‑18 vs Plain‑18 — 15‑minute L40S workflow (final)
======================================================
* **Auto‑dataset**: ImageNet if `$IMAGENET_DIR` exists, else CIFAR‑100, or
  `--dataset tiny` for Tiny‑ImageNet.
* **Training**: AMP, channels‑last, `torch.compile`, OneCycleLR, 15‑min guard.
* **Checkpoints**: saved with *uncompiled* weights; loader auto‑patches any
  old `_orig_mod.` prefixes so re‑training is never needed.
* **Analysis**: layer‑wise activation/parameter JSON + PNGs into
  `viz/plots/act_analysis_rvp/`.

Run examples
------------
```bash
# fastest quick‑run (downloads CIFAR‑100)
python train_and_analyze_resnet18_vs_plain18.py

# Tiny‑ImageNet proxy
python train_and_analyze_resnet18_vs_plain18.py --dataset tiny --epochs 15

# Only analysis (skip training)
python train_and_analyze_resnet18_vs_plain18.py --no_train
```
"""

from __future__ import annotations

import argparse, json, os, time, warnings, contextlib
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.datasets.utils import download_and_extract_archive

# ────────────────────────── CONFIG ──────────────────────────
BASE = Path.cwd()
CHK_DIR = BASE / "checkpoints"; CHK_DIR.mkdir(exist_ok=True)
CURVE_DIR = BASE / "viz"; CURVE_DIR.mkdir(exist_ok=True)
PLOT_DIR = BASE / "viz" / "plots" / "act_analysis_rvp"; PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE / "datasets"; DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "Need a CUDA GPU for the 15‑min budget"

torch.backends.cudnn.benchmark = True

# ────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["auto", "imagenet", "cifar100", "tiny"], default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.4)
parser.add_argument("--subset", type=float, default=1.0, help="Fraction of ImageNet train split")
parser.add_argument("--no_train", action="store_true")
parser.add_argument("--no_analysis", action="store_true")
parser.add_argument("--amp", action="store_true", default=True)
args = parser.parse_args()

IMAGENET_DIR = os.getenv("IMAGENET_DIR")

# ──────────────────── DATASET CHOICE ────────────────────
if args.dataset != "auto":
    DATASET = args.dataset
else:
    DATASET = "imagenet" if IMAGENET_DIR and Path(IMAGENET_DIR).is_dir() else "cifar100"
NUM_CLASSES = 1000 if DATASET in {"imagenet", "tiny"} else 100
IMG_RES = 224
print(f"[INFO] dataset={DATASET}  epochs={args.epochs}  batch={args.batch}")

# ─────────────────── ARCHITECTURES ────────────────────
class PlainBasic(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.act   = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x)

def _layer(block, in_c, out_c, n, stride=1):
    seq = [block(in_c, out_c, stride)]
    seq.extend(block(out_c, out_c) for _ in range(n - 1))
    return nn.Sequential(*seq)

class PlainResNet18(nn.Module):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = _layer(PlainBasic, 64, 64, 2)
        self.layer2 = _layer(PlainBasic, 64, 128, 2, 2)
        self.layer3 = _layer(PlainBasic, 128, 256, 2, 2)
        self.layer4 = _layer(PlainBasic, 256, 512, 2, 2)
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, nc))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)

MODELS: Dict[str, nn.Module] = {
    "resnet18": models.resnet18(num_classes=NUM_CLASSES, weights=None),
    "plain18" : PlainResNet18(),
}

# ─────────────────── DATA LOADERS ────────────────────

def loaders():
    if DATASET == "imagenet":
        tra = T.Compose([T.RandomResizedCrop(IMG_RES), T.RandomHorizontalFlip(), T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
        val = T.Compose([T.Resize(256), T.CenterCrop(IMG_RES), T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
        root = Path(IMAGENET_DIR)
        tr = datasets.ImageFolder(root/"train", tra)
        if args.subset < 1.0:
            idx = np.random.RandomState(0).choice(len(tr), int(args.subset*len(tr)), replace=False)
            tr = Subset(tr, idx.tolist())
        va = datasets.ImageFolder(root/"val", val)
    elif DATASET == "tiny":
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_f = DATA_DIR / "tiny.zip"
        if not zip_f.exists():
            download_and_extract_archive(url, str(DATA_DIR), filename="tiny.zip")
        base = DATA_DIR/"tiny-imagenet-200"
        tr_dir = base/"train"; va_dir = base/"val/images"
        n, s = [.480,.448,.397], [.277,.269,.282]
        tra = T.Compose([T.Resize(IMG_RES), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(n,s)])
        val = T.Compose([T.Resize(IMG_RES), T.ToTensor(), T.Normalize(n,s)])
        tr = datasets.ImageFolder(tr_dir, tra)
        va = datasets.ImageFolder(va_dir, val)
    else:  # cifar100
        n, s = [.5071,.4865,.4409], [.2673,.2564,.2762]
        tra = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32,4), T.Resize(IMG_RES), T.ToTensor(), T.Normalize(n,s)])
        val = T.Compose([T.Resize(IMG_RES), T.ToTensor(), T.Normalize(n,s)])
        tr = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=tra)
        va = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val)
    def ld(ds, bs, sh):
        return DataLoader(ds, bs, sh, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return ld(tr, args.batch, True), ld(va, 256, False)

if not args.no_train:
    train_loader, val_loader = loaders()

# ───────────────── TRAIN LOOP UTILS ─────────────────
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
@contextlib.contextmanager
def autocast():
    with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.float16):
        yield

def try_compile(m):
    try:
        return torch.compile(m, backend="inductor")
    except Exception as e:
        warnings.warn(f"compile failed: {e}"); return m

def strip_prefix(state: Dict[str, torch.Tensor]):
    if all(k.startswith("_orig_mod.") for k in state):
        return {k[len("_orig_mod."):]: v for k, v in state.items()}
    return state

def save_clean(model: nn.Module, path: Path):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(base.state_dict(), path)

# ─────────────────── TRAINING ───────────────────

def train_one(name: str, model: nn.Module):
    ckpt = CHK_DIR / f"{DATASET}_{name}.pth"
    if ckpt.exists():
        print(f"[skip] {name}: checkpoint exists")
        return
    model.to(DEVICE, memory_format=torch.channels_last)
    model = try_compile(model)
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched= optim.lr_scheduler.OneCycleLR(opt, args.lr, args.epochs, len(train_loader))
    best = 0.0; t0 = time.time()
    t_loss=t_acc=v_loss=v_acc=[]
    for ep in range(args.epochs):
        # train
        model.train(); running_l=running_c=0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, memory_format=torch.channels_last); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(xb); loss = crit(out, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running_l += loss.item()*xb.size(0); running_c += (out.argmax(1)==yb).sum().item()
        tr_l = running_l/len(train_loader.dataset); tr_a = running_c/len(train_loader.dataset)
        # val
        model.eval(); run_l=run_c=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb=xb.to(DEVICE, memory_format=torch.channels_last); yb=yb.to(DEVICE)
                out=model(xb); loss=crit(out,yb)
                run_l+=loss.item()*xb.size(0); run_c+=(out.argmax(1)==yb).sum().item()
        va_l = run_l/len(val_loader.dataset); va_a = run_c/len(val_loader.dataset)
        print(f"{name} ep{ep+1:02}/{args.epochs}  tr {tr_l:.3f}/{tr_a:.2%}  val {va_l:.3f}/{va_a:.2%}")
        best = max(best, va_a)
        if va_a == best: save_clean(model, ckpt)
        sched.step()
        if time.time()-t0 > 14*60:
            print("[!] 15‑min budget reached, early stop"); break
    print(f"{name} best val acc: {best:.2%}")

# ─────────────────── ANALYSIS ───────────────────
@torch.no_grad()
def analyse(name: str, model: nn.Module):
    model.eval().to(DEVICE)
    blocks = [b for l in [model.layer1, model.layer2, model.layer3, model.layer4] for b in l]
    preprocess = T.Compose([T.Resize(256), T.CenterCrop(IMG_RES), T.ToTensor(),
                            T.Normalize([.485,.456,.406],[.229,.224,.225])])
    dummy = preprocess(Image.new("RGB", (IMG_RES, IMG_RES), (128,128,128))).unsqueeze(0).to(DEVICE)
    actL2m=[]; actL2s=[]; actRm=[]; actRs=[]; parL2m=[]; parL2s=[]; parRm=[]; parRs=[]
    for b in blocks:
        flat = torch.cat([p.detach().flatten() for p in b.parameters()])
        l2s  = torch.tensor([p.detach().norm() for p in b.parameters()])
        parL2m.append(l2s.mean().item()); parL2s.append(l2s.std().item())
        parRm.append(flat.mean().item()); parRs.append(flat.std().item())
    def hk(_,__,out):
        h = out.detach().float().squeeze(0)  # (C,H,W)
        flat = h.flatten(); actRm.append(flat.mean().item()); actRs.append(flat.std().item())
        l2 = h.flatten(1).norm(2, dim=0); actL2m.append(l2.mean().item()); actL2s.append(l2.std().item())
    hdls=[b.register_forward_hook(hk) for b in blocks]; _=model(dummy); [h.remove() for h in hdls]
    stats={"l2_mean":actL2m,"l2_std":actL2s,"raw_mean":actRm,"raw_std":actRs,
           "param_l2_mean":parL2m,"param_l2_std":parL2s,"param_raw_mean":parRm,"param_raw_std":parRs}
    (PLOT_DIR/f"{DATASET}_{name}.json").write_text(json.dumps(stats))

    # quick plots
    def p(vals,stds,title,y,tag):
        x=range(1,len(vals)+1); plt.figure(figsize=(8,4));
        plt.fill_between(x, np.array(vals)-np.array(stds), np.array(vals)+np.array(stds), alpha=0.25)
        plt.plot(x, vals, marker="o"); plt.title(title); plt.xlabel("Block"); plt.ylabel(y); plt.grid(ls="--", lw=0.4)
        plt.tight_layout(); plt.savefig(PLOT_DIR/f"{DATASET}_{name}_{tag}.png", dpi=300); plt.close()
    p(actL2m,actL2s,f"{name} ||act||₂","L2 norm","l2")
    p(actRm,actRs,f"{name} raw act","Activation","raw")
    p(parL2m,parL2s,f"{name} ||θ||₂","Param L2","param_l2")
    p(parRm,parRs,f"{name} raw θ","Param value","param_raw")

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    if not args.no_train:
        train_loader, val_loader = loaders()
        for k, m in MODELS.items():
            train_one(k, m)
    if not args.no_analysis:
        for k, m in MODELS.items():
            ckpt = CHK_DIR / f"{DATASET}_{k}.pth"
            state = torch.load(ckpt, weights_only=True, map_location="cpu")
            m.load_state_dict(strip_prefix(state))
            analyse(k, m)
    print("✓ complete")
