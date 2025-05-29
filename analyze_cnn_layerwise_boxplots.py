"""
Scratch ResNets vs Plain CNNs – identical depths, skip‑only difference
====================================================================
Implements and trains six networks **from scratch**:

| tag   | residual? | blocks cfg |
|-------|-----------|------------|
| plain14 | ❌ | [3, 2, 2, 2] |
| res14   | ✅ | [3, 2, 2, 2] |
| plain18 | ❌ | [2, 2, 2, 2] |
| res18   | ✅ | [2, 2, 2, 2] |
| plain34 | ❌ | [3, 4, 6, 3] |
| res34   | ✅ | [3, 4, 6, 3] |

**Key points**
--------------
* Residual blocks add identity → *ReLU*; plain blocks omit the add.  The
  **forward hooks** are attached to **each block module**, capturing the
  *post‑addition, post‑activation* tensor exactly as required.
* Auto‑dataset → ImageNet if `$IMAGENET_DIR`, else CIFAR‑100 (or Tiny via
  `--dataset tiny`).
* GPU optimised: AMP, channels‑last, `torch.compile`, OneCycleLR, ≤ 15 min per
  model.
* Outputs: checkpoints, training curves, per‑model JSON + PNG stats, combined
  mean‑L2 plot, and mid‑block box plot in `viz/plots/act_analysis_rvp/`.

Run
---
```bash
# full pipeline (auto‑downloads CIFAR‑100)
python train_and_analyze_resnets_from_scratch.py

# reuse existing ckpts, only analysis / plots
python train_and_analyze_resnets_from_scratch.py --no_train
```
"""

from __future__ import annotations
import argparse, json, os, time, warnings, contextlib, re, gc
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Subset
from PIL import Image

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "CUDA GPU required"

# ───────── paths ─────────
BASE = Path.cwd()
CHK = BASE/"checkpoints"; CHK.mkdir(exist_ok=True)
PLOT= BASE/"viz"/"plots"/"act_analysis_rvp"; PLOT.mkdir(parents=True, exist_ok=True)
DATA= BASE/"datasets"; DATA.mkdir(exist_ok=True)

# ───────── cli ─────────
pa = argparse.ArgumentParser()
pa.add_argument("--dataset", choices=["auto","imagenet","cifar100","tiny"], default="auto")
pa.add_argument("--epochs", type=int, default=-1)
pa.add_argument("--batch", type=int, default=1024)
pa.add_argument("--lr", type=float, default=0.4)
pa.add_argument("--subset", type=float, default=1.0)
pa.add_argument("--no_train", action="store_true")
pa.add_argument("--no_analysis", action="store_true")
pa.add_argument("--amp", action="store_true", default=True)
args = pa.parse_args()

IMAGENET_DIR = os.getenv("IMAGENET_DIR")
DATASET = args.dataset if args.dataset != "auto" else ("imagenet" if IMAGENET_DIR and Path(IMAGENET_DIR).is_dir() else "cifar100")
NUM_CLS = 1000 if DATASET in {"imagenet","tiny"} else 100
IMG_RES = 224
# epochs schedule
if args.epochs>0: EPOCHS=args.epochs
else: EPOCHS=20 if DATASET=="cifar100" else 15 if DATASET in {"tiny"} or args.subset<0.15 else 5
print(f"[CFG] {DATASET=} {EPOCHS=} {args.batch=}")

# ───────── blocks ─────────
class _BlockBase(nn.Module):
    def __init__(self,in_c,out_c,stride=1): super().__init__()
    def _conv3(self,ic,oc,s): return nn.Conv2d(ic,oc,3,s,1,bias=False)

class PlainBlock(_BlockBase):
    def __init__(self,in_c,out_c,stride=1):
        super().__init__(in_c,out_c,stride)
        self.conv1=self._conv3(in_c,out_c,stride); self.bn1=nn.BatchNorm2d(out_c)
        self.conv2=self._conv3(out_c,out_c,1); self.bn2=nn.BatchNorm2d(out_c); self.act=nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.act(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))
        return self.act(x)

class ResBlock(_BlockBase):
    def __init__(self,in_c,out_c,stride=1):
        super().__init__(in_c,out_c,stride)
        self.conv1=self._conv3(in_c,out_c,stride); self.bn1=nn.BatchNorm2d(out_c)
        self.conv2=self._conv3(out_c,out_c,1); self.bn2=nn.BatchNorm2d(out_c); self.act=nn.ReLU(inplace=True)
        self.down = nn.Identity() if (in_c==out_c and stride==1) else nn.Sequential(nn.Conv2d(in_c,out_c,1,stride,bias=False),nn.BatchNorm2d(out_c))
    def forward(self,x):
        identity=self.down(x)
        out=self.act(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=self.act(out+identity)  # post-add, post-activation
        return out

# ───────── network factory ─────────
CFG={"14":[3,2,2,2],"18":[2,2,2,2],"34":[3,4,6,3]}

def make_layer(block,in_c,out_c,n,stride):
    layers=[block(in_c,out_c,stride)]
    layers+= [block(out_c,out_c) for _ in range(n-1)]
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self,cfg,block,nc=NUM_CLS):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(3,2,1))
        sizes=[64,128,256,512]
        self.layer1=make_layer(block,64 ,sizes[0],cfg[0],1)
        self.layer2=make_layer(block,sizes[0],sizes[1],cfg[1],2)
        self.layer3=make_layer(block,sizes[1],sizes[2],cfg[2],2)
        self.layer4=make_layer(block,sizes[2],sizes[3],cfg[3],2)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,nc))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.stem(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        return self.head(x)

MODELS:Dict[str,nn.Module]={}
for d,c in CFG.items():
    MODELS[f"plain{d}"]=Net(c,PlainBlock)
    MODELS[f"res{d}"]=Net(c,ResBlock)

# ───────── data ─────────

def make_loaders():
    if DATASET=="imagenet":
        tra=T.Compose([T.RandomResizedCrop(IMG_RES),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        val=T.Compose([T.Resize(256),T.CenterCrop(IMG_RES),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        root=Path(IMAGENET_DIR); tr=datasets.ImageFolder(root/"train",tra)
        if args.subset<1.0:
            idx=np.random.RandomState(0).choice(len(tr),int(args.subset*len(tr)),replace=False); tr=Subset(tr,idx.tolist())
        va=datasets.ImageFolder(root/"val",val)
    elif DATASET=="tiny":
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"; arch=DATA/"tiny.zip"
        if not arch.exists(): download_and_extract_archive(url,str(DATA),filename="tiny.zip")
        base=DATA/"tiny-imagenet-200"; tr_d=base/"train"; va_d=base/"val/images"
        n,s=[.480,.448,.397],[.277,.269,.282]
        tra=T.Compose([T.Resize(IMG_RES),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(n,s)])
        val=T.Compose([T.Resize(IMG_RES),T.ToTensor(),T.Normalize(n,s)])
        tr=datasets.ImageFolder(tr_d,tra); va=datasets.ImageFolder(va_d,val)
    else:
        n,s=[.5071,.4865,.4409],[.2673,.2564,.2762]
        tra=T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(32,4),T.Resize(IMG_RES),T.ToTensor(),T.Normalize(n,s)])
        val=T.Compose([T.Resize(IMG_RES),T.ToTensor(),T.Normalize(n,s)])
        tr=datasets.CIFAR100(DATA,train=True,download=True,transform=tra)
        va=datasets.CIFAR100(DATA,train=False,download=True,transform=val)
    def ld(ds,bs,sh): return DataLoader(ds,bs,sh,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4)
    return ld(tr,args.batch,True), ld(va,256,False)

if not args.no_train:
    train_loader,val_loader=make_loaders()

# ───────── helpers ─────────
scaler=torch.cuda.amp.GradScaler(enabled=args.amp)
@contextlib.contextmanager
def autocast():
    with torch.cuda.amp.autocast(enabled=args.amp,dtype=torch.float16): yield

def compile_try(m):
    try:return torch.compile(m,backend="inductor")
    except Exception: return m

def save_clean(m,p):
    base=m._orig_mod if hasattr(m,"_orig_mod") else m; torch.save(base.state_dict(),p)

def strip(st):
    return {re.sub(r"^_orig_mod\.","",k):v for k,v in st.items()}

# ───────── training ─────────

def train(tag,m):
    ck=CHK/f"{DATASET}_{tag}.pth"; 
    if ck.exists(): print(f"[skip] {tag}"); return
    m.to(DEVICE,memory_format=torch.channels_last); m=compile_try(m)
    crit=nn.CrossEntropyLoss(); opt=optim.SGD(m.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)
    sched=optim.lr_scheduler.OneCycleLR(opt,args.lr,EPOCHS,len(train_loader))
    best=0; t0=time.time()
    for ep in range(EPOCHS):
        m.train(); c=n=0
        for x,y in train_loader:
            x=x.to(DEVICE,memory_format=torch.channels_last); y=y.to(DEVICE);
            opt.zero_grad(set_to_none=True)
            with autocast(): out=m(x); loss=crit(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); c+=(out.argmax(1)==y).sum().item(); n+=y.size(0)
        tr_a=c/n; m.eval(); c=n=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(DEVICE,memory_format=torch.channels_last); y=y.to(DEVICE); out=m(x); c+=(out.argmax(1)==y).sum().item(); n+=y.size(0)
        va_a=c/n; print(f"{tag} ep{ep+1}/{EPOCHS} {va_a:.2%}")
        if va_a>best: best=va_a; save_clean(m,ck)
        sched.step();
        if time.time()-t0>14*60: break

# ───────── analysis ─────────
@torch.no_grad()
def analyse(tag,m,collect):
    m.eval().to(DEVICE)
    blocks=[b for l in [m.layer1,m.layer2,m.layer3,m.layer4] for b in l]
    tf=T.Compose([T.Resize(256),T.CenterCrop(IMG_RES),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    x=tf(Image.new("RGB",(IMG_RES,IMG_RES),(128,128,128))).unsqueeze(0).to(DEVICE)
    L2m; L2s=[] ,[]
    def hk(_,__,o):
        h=o.detach().float().squeeze(0); l2=h.flatten(1).norm(2,dim=0); L2m.append(l2.mean().item()); L2s.append(l2.std().item())
    hd=[b.register_forward_hook(hk) for b in blocks]; _=m(x); [h.remove() for h in hd]
    collect[tag]={"l2_mean":L2m,"l2_std":L2s}; (PLOT/f"{DATASET}_{tag}.json").write_text(json.dumps(collect[tag]))

# ───────── main ─────────
if not args.no_train:
    for t,m in MODELS.items(): train(t,m)
if not args.no_analysis:
    stats={}
    for t,m in MODELS.items(): m.load_state_dict(strip(torch.load(CHK/f"{DATASET}_{t}.pth",weights_only=True,map_location="cpu"))); analyse(t,m,stats)
    # combined plots
    plt.figure(figsize=(10,6))
    for t,s in stats.items(): plt.plot(range(1,len(s["l2_mean"])+1),s["l2_mean"],label=t,lw=1.2)
    plt.title("Activation L2 (mean)"); plt.xlabel("Block"); plt.ylabel("L2 norm"); plt.grid(ls="--",lw=0.4); plt.legend(ncol=2,fontsize=6); plt.tight_layout(); plt.savefig(PLOT/f"combined_l2_{DATASET}.png",dpi=300); plt.close()
    plt.figure(figsize=(12,6))
    mids=[s["l2_mean"][len(s["l2_mean"] )//2-4:len(s["l2_mean"] )//2+4] for s in stats.values()]
    plt.boxplot(mids,patch_artist=True); plt.xticks(range(1,len(stats)+1),stats.keys(),rotation=45,ha="right");
    plt.ylabel("L2 (mid blocks)"); plt.title("Mid-block activation distribution"); plt.grid(ls="--",axis="y",lw=0.4); plt.tight_layout(); plt.savefig
