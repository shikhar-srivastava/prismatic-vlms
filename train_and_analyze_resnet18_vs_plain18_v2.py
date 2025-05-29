#!/usr/bin/env python3
# train_and_analyze_resnet_variants.py
#
# Purpose:
#   1. Train six networks (resnet14, resnet18, resnet34, plain14, plain18, plain34)
#      on one of {ImageNet, CIFAR-100, Tiny-ImageNet}, with a standard multi-step
#      schedule and no time restriction.
#   2. Analyze per-block activations (including whisker box plots with outlier
#      annotations) and parameter norms.

from __future__ import annotations
import argparse
import json
import os
import re
import warnings
import contextlib
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
import seaborn as sns

sns.set_theme(style="whitegrid")

# ───────────────────────── CONFIG ─────────────────────────
BASE = Path.cwd()
CHK_DIR = BASE / "checkpoints"
CHK_DIR.mkdir(exist_ok=True)

CURVE_DIR = BASE / "viz"
CURVE_DIR.mkdir(exist_ok=True)

PLOT_DIR = BASE / "viz" / "plots" / "act_analysis_rvp"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE / "datasets"
DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "A CUDA-capable GPU is required."

torch.backends.cudnn.benchmark = True

# ───────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["auto", "imagenet", "cifar100", "tiny"], default="auto")
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--subset", type=float, default=1.0, help="subset ratio for ImageNet training data")
parser.add_argument("--no_train", action="store_true", help="skip training if set")
parser.add_argument("--no_analysis", action="store_true", help="skip analysis if set")
parser.add_argument("--amp", action="store_true", default=True, help="use PyTorch AMP for training")
args = parser.parse_args()

IMAGENET_DIR = os.getenv("IMAGENET_DIR")

# Decide dataset automatically if needed
if args.dataset != "auto":
    DATASET = args.dataset
else:
    if IMAGENET_DIR and Path(IMAGENET_DIR).is_dir():
        DATASET = "imagenet"
    else:
        DATASET = "cifar100"

# We fix image resolution to 224 for all datasets
IMG_RES = 224
# For CIFAR/Tiny, we use 100 classes or 200 classes. For ImageNet, 1000 classes.
NUM_CLASSES = 1000 if DATASET in {"imagenet", "tiny"} else 100
print(f"[INFO] dataset={DATASET}, classes={NUM_CLASSES}")

# ────────────── DEFINE BLOCKS & RESNET ARCH ──────────────
class BasicBlock(nn.Module):
    """Residual BasicBlock: 2×3×3 with skip connection."""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.act(out)


class PlainBlock(nn.Module):
    """Plain BasicBlock: 2×3×3 with no residual connection."""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For plain net, if shape changes, do a projection (w/o adding residual).
        # self.projection = None
        # if stride != 1 or in_channels != out_channels:
        #     self.projection = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1,
        #                   stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        out = self.bn2(self.conv2(out))
        # if self.projection is not None:
        #     out = self.projection(out)
        return self.act(out)


class ResNet(nn.Module):
    """
    General ResNet with a given block type (BasicBlock or PlainBlock).
    The only difference: presence (BasicBlock) vs. absence (PlainBlock)
    of the residual skip.
    """
    def __init__(self, block, layers: List[int], num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        # first block
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        # subsequent
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ────────────── MODEL FACTORIES ──────────────
def resnet14(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [3, 2, 2, 2], num_classes=num_classes)

def resnet18(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def plain14(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [3, 2, 2, 2], num_classes=num_classes)

def plain18(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [2, 2, 2, 2], num_classes=num_classes)

def plain34(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [3, 4, 6, 3], num_classes=num_classes)

MODELS = {
    "resnet14": resnet14(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "plain14":  plain14(),
    "plain18":  plain18(),
    "plain34":  plain34(),
}


# ─────────── SCHEDULE CONFIG FOR DIFFERENT DEPTHS ───────────
# We define typical multi-step schedules. The pairs are (epochs, LR, milestones).
# For instance, if the tag has '14', we do 100 epochs, initial lr=0.1,
# with steps at epoch 50 and 75.  (You can adjust these to your preference.)
DEPTH_CFG = {
    14: (100, 0.1, [50, 75]),
    18: (150, 0.1, [75, 110]),
    34: (200, 0.1, [100, 150]),
}


# ─────────────────── DATA LOADERS ───────────────────
def make_loaders():
    """
    Returns train_loader, val_loader
    """
    if DATASET == "imagenet":
        root = Path(IMAGENET_DIR)
        tra = T.Compose([
            T.RandomResizedCrop(IMG_RES),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        val = T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_RES),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        train_ds = datasets.ImageFolder(root / "train", tra)
        if args.subset < 1.0:
            idx = np.random.RandomState(0).choice(len(train_ds),
                                                  int(args.subset * len(train_ds)),
                                                  replace=False)
            train_ds = Subset(train_ds, idx)
        val_ds = datasets.ImageFolder(root / "val", val)

    elif DATASET == "tiny":
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        arch = DATA_DIR / "tiny.zip"
        if not arch.exists():
            download_and_extract_archive(url, download_root=DATA_DIR, filename="tiny.zip")

        base = DATA_DIR / "tiny-imagenet-200"
        train_dir = base / "train"
        val_dir   = base / "val" / "images"
        n_mean, n_std = [0.480,0.448,0.397], [0.277,0.269,0.282]
        tra = T.Compose([
            T.Resize(IMG_RES),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(n_mean,n_std),
        ])
        val = T.Compose([
            T.Resize(IMG_RES),
            T.ToTensor(),
            T.Normalize(n_mean,n_std),
        ])
        train_ds = datasets.ImageFolder(train_dir, tra)
        val_ds   = datasets.ImageFolder(val_dir, val)

    else:
        # CIFAR-100
        n_mean, n_std = [0.5071,0.4865,0.4409], [0.2673,0.2564,0.2762]
        tra = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.Resize(IMG_RES),
            T.ToTensor(),
            T.Normalize(n_mean, n_std),
        ])
        val = T.Compose([
            T.Resize(IMG_RES),
            T.ToTensor(),
            T.Normalize(n_mean, n_std),
        ])
        train_ds = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=tra)
        val_ds   = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val)

    def ld(ds, bs, shuffle):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    train_loader = ld(train_ds, args.batch, True)
    val_loader   = ld(val_ds,   256,       False)
    return train_loader, val_loader


# ───────── TRAIN UTILITIES ─────────
def parse_depth_from_tag(tag: str) -> int:
    """
    E.g. "resnet14" or "plain18" -> returns 14 or 18 as integer
    """
    # simple parse
    m = re.search(r"(\d+)$", tag)
    if m:
        return int(m.group(1))
    return 18  # fallback

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

@contextlib.contextmanager
def autocast():
    with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.float16):
        yield

def try_compile(m: nn.Module) -> nn.Module:
    try:
        return torch.compile(m, backend="inductor")
    except Exception:
        return m

def strip_prefix(state_dict):
    """
    Remove any leading '_orig_mod.' keys left by torch.compile at save-time.
    """
    new_dict = {}
    for k, v in state_dict.items():
        k_new = re.sub(r"^_orig_mod\.", "", k)
        new_dict[k_new] = v
    return new_dict

def save_ckpt(model: nn.Module, path: Path):
    """Saves the underlying un-compiled model weights."""
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(base.state_dict(), path)

# ───────── TRAIN LOOP ─────────
def train(tag: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    ck = CHK_DIR / f"{DATASET}_{tag}.pth"
    if ck.exists():
        print(f"[skip] {tag} (checkpoint exists).")
        return

    # Decide epochs, LR, milestones
    depth = parse_depth_from_tag(tag)
    epochs, base_lr, ms = DEPTH_CFG.get(depth, (100, 0.1, [50, 75]))
    print(f"Training {tag} for {epochs} epochs, LR={base_lr}, milestones={ms}")

    model.to(DEVICE, memory_format=torch.channels_last)
    model = try_compile(model)

    crit = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.1)

    for ep in range(epochs):
        model.train()
        epoch_loss, correct = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, memory_format=torch.channels_last)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(xb)
                loss = crit(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc  = correct / len(train_loader.dataset)

        # validate
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, memory_format=torch.channels_last)
                yb = yb.to(DEVICE)
                out = model(xb)
                l = crit(out, yb)
                val_loss += l.item() * xb.size(0)
                val_correct += (out.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = val_correct / len(val_loader.dataset)

        print(f"{tag} epoch {ep+1}/{epochs}  "
              f"train {train_loss:.3f}/{train_acc:.2%}  val {val_loss:.3f}/{val_acc:.2%}")

        # checkpoint best
        # We won't track "best" by default here – if you prefer to store the best only:
        # if val_acc > best_acc: ...
        # but for simplicity let's just store final:
        # Or store best each time:
        save_ckpt(model, ck)

        scheduler.step()

# ───────── PLOTTING UTILITIES ─────────
def _plot_box(activations: List[np.ndarray],
              outliers: List[List[Tuple[float,int,int]]],
              label: str):
    """
    Create a whisker box plot for raw activation distribution across blocks,
    highlighting outliers.
    """
    fig_h = max(3, 0.5 * len(activations) + 4)
    plt.figure(figsize=(12, fig_h), dpi=300)
    data = [act.flatten() for act in activations]

    b = plt.boxplot(
        data,
        vert=True,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "r", "alpha": 0.6},
    )

    # overlay custom outliers
    for i, outs in enumerate(outliers):
        if not outs:
            continue
        xs = np.full(len(outs), i + 1)
        ys = [o[0] for o in outs]
        plt.scatter(xs, ys, c="red", marker="x")
        for x_, (val, fpos, ch) in zip(xs, outs):
            plt.text(x_ + 0.15, val, f"ch{ch}/{fpos}", fontsize=6, ha="left")

    plt.xlabel("Block")
    plt.ylabel("Raw activation value")
    plt.title(f"{label} – raw activation distribution (whisker box)", weight="bold")
    plt.grid(ls="--", lw=0.4, axis="y")
    plt.tight_layout()
    outpath = PLOT_DIR / f"{label}_box.png"
    plt.savefig(outpath)
    plt.close()

# ───────── ANALYSIS ─────────
@torch.no_grad()
def analyse(tag: str, model: nn.Module, results: Dict[str, Dict[str, List[float]]]):
    """
    1. Collect per-block activation L2 stats
    2. Activation box-plot
    3. Parameter L2 stats
    """
    model.eval().to(DEVICE)
    # gather blocks
    blocks = []
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for b in layer:
            blocks.append(b)

    # Get a single random image from CIFAR-100 test set
    if DATASET == "cifar100":
        print("Using CIFAR-100 test set")
        n_mean, n_std = [0.5071,0.4865,0.4409], [0.2673,0.2564,0.2762]
        val_transform = T.Compose([
            T.Resize(IMG_RES),
            T.ToTensor(),
            T.Normalize(n_mean, n_std),
        ])
        test_ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_transform)
        # Get a random sample
        random_idx = torch.randint(0, len(test_ds), (1,)).item()
        random_image, _ = test_ds[random_idx]
        dummy = random_image.unsqueeze(0).to(DEVICE)  # Add batch dimension
    else:
        print("Using dummy (zero image) test set")
        # Fallback to original dummy input for other datasets
        dummy = torch.zeros((1,3,IMG_RES,IMG_RES), dtype=torch.float32)
        # approximate "centered" input
        dummy[:,0,:,:] = 0.485 / 0.229
        dummy[:,1,:,:] = 0.456 / 0.224
        dummy[:,2,:,:] = 0.406 / 0.225
        dummy = dummy.to(DEVICE)

    aL2m, aL2s = [], []
    pL2m, pL2s = [], []
    activations = []
    outliers = []

    # param L2 stats
    for blk in blocks:
        flat = torch.cat([p.detach().flatten() for p in blk.parameters()])
        block_l2s = torch.tensor([p.detach().norm() for p in blk.parameters()])
        pL2m.append(block_l2s.mean().item())
        pL2s.append(block_l2s.std().item())

    # activation hooking
    def hook_fn(_, __, out):
        outf = out.detach().float().squeeze(0)  # shape [C,H,W]
        activations.append(outf.cpu().numpy())

        # L2 across channels
        fc = outf.flatten(1)  # [C, H*W]
        norms_c = fc.norm(dim=1, p=2)
        aL2m.append(norms_c.mean().item())
        aL2s.append(norms_c.std().item())

    hdls = []
    for blk in blocks:
        hdls.append(blk.register_forward_hook(hook_fn))

    _ = model(dummy)
    for h in hdls:
        h.remove()

    # identify numeric outliers
    for arr in activations:
        if arr.ndim == 3:
            C, H, W = arr.shape
            flat = arr.flatten()
            if flat.size > 40:
                idx = np.argsort(flat)
                pick = np.concatenate([idx[:20], idx[-20:]])
            else:
                pick = np.arange(flat.size)
            outs = []
            for i in pick:
                v = float(flat[i])
                c, r, w_ = np.unravel_index(int(i), (C, H, W))
                outs.append((v, r*W + w_, c))
            outliers.append(outs)
        else:
            outliers.append([])

    # store stats
    stats = {
        "l2_mean": aL2m,
        "l2_std":  aL2s,
        "param_l2_mean": pL2m,
        "param_l2_std":  pL2s,
    }
    results[tag] = stats
    (PLOT_DIR / f"{DATASET}_{tag}.json").write_text(json.dumps(stats))

    # whisker box
    _plot_box(activations, outliers, label=f"{DATASET}_{tag}")

    # also produce line plots for each model
    def _plot_line(vals, stds, title_, ylabel_, fname_):
        x = range(1, len(vals)+1)
        plt.figure(figsize=(8,4), dpi=300)
        lb = [v - s for v, s in zip(vals, stds)]
        ub = [v + s for v, s in zip(vals, stds)]
        plt.fill_between(x, lb, ub, alpha=0.15, color="blue")
        plt.plot(x, vals, marker="o", color="blue", linewidth=1.2)
        plt.title(title_, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylabel_)
        plt.grid(ls="--", lw=0.4)
        plt.tight_layout()
        outn = PLOT_DIR / f"{fname_}.png"
        plt.savefig(outn)
        plt.close()

    _plot_line(aL2m, aL2s, f"{tag} activation L2", "Activation L2",
               f"{DATASET}_{tag}_act_l2")
    _plot_line(pL2m, pL2s, f"{tag} param L2", "Parameter L2",
               f"{DATASET}_{tag}_param_l2")

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    # possibly create data loaders
    train_loader, val_loader = None, None
    if not args.no_train:
        train_loader, val_loader = make_loaders()
        for model_name, model_obj in MODELS.items():
            train(model_name, model_obj, train_loader, val_loader)

    if not args.no_analysis:
        if train_loader is None or val_loader is None:
            # Just to ensure shapes etc. for dummy input, though we only
            # actually need a single forward pass with a dummy image
            _ = make_loaders()

        all_stats = {}
        for model_name, model_obj in MODELS.items():
            ck_path = CHK_DIR / f"{DATASET}_{model_name}.pth"
            state = torch.load(ck_path, map_location="cpu")
            model_obj.load_state_dict(strip_prefix(state))
            analyse(model_name, model_obj, all_stats)

        # Combined L2 plot across all models - Professional Version
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        # Define color scheme and line styles
        colors = {
            'resnet': '#2E86AB',  # Professional blue
            'plain': '#A23B72'   # Professional magenta
        }
        
        line_styles = {
            14: '-',      # solid
            18: '--',     # dashed  
            34: '-.'      # dash-dot
        }
        
        line_widths = {
            14: 2.0,
            18: 2.2,
            34: 2.4
        }
        
        # Group and plot models
        legend_elements = []
        
        for tag, stats in all_stats.items():
            # Parse model type and depth
            model_type = 'resnet' if tag.startswith('resnet') else 'plain'
            depth = parse_depth_from_tag(tag)
            
            x_vals = np.array(range(1, len(stats["l2_mean"]) + 1))
            means = np.array(stats["l2_mean"])
            stds = np.array(stats["l2_std"])
            
            color = colors[model_type]
            linestyle = line_styles[depth]
            linewidth = line_widths[depth]
            alpha_fill = 0.15
            
            # Plot mean line
            line = ax.plot(x_vals, means, 
                          color=color, 
                          linestyle=linestyle, 
                          linewidth=linewidth,
                          label=f'{model_type.upper()}-{depth}',
                          marker='o' if depth == 18 else ('s' if depth == 14 else '^'),
                          markersize=6,
                          markerfacecolor=color,
                          markeredgecolor='white',
                          markeredgewidth=1,
                          alpha=0.9)
            
            # Add error band (mean ± std)
            ax.fill_between(x_vals, means - stds, means + stds, 
                           color=color, alpha=alpha_fill, 
                           linewidth=0)
            
            legend_elements.append(line[0])
        
        # Enhance the plot aesthetics
        ax.set_xlabel('Block Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Activation L2 Norm', fontsize=14, fontweight='bold')
        ax.set_title(f'Per-Block Activation Analysis: ResNet vs Plain Networks\n'
                    f'Dataset: {DATASET.upper()} | Error bands show ±1 standard deviation',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=12, 
                      colors='#333333', width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Set x-axis to show integer ticks only
        ax.set_xticks(range(1, max([len(s["l2_mean"]) for s in all_stats.values()]) + 1))
        
        # Custom legend with better organization
        resnet_elements = [el for el in legend_elements if 'RESNET' in el.get_label()]
        plain_elements = [el for el in legend_elements if 'PLAIN' in el.get_label()]
        
        # Sort by depth
        resnet_elements.sort(key=lambda x: int(x.get_label().split('-')[1]))
        plain_elements.sort(key=lambda x: int(x.get_label().split('-')[1]))
        
        legend1 = ax.legend(resnet_elements, [el.get_label() for el in resnet_elements],
                           title='ResNet Models', loc='upper left', 
                           fontsize=11, title_fontsize=12, framealpha=0.9,
                           fancybox=True, shadow=True)
        legend1.get_title().set_fontweight('bold')
        
        legend2 = ax.legend(plain_elements, [el.get_label() for el in plain_elements],
                           title='Plain Models', loc='upper right',
                           fontsize=11, title_fontsize=12, framealpha=0.9,
                           fancybox=True, shadow=True)
        legend2.get_title().set_fontweight('bold')
        
        # Add the first legend back
        ax.add_artist(legend1)
        
        # Add subtle background color
        ax.set_facecolor('#fafafa')
        
        # Tight layout with padding
        plt.tight_layout(pad=2.0)
        
        # Save with high quality
        output_path = PLOT_DIR / f"combined_l2_professional_{DATASET}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"[INFO] Professional combined plot saved to: {output_path}")

    print("✓ done")
