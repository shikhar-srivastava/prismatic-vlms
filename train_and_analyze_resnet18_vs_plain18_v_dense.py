#!/usr/bin/env python3
# train_and_analyze_resnet_vs_densenet_variants.py
#
# Purpose:
#   1. Train nine networks (resnet14, resnet18, resnet34, plain14, plain18, plain34,
#      densenet14, densenet18, densenet34) on one of {ImageNet, CIFAR-100, Tiny-ImageNet},
#      with a standard multi-step schedule and no time restriction.
#   2. Analyze per-block activations (including whisker box plots with outlier
#      annotations) and parameter norms.
#   3. Compare ResNet vs Plain vs DenseNet architectures with equivalent depths.

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
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

PLOT_DIR = BASE / "viz" / "plots" / "act_analysis_rvd_retry"  # ResNet vs Dense
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
parser.add_argument("--train_dense_only", action="store_true", help="only train DenseNet models")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        out = self.bn2(self.conv2(out))
        return self.act(out)


# ────────────── DENSENET IMPLEMENTATION ──────────────
class _DenseLayer(nn.Module):
    """
    DenseNet Layer: BatchNorm + ReLU + Conv1x1 + BatchNorm + ReLU + Conv3x3
    Each layer adds 'growth_rate' number of feature maps.
    """
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        # Bottleneck layer
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        
        # Output layer
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            concat_features = x
        else:
            # x is a list of tensors to concatenate
            concat_features = torch.cat(x, 1)
            
        # Bottleneck
        out = self.conv1(self.relu1(self.norm1(concat_features)))
        
        # Output conv
        out = self.conv2(self.relu2(self.norm2(out)))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
            
        return out


class _DenseBlock(nn.Module):
    """
    Dense Block: Contains multiple DenseLayers with feature concatenation.
    Each layer receives all previous feature maps as input.
    """
    def __init__(self, num_layers: int, num_input_features: int, 
                 growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.layers.append(layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
            
        return torch.cat(features, 1)


class _Transition(nn.Module):
    """
    Transition layer: BatchNorm + ReLU + Conv1x1 + AvgPool2d
    Reduces spatial dimensions and can compress features.
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, 
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


class ResNet(nn.Module):
    """
    General ResNet with a given block type (BasicBlock or PlainBlock).
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


class DenseNet(nn.Module):
    """
    Custom DenseNet implementation comparable to ResNet architectures.
    
    Key Design Decisions:
    1. Uses similar stem (7x7 conv + maxpool) as ResNet for fair comparison
    2. Block configurations designed to match ResNet depths (14, 18, 34 layers)
    3. Growth rate and compression tuned for similar parameter counts
    4. Four dense blocks with transitions, similar to ResNet's four layer groups
    """
    def __init__(self, block_config: tuple, growth_rate: int = 12, 
                 num_init_features: int = 64, bn_size: int = 4, 
                 compression: float = 0.5, drop_rate: float = 0.0, num_classes: int = 1000):
        super().__init__()
        
        # Stem - similar to ResNet for fair comparison
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


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

def densenet14(num_classes=NUM_CLASSES):
    """
    DenseNet-14: Comparable to ResNet-14 in both depth and parameters
    - Block config: (3, 4, 4, 3) = 14 dense layers total
    - Growth rate: 64 (optimized for ~11M parameters)
    - Compression: 0.8 (reduced compression to keep more parameters)
    - Target: ~11M parameters (similar to ResNet-14)
    - Dense connections within each block
    """
    return DenseNet(block_config=(3, 4, 4, 3), growth_rate=64, compression=0.8, num_classes=num_classes)

def densenet18(num_classes=NUM_CLASSES):
    """
    DenseNet-18: Comparable to ResNet-18 in both depth and parameters
    - Block config: (4, 4, 4, 6) = 18 dense layers total
    - Growth rate: 64 (optimized for ~11M parameters)
    - Compression: 0.8 (reduced compression to keep more parameters)
    - Target: ~11M parameters (similar to ResNet-18)
    - More balanced layer distribution
    """
    return DenseNet(block_config=(4, 4, 4, 6), growth_rate=64, compression=0.8, num_classes=num_classes)

def densenet34(num_classes=NUM_CLASSES):
    """
    DenseNet-34: Comparable to ResNet-34 in both depth and parameters
    - Block config: (6, 8, 12, 8) = 34 dense layers total  
    - Growth rate: 64 (optimized for ~21M parameters)
    - Compression: 0.75 (balanced compression for parameter control)
    - Target: ~21M parameters (similar to ResNet-34)
    - Deeper dense blocks matching ResNet-34 complexity
    """
    return DenseNet(block_config=(6, 8, 12, 8), growth_rate=64, compression=0.75, num_classes=num_classes)

# Model selection based on training mode
if args.train_dense_only:
    MODELS = {
        "densenet14": densenet14(),
        "densenet18": densenet18(),
        "densenet34": densenet34(),
    }
else:
    MODELS = {
        "resnet14": resnet14(),
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "plain14":  plain14(),
        "plain18":  plain18(),
        "plain34":  plain34(),
        "densenet14": densenet14(),
        "densenet18": densenet18(),
        "densenet34": densenet34(),
    }


# ─────────── SCHEDULE CONFIG FOR DIFFERENT DEPTHS ───────────
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
    E.g. "resnet14" or "plain18" or "densenet34" -> returns 14 or 18 or 34 as integer
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
    
    # Skip training for ResNet/Plain models if using existing checkpoints
    if not args.train_dense_only and not tag.startswith("densenet") and ck.exists():
        print(f"[skip] {tag} (using existing checkpoint).")
        return
    elif tag.startswith("densenet") and ck.exists():
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

    # Track training metrics for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []

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

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f"{tag} epoch {ep+1}/{epochs}  "
              f"train {train_loss:.3f}/{train_acc:.2%}  val {val_loss:.3f}/{val_acc:.2%}")

        save_ckpt(model, ck)
        scheduler.step()

    # Save training curves
    save_training_plots(tag, train_losses, train_accs, val_losses, val_accs, learning_rates, epochs)

def save_training_plots(tag: str, train_losses: List[float], train_accs: List[float], 
                       val_losses: List[float], val_accs: List[float], 
                       learning_rates: List[float], epochs: int):
    """
    Save training curve plots for loss, accuracy, and learning rate.
    """
    epochs_range = range(1, epochs + 1)
    
    # Create training curves directory
    curves_dir = PLOT_DIR / "training_curves"
    curves_dir.mkdir(exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(12, 4), dpi=300)
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'{tag.upper()} - Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [acc * 100 for acc in train_accs], 'b-', label='Training Acc', linewidth=2)
    plt.plot(epochs_range, [acc * 100 for acc in val_accs], 'r-', label='Validation Acc', linewidth=2)
    plt.title(f'{tag.upper()} - Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Learning Rate
    plt.subplot(1, 3, 3)
    plt.semilogy(epochs_range, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    plt.title(f'{tag.upper()} - Learning Rate', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = curves_dir / f"{DATASET}_{tag}_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Training curves saved to: {plot_path}")
    
    # Also save individual high-quality plots
    _save_individual_training_plots(tag, train_losses, train_accs, val_losses, val_accs, 
                                   learning_rates, epochs, curves_dir)

def _save_individual_training_plots(tag: str, train_losses: List[float], train_accs: List[float], 
                                   val_losses: List[float], val_accs: List[float], 
                                   learning_rates: List[float], epochs: int, curves_dir: Path):
    """
    Save individual high-quality training plots.
    """
    epochs_range = range(1, epochs + 1)
    
    # Define colors for consistency
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Magenta
    lr_color = '#F18F01'     # Orange
    
    # Loss plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs_range, train_losses, color=train_color, linewidth=2.5, 
             label='Training Loss', marker='o', markersize=4, alpha=0.8)
    plt.plot(epochs_range, val_losses, color=val_color, linewidth=2.5, 
             label='Validation Loss', marker='s', markersize=4, alpha=0.8)
    plt.title(f'{tag.upper()} Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#fafafa')
    
    # Add min validation loss annotation
    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    min_val_loss = min(val_losses)
    plt.annotate(f'Min Val Loss: {min_val_loss:.3f}\nEpoch: {min_val_loss_epoch}', 
                xy=(min_val_loss_epoch, min_val_loss), xytext=(10, 10), 
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(curves_dir / f"{DATASET}_{tag}_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 6), dpi=300)
    train_accs_pct = [acc * 100 for acc in train_accs]
    val_accs_pct = [acc * 100 for acc in val_accs]
    
    plt.plot(epochs_range, train_accs_pct, color=train_color, linewidth=2.5, 
             label='Training Accuracy', marker='o', markersize=4, alpha=0.8)
    plt.plot(epochs_range, val_accs_pct, color=val_color, linewidth=2.5, 
             label='Validation Accuracy', marker='s', markersize=4, alpha=0.8)
    plt.title(f'{tag.upper()} Training and Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#fafafa')
    
    # Add max validation accuracy annotation
    max_val_acc_epoch = val_accs_pct.index(max(val_accs_pct)) + 1
    max_val_acc = max(val_accs_pct)
    plt.annotate(f'Max Val Acc: {max_val_acc:.1f}%\nEpoch: {max_val_acc_epoch}', 
                xy=(max_val_acc_epoch, max_val_acc), xytext=(10, -30), 
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(curves_dir / f"{DATASET}_{tag}_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning rate plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.semilogy(epochs_range, learning_rates, color=lr_color, linewidth=2.5, 
                 marker='^', markersize=4, alpha=0.8, label='Learning Rate')
    plt.title(f'{tag.upper()} Learning Rate Schedule', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(curves_dir / f"{DATASET}_{tag}_learning_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training metrics as JSON
    metrics = {
        'epochs': list(epochs_range),
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_losses': val_losses,
        'val_accuracies': val_accs,
        'learning_rates': learning_rates,
        'best_val_loss': min(val_losses),
        'best_val_loss_epoch': val_losses.index(min(val_losses)) + 1,
        'best_val_accuracy': max(val_accs),
        'best_val_accuracy_epoch': val_accs.index(max(val_accs)) + 1,
        'final_train_accuracy': train_accs[-1],
        'final_val_accuracy': val_accs[-1]
    }
    
    metrics_path = curves_dir / f"{DATASET}_{tag}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Training metrics saved to: {metrics_path}")
    print(f"[INFO] Best validation accuracy: {max(val_accs):.4f} at epoch {val_accs.index(max(val_accs)) + 1}")
    print(f"[INFO] Best validation loss: {min(val_losses):.4f} at epoch {val_losses.index(min(val_losses)) + 1}")

# ───────── PLOTTING UTILITIES ─────────
def _compute_top_stats(activations):
    """
    Compute top1, top2, top3, and median activations for each layer.
    
    Args:
        activations: List of numpy arrays containing activations for each layer
        
    Returns:
        Dict with keys 'top1', 'top2', 'top3', 'median' and values as lists
    """
    top1_vals, top2_vals, top3_vals, median_vals = [], [], [], []
    
    for act in activations:
        flat = act.flatten()
        if flat.size == 0:
            top1_vals.append(0.0)
            top2_vals.append(0.0) 
            top3_vals.append(0.0)
            median_vals.append(0.0)
            continue
            
        # Sort in descending order to get top values
        sorted_vals = np.sort(flat)[::-1]
        
        top1 = sorted_vals[0] if len(sorted_vals) >= 1 else 0.0
        top2 = sorted_vals[1] if len(sorted_vals) >= 2 else top1
        top3 = sorted_vals[2] if len(sorted_vals) >= 3 else top2
        median = np.median(flat)
        
        top1_vals.append(float(top1))
        top2_vals.append(float(top2))
        top3_vals.append(float(top3))
        median_vals.append(float(median))
    
    return {
        'top1': top1_vals,
        'top2': top2_vals, 
        'top3': top3_vals,
        'median': median_vals
    }

def _plot_top_activations(top_stats, label, safe_label):
    """
    Create a professional plot showing top1, top2, top3, and median activations per layer.
    """
    if not top_stats or not top_stats['top1']:
        return
        
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    x_vals = np.array(range(1, len(top_stats['top1']) + 1))
    
    # Professional color scheme with high contrast and visual hierarchy
    colors = {
        'top1': '#C4261D',    # Bold red for highest activation
        'top2': '#F18F01',    # Vibrant orange for second highest
        'top3': '#F4B942',    # Golden yellow for third highest
        'median': '#2E86AB'   # Professional blue for median
    }
    
    line_styles = {
        'top1': '-',     # solid for top values
        'top2': '-',     # solid for top values
        'top3': '-',     # solid for top values 
        'median': '--'   # dashed to distinguish median
    }
    
    line_widths = {
        'top1': 3.0,     # Thickest for most important
        'top2': 2.5,
        'top3': 2.2,
        'median': 2.5    # Thick for median as reference
    }
    
    markers = {
        'top1': 'o',     # Circle for top1
        'top2': 's',     # Square for top2
        'top3': '^',     # Triangle for top3
        'median': 'D'    # Diamond for median
    }
    
    marker_sizes = {
        'top1': 8,
        'top2': 7,
        'top3': 6,
        'median': 7
    }
    
    # Plot each statistic with professional styling
    legend_elements = []
    for stat_name in ['top1', 'top2', 'top3', 'median']:
        values = np.array(top_stats[stat_name])
        
        # Create display label
        if stat_name == 'median':
            display_label = 'Median'
        else:
            display_label = f'Top-{stat_name[-1]}'
        
        line = ax.plot(x_vals, values,
                      color=colors[stat_name],
                      linestyle=line_styles[stat_name], 
                      linewidth=line_widths[stat_name],
                      label=display_label,
                      marker=markers[stat_name],
                      markersize=marker_sizes[stat_name],
                      markerfacecolor=colors[stat_name],
                      markeredgecolor='white',
                      markeredgewidth=1.5,
                      alpha=0.9,
                      zorder=10 if stat_name == 'top1' else 5)
        legend_elements.append(line[0])
    
    # Enhance the plot aesthetics
    ax.set_xlabel('Block Index', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Activation Value', fontsize=14, fontweight='bold', color='#333333') 
    ax.set_title(f'{label} - Top Activation Analysis\n'
                f'Block-wise peak activations and median reference values',
                fontsize=16, fontweight='bold', pad=25, color='#333333')
    
    # Customize grid for better readability
    ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add minor grid for finer granularity
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', linewidth=0.5)
    ax.minorticks_on()
    
    # Customize spines with professional appearance
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color('#333333')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12,
                  colors='#333333', width=1.3, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3, colors='#666666')
    
    # Set x-axis to show integer ticks only
    max_blocks = len(top_stats['top1'])
    ax.set_xticks(range(1, max_blocks + 1))
    
    # Professional legend with enhanced styling
    legend = ax.legend(legend_elements, [el.get_label() for el in legend_elements],
                      title='Activation Statistics', loc='best', 
                      fontsize=12, title_fontsize=13, framealpha=0.95,
                      fancybox=True, shadow=True, borderpad=1,
                      columnspacing=1.2, handletextpad=0.8)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('#333333')
    
    # Style legend frame
    frame = legend.get_frame()
    frame.set_facecolor('#ffffff')
    frame.set_edgecolor('#cccccc')
    frame.set_linewidth(1.2)
    
    # Add subtle background color with gradient-like effect
    ax.set_facecolor('#fafafa')
    
    # Add annotation box with summary statistics
    if max_blocks > 0:
        max_top1 = max(top_stats['top1'])
        max_median = max(top_stats['median'])
        textstr = f'Peak Top-1: {max_top1:.3f}\nPeak Median: {max_median:.3f}\nBlocks: {max_blocks}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Tight layout with enhanced padding
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    output_path = PLOT_DIR / f"{safe_label}_top.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', 
               metadata={'Title': f'{label} Top Activations Analysis'})
    plt.close()
    
    print(f"  [OK] Top activations plot saved to: {output_path}")

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
def get_blocks_for_analysis(tag: str, model: nn.Module):
    """
    Extract analyzable blocks from different model types.
    Returns list of modules to analyze.
    """
    if tag.startswith("resnet") or tag.startswith("plain"):
        # ResNet/Plain: get individual BasicBlocks/PlainBlocks
        blocks = []
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for b in layer:
                blocks.append(b)
        return blocks
    elif tag.startswith("densenet"):
        # DenseNet: get DenseBlocks and individual DenseLayers
        blocks = []
        for name, module in model.features.named_children():
            if name.startswith('denseblock'):
                # Add the entire dense block
                blocks.append(module)
                # Also add individual dense layers for detailed analysis
                for layer in module.layers:
                    blocks.append(layer)
            elif name.startswith('transition'):
                # Add transition layers
                blocks.append(module)
        return blocks
    else:
        raise ValueError(f"Unknown model type: {tag}")

def get_layers_for_analysis(tag: str, model: nn.Module):
    """
    Extract individual layers (not blocks) from different model types for fine-grained analysis.
    This provides more detailed layer-by-layer activation analysis.
    
    Returns:
        List of tuples (layer, layer_type_string) for annotation purposes
    """
    layers = []
    
    if tag.startswith("resnet") or tag.startswith("plain"):
        # For ResNet/Plain: extract all individual layers including conv, bn, relu
        def extract_resnet_layers(layer_group):
            layer_list = []
            for block in layer_group:
                # Each BasicBlock/PlainBlock typically has: conv1, bn1, act, conv2, bn2, (optional downsample)
                for name, module in block.named_children():
                    if isinstance(module, nn.Conv2d):
                        layer_list.append((module, "Conv2d"))
                    elif isinstance(module, nn.BatchNorm2d):
                        layer_list.append((module, "BatchNorm2d"))
                    elif isinstance(module, nn.ReLU):
                        layer_list.append((module, "ReLU"))
                    elif isinstance(module, nn.Sequential):  # downsample
                        for sub_module in module:
                            if isinstance(sub_module, nn.Conv2d):
                                layer_list.append((sub_module, "Conv2d"))
                            elif isinstance(sub_module, nn.BatchNorm2d):
                                layer_list.append((sub_module, "BatchNorm2d"))
            return layer_list
        
        # Add initial layers
        if hasattr(model, 'conv1'):
            layers.append((model.conv1, "Conv2d"))
        if hasattr(model, 'bn1'):
            layers.append((model.bn1, "BatchNorm2d"))
        if hasattr(model, 'act'):
            layers.append((model.act, "ReLU"))
        if hasattr(model, 'pool'):  # Add missing maxpool layer
            layers.append((model.pool, "MaxPool2d"))
            
        # Add layers from each layer group
        layers.extend(extract_resnet_layers(model.layer1))
        layers.extend(extract_resnet_layers(model.layer2))
        layers.extend(extract_resnet_layers(model.layer3))
        layers.extend(extract_resnet_layers(model.layer4))
        
        # Add final layers
        if hasattr(model, 'avgpool'):
            layers.append((model.avgpool, "AdaptiveAvgPool2d"))
        if hasattr(model, 'fc'):
            layers.append((model.fc, "Linear"))
            
    elif tag.startswith("densenet"):
        # For DenseNet: extract individual layers from features
        def extract_densenet_layers(module, layers_list):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    layers_list.append((child, "Conv2d"))
                elif isinstance(child, nn.BatchNorm2d):
                    layers_list.append((child, "BatchNorm2d"))
                elif isinstance(child, nn.ReLU):
                    layers_list.append((child, "ReLU"))
                elif isinstance(child, nn.AvgPool2d):
                    layers_list.append((child, "AvgPool2d"))
                elif hasattr(child, 'named_children'):  # Recurse into submodules
                    extract_densenet_layers(child, layers_list)
        
        extract_densenet_layers(model.features, layers)
        
        # Add missing final classifier layer
        if hasattr(model, 'classifier'):
            layers.append((model.classifier, "Linear"))
        
    # Filter out any None values and ensure we have actual parameters
    layers = [(layer, layer_type) for layer, layer_type in layers if layer is not None and 
              (hasattr(layer, 'weight') or isinstance(layer, (nn.ReLU, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)))]
    
    return layers

@torch.no_grad()
def analyse(tag: str, model: nn.Module, results: Dict[str, Dict[str, List[float]]], use_layers: bool = False):
    """
    1. Collect per-block/layer activation L2 stats
    2. Activation box-plot
    3. Parameter L2 stats
    4. Generate top activations plot
    5. Optional layer-wise analysis with annotations
    """
    model.eval().to(DEVICE)
    
    # Choose between block or layer analysis
    if use_layers:
        analysis_units = get_layers_for_analysis(tag, model)
        analysis_type = "layers"
        print(f"  [INFO] Found {len(analysis_units)} individual layers for {tag}")
    else:
        analysis_units = get_blocks_for_analysis(tag, model)
        analysis_type = "blocks"
        print(f"  [INFO] Found {len(analysis_units)} blocks for {tag}")

    # Get a single random image
    if DATASET == "cifar100":
        print("Using CIFAR-100 test set")
        n_mean, n_std = [0.5071,0.4865,0.4409], [0.2673,0.2564,0.2762]
        val_transform = T.Compose([
            T.Resize(IMG_RES),
            T.ToTensor(),
            T.Normalize(n_mean, n_std),
        ])
        test_ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_transform)
        random_idx = torch.randint(0, len(test_ds), (1,)).item()
        random_image, _ = test_ds[random_idx]
        dummy = random_image.unsqueeze(0).to(DEVICE)
    else:
        print("Using dummy (zero image) test set")
        dummy = torch.zeros((1,3,IMG_RES,IMG_RES), dtype=torch.float32)
        dummy[:,0,:,:] = 0.485 / 0.229
        dummy[:,1,:,:] = 0.456 / 0.224
        dummy[:,2,:,:] = 0.406 / 0.225
        dummy = dummy.to(DEVICE)

    aL2m, aL2s = [], []
    pL2m, pL2s = [], []
    activations = []
    outliers = []
    layer_types = None  # Initialize layer_types to avoid scope issues

    # Extract actual modules and layer types
    if use_layers:
        modules = [layer for layer, layer_type in analysis_units]
        layer_types = [layer_type for layer, layer_type in analysis_units]
    else:
        modules = analysis_units

    # param L2 stats
    for unit in modules:
        param_list = [p.detach().flatten() for p in unit.parameters() if p.requires_grad]
        if param_list:  # Check if there are any parameters
            flat = torch.cat(param_list)
            unit_l2s = torch.tensor([p.detach().norm() for p in unit.parameters() if p.requires_grad])
            pL2m.append(unit_l2s.mean().item())
            pL2s.append(unit_l2s.std().item())
        else:
            # Layer has no parameters (e.g., ReLU, pooling layers)
            pL2m.append(0.0)
            pL2s.append(0.0)

    # activation hooking
    def hook_fn(_, __, out):
        outf = out.detach().float()
        
        # Handle different activation shapes from different layer types
        if outf.ndim == 4:  # Spatial features: [B,C,H,W]
            outf = outf.squeeze(0)  # Remove batch dimension -> [C,H,W]
            activations.append(outf.cpu().numpy())
            
            # L2 across channels
            fc = outf.flatten(1)  # [C, H*W]
            norms_c = fc.norm(dim=1, p=2)
            aL2m.append(norms_c.mean().item())
            aL2s.append(norms_c.std().item())
            
        elif outf.ndim == 2:  # Linear layer features: [B,Features]
            outf = outf.squeeze(0)  # Remove batch dimension -> [Features]
            activations.append(outf.cpu().numpy())
            
            # For 1D features, compute norm directly
            aL2m.append(outf.norm(dim=0, p=2).item())
            aL2s.append(0.0)  # No variation across "channels" for 1D output
            
        else:  # Fallback for other shapes
            # Remove batch dimension if present
            if outf.shape[0] == 1:
                outf = outf.squeeze(0)
            activations.append(outf.cpu().numpy())
            
            # Compute norm of flattened tensor
            flat = outf.flatten()
            aL2m.append(flat.norm(p=2).item())
            aL2s.append(0.0)

    hdls = []
    for unit in modules:
        hdls.append(unit.register_forward_hook(hook_fn))

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
    
    # Generate appropriate label
    label_suffix = f"_{analysis_type}" if use_layers else ""
    plot_label = f"{DATASET}_{tag}{label_suffix}"
    results[f"{tag}{label_suffix}"] = stats
    (PLOT_DIR / f"{plot_label}.json").write_text(json.dumps(stats))

    # Generate plots based on analysis type
    if use_layers:
        # Layer-wise plots with annotations
        _plot_box_layers(activations, outliers, plot_label, layer_types)
        top_stats = _compute_top_stats(activations)
        _plot_top_activations_layers(top_stats, tag, plot_label, layer_types)
    else:
        # Standard block plots
        _plot_box(activations, outliers, label=plot_label)
        top_stats = _compute_top_stats(activations)
        _plot_top_activations(top_stats, tag, plot_label)

    # line plots for each model
    def _plot_line(vals, stds, title_, ylabel_, fname_):
        x = range(1, len(vals)+1)
        plt.figure(figsize=(8,4), dpi=300)
        lb = [v - s for v, s in zip(vals, stds)]
        ub = [v + s for v, s in zip(vals, stds)]
        plt.fill_between(x, lb, ub, alpha=0.15, color="blue")
        plt.plot(x, vals, marker="o", color="blue", linewidth=1.2)
        plt.title(title_, weight="bold")
        plt.xlabel("Block" if not use_layers else "Layer")
        plt.ylabel(ylabel_)
        plt.grid(ls="--", lw=0.4)
        plt.tight_layout()
        outn = PLOT_DIR / f"{fname_}.png"
        plt.savefig(outn)
        plt.close()

    unit_type = "layer" if use_layers else "block"
    _plot_line(aL2m, aL2s, f"{tag} activation L2 ({unit_type}-wise)", "Activation L2",
               f"{plot_label}_act_l2")
    _plot_line(pL2m, pL2s, f"{tag} param L2 ({unit_type}-wise)", "Parameter L2",
               f"{plot_label}_param_l2")

def create_training_summary_table(all_metrics: Dict, curves_dir: Path):
    """
    Create a summary table of training results.
    """
    # Prepare data for summary
    summary_data = []
    
    for model_name, metrics in all_metrics.items():
        summary_data.append({
            'Model': model_name.upper(),
            'Best Val Acc (%)': f"{metrics['best_val_accuracy']*100:.2f}",
            'Best Val Acc Epoch': metrics['best_val_accuracy_epoch'],
            'Final Val Acc (%)': f"{metrics['final_val_accuracy']*100:.2f}",
            'Best Val Loss': f"{metrics['best_val_loss']:.4f}",
            'Best Val Loss Epoch': metrics['best_val_loss_epoch'],
            'Final Train Acc (%)': f"{metrics['final_train_accuracy']*100:.2f}",
        })
    
    # Sort by model type and depth for better presentation
    def sort_key(item):
        name = item['Model'].lower()
        if name.startswith('resnet'):
            return (0, int(name.replace('resnet', '')))
        elif name.startswith('plain'):
            return (1, int(name.replace('plain', '')))
        elif name.startswith('densenet'):
            return (2, int(name.replace('densenet', '')))
        return (3, 0)
    
    summary_data.sort(key=sort_key)
    
    # Create a nice table visualization
    fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.6 + 2), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    headers = list(summary_data[0].keys())
    table_data = [list(row.values()) for row in summary_data]
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code by model type
    for i, row_data in enumerate(summary_data):
        model_name = row_data['Model'].lower()
        if model_name.startswith('resnet'):
            color = '#E3F2FD'  # Light blue
        elif model_name.startswith('plain'):
            color = '#FCE4EC'  # Light magenta
        elif model_name.startswith('densenet'):
            color = '#FFF3E0'  # Light orange
        else:
            color = '#F5F5F5'  # Light gray
            
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    # Style headers
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#455A64')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Training Results Summary - {DATASET.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save table
    table_path = curves_dir / f"training_summary_table_{DATASET}.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[INFO] Training summary table saved to: {table_path}")
    
    # Also save as JSON for easy access
    summary_json_path = curves_dir / f"training_summary_{DATASET}.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"[INFO] Training summary JSON saved to: {summary_json_path}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print(f"TRAINING RESULTS SUMMARY - {DATASET.upper()}")
    print(f"{'='*80}")
    for row in summary_data:
        print(f"{row['Model']:12s} | Best Val Acc: {row['Best Val Acc (%)']}% | Final Val Acc: {row['Final Val Acc (%)']}%")
    print(f"{'='*80}") 
    
def create_comparative_training_plots():
    """
    Create comparative training plots across all trained models.
    """
    curves_dir = PLOT_DIR / "training_curves"
    if not curves_dir.exists():
        print("[WARN] No training curves directory found, skipping comparative plots")
        return
    
    # Load all available metrics
    all_metrics = {}
    for model_name in MODELS.keys():
        metrics_file = curves_dir / f"{DATASET}_{model_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics[model_name] = json.load(f)
    
    if not all_metrics:
        print("[WARN] No training metrics found, skipping comparative plots")
        return
    
    print(f"[INFO] Creating comparative training plots for {len(all_metrics)} models...")
    
    # Enhanced color scheme and styling for all model types
    colors = {
        'resnet': '#2E86AB',    # Professional blue
        'plain': '#A23B72',     # Professional magenta  
        'densenet': '#F18F01'   # Professional orange
    }
    
    line_styles = {
        14: '-',      # solid
        18: '--',     # dashed  
        34: '-.'      # dash-dot
    }
    
    markers = {
        14: 's',    # square
        18: 'o',    # circle
        34: '^'     # triangle
    }
    
    # Create comparative validation accuracy plot
    plt.figure(figsize=(14, 8), dpi=300)
    
    legend_elements = []
    
    for model_name, metrics in all_metrics.items():
        # Parse model type and depth
        if model_name.startswith('resnet'):
            model_type = 'resnet'
        elif model_name.startswith('plain'):
            model_type = 'plain'
        elif model_name.startswith('densenet'):
            model_type = 'densenet'
        else:
            continue
            
        depth = parse_depth_from_tag(model_name)
        
        color = colors[model_type]
        linestyle = line_styles.get(depth, '-')
        marker = markers.get(depth, 'o')
        
        epochs = metrics['epochs']
        val_accs = [acc * 100 for acc in metrics['val_accuracies']]
        
        line = plt.plot(epochs, val_accs, 
                       color=color, 
                       linestyle=linestyle, 
                       linewidth=2.5,
                       label=f'{model_type.upper()}-{depth}',
                       marker=marker,
                       markersize=5,
                       markerfacecolor=color,
                       markeredgecolor='white',
                       markeredgewidth=1,
                       alpha=0.9,
                       markevery=max(1, len(epochs)//20))  # Show markers every ~5% of epochs
        
        legend_elements.append(line[0])
    
    # Enhanced plot aesthetics
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title(f'Training Comparison: Validation Accuracy Across Architectures\n'
              f'Dataset: {DATASET.upper()}',
              fontsize=18, fontweight='bold', pad=25)
    
    # Customize grid and styling
    plt.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.8)
    plt.gca().set_axisbelow(True)
    plt.gca().set_facecolor('#fafafa')
    
    # Organized legend by model type
    resnet_elements = [el for el in legend_elements if 'RESNET' in el.get_label()]
    plain_elements = [el for el in legend_elements if 'PLAIN' in el.get_label()]
    densenet_elements = [el for el in legend_elements if 'DENSENET' in el.get_label()]
    
    # Sort by depth
    for elements in [resnet_elements, plain_elements, densenet_elements]:
        elements.sort(key=lambda x: int(x.get_label().split('-')[1]))
    
    # Create multiple legends
    legend_spacing = 0.33
    if resnet_elements:
        legend1 = plt.legend(resnet_elements, [el.get_label() for el in resnet_elements],
                           title='ResNet Models', loc='center right', 
                           bbox_to_anchor=(1.0, 0.8),
                           fontsize=11, title_fontsize=12, framealpha=0.9)
        legend1.get_title().set_fontweight('bold')
        plt.gca().add_artist(legend1)
    
    if plain_elements:
        legend2 = plt.legend(plain_elements, [el.get_label() for el in plain_elements],
                           title='Plain Models', loc='center right',
                           bbox_to_anchor=(1.0, 0.5),
                           fontsize=11, title_fontsize=12, framealpha=0.9)
        legend2.get_title().set_fontweight('bold')
        plt.gca().add_artist(legend2)
    
    if densenet_elements:
        legend3 = plt.legend(densenet_elements, [el.get_label() for el in densenet_elements],
                           title='DenseNet Models', loc='center right',
                           bbox_to_anchor=(1.0, 0.2),
                           fontsize=11, title_fontsize=12, framealpha=0.9)
        legend3.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save comparative plot
    comp_plot_path = curves_dir / f"comparative_validation_accuracy_{DATASET}.png"
    plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[INFO] Comparative validation accuracy plot saved to: {comp_plot_path}")
    
    # Create summary table
    create_training_summary_table(all_metrics, curves_dir)

def _plot_box_layers(activations: List[np.ndarray], outliers: List[List[Tuple[float,int,int]]], 
                     label: str, layer_types: List[str]):
    """
    Box plot variant for layer-wise analysis with layer type annotations.
    """
    if not activations:
        return

    # Flatten each layer's activations
    layer_vals = [act.flatten() for act in activations]

    plt.figure(figsize=(16, 4 + 0.3 * len(layer_vals)), dpi=300)
    bp = plt.boxplot(
        layer_vals,
        vert=True,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 1.5,
            "markerfacecolor": "r",
            "alpha": 0.5,
        },
    )
    
    # Color the boxes with a gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overplot specific outliers (reduced for clarity due to many layers)
    for i, outs in enumerate(outliers):
        if not outs or i % 3 != 0:  # Show every 3rd layer's outliers to avoid clutter
            continue
        xs = np.full(len(outs[:10]), i+1)  # Limit to top 10 outliers
        ys = [o[0] for o in outs[:10]]
        plt.scatter(xs, ys, color="red", marker="x", s=8, alpha=0.8)

    plt.xlabel("Layer Index", fontsize=12, fontweight='bold')
    plt.ylabel("Activation Values", fontsize=12, fontweight='bold')
    plt.title(f"{label} - Layer-wise Raw Activation Distribution", fontweight="bold", fontsize=14)
    plt.grid(True, ls="--", lw=0.4, axis="y", alpha=0.6)
    
    # Adjust x-axis for many layers
    if len(layer_vals) > 20:
        step = max(1, len(layer_vals) // 10)
        plt.xticks(range(1, len(layer_vals) + 1, step))
    
    # Add layer type annotations
    if layer_types:
        y_min, y_max = plt.ylim()
        annotation_y = y_min - (y_max - y_min) * 0.15  # Place annotations below plot
        
        # Create abbreviated layer type names for cleaner display
        abbreviations = {
            'Conv2d': 'C',
            'BatchNorm2d': 'BN', 
            'ReLU': 'R',
            'AvgPool2d': 'AP',
            'MaxPool2d': 'MP',
            'AdaptiveAvgPool2d': 'AAP',
            'Linear': 'L'
        }
        
        # Add annotations for every layer with tiny font
        for i, layer_type in enumerate(layer_types):
            abbrev = abbreviations.get(layer_type, layer_type[:2])  # Use first 2 chars if not in dict
            plt.text(i+1, annotation_y, abbrev, 
                    rotation=90, fontsize=4, ha='center', va='top',
                    color='#666666', alpha=0.7, weight='normal')
        
        # Adjust plot limits to accommodate annotations
        plt.ylim(annotation_y - (y_max - y_min) * 0.08, y_max)
    
    plt.tight_layout()
    out_png = PLOT_DIR / f"{label}_box_layers.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Layer-wise box plot saved to: {out_png}")

def _plot_top_activations_layers(top_stats, tag, label, layer_types):
    """
    Top activations plot variant for layer-wise analysis with annotations.
    """
    if not top_stats or not top_stats['top1']:
        return
        
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    
    x_vals = np.array(range(1, len(top_stats['top1']) + 1))
    
    # Professional color scheme optimized for layer-wise analysis
    colors = {
        'top1': '#C4261D',    # Bold red for highest activation
        'top2': '#F18F01',    # Vibrant orange for second highest
        'top3': '#F4B942',    # Golden yellow for third highest
        'median': '#2E86AB'   # Professional blue for median
    }
    
    line_styles = {
        'top1': '-',     # solid for top values
        'top2': '-',     # solid for top values
        'top3': '-',     # solid for top values 
        'median': '--'   # dashed to distinguish median
    }
    
    line_widths = {
        'top1': 2.5,     # Slightly thinner for many data points
        'top2': 2.2,
        'top3': 2.0,
        'median': 2.2
    }
    
    markers = {
        'top1': 'o',     # Circle for top1
        'top2': 's',     # Square for top2
        'top3': '^',     # Triangle for top3
        'median': 'D'    # Diamond for median
    }
    
    marker_sizes = {
        'top1': 5,  # Smaller markers for dense plots
        'top2': 4,
        'top3': 4,
        'median': 4
    }
    
    # Plot each statistic
    legend_elements = []
    for stat_name in ['top1', 'top2', 'top3', 'median']:
        values = np.array(top_stats[stat_name])
        
        if stat_name == 'median':
            display_label = 'Median'
        else:
            display_label = f'Top-{stat_name[-1]}'
        
        # Use fewer markers for dense plots
        markevery = max(1, len(x_vals) // 20) if len(x_vals) > 40 else 1
        
        line = ax.plot(x_vals, values,
                      color=colors[stat_name],
                      linestyle=line_styles[stat_name], 
                      linewidth=line_widths[stat_name],
                      label=display_label,
                      marker=markers[stat_name],
                      markersize=marker_sizes[stat_name],
                      markerfacecolor=colors[stat_name],
                      markeredgecolor='white',
                      markeredgewidth=1,
                      alpha=0.9,
                      markevery=markevery,
                      zorder=10 if stat_name == 'top1' else 5)
        legend_elements.append(line[0])
    
    # Enhanced aesthetics for layer-wise analysis
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Activation Value', fontsize=14, fontweight='bold', color='#333333') 
    ax.set_title(f'{tag} - Layer-wise Top Activation Analysis\n'
                f'Individual layer peak activations and median reference values',
                fontsize=16, fontweight='bold', pad=25, color='#333333')
    
    # Customize grid and ticks for many layers
    ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray', linewidth=0.5)
    ax.minorticks_on()
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color('#333333')
    
    ax.tick_params(axis='both', which='major', labelsize=11,
                  colors='#333333', width=1.3, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3, colors='#666666')
    
    # Adjust x-axis for many layers
    max_layers = len(top_stats['top1'])
    if max_layers > 25:
        step = max(1, max_layers // 15)
        ax.set_xticks(range(1, max_layers + 1, step))
    else:
        ax.set_xticks(range(1, max_layers + 1))
    
    # Professional legend
    legend = ax.legend(legend_elements, [el.get_label() for el in legend_elements],
                      title='Activation Statistics', loc='best', 
                      fontsize=11, title_fontsize=12, framealpha=0.95,
                      fancybox=True, shadow=True, borderpad=1)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('#333333')
    
    ax.set_facecolor('#fafafa')
    
    # Add annotation with layer-specific info
    if max_layers > 0:
        max_top1 = max(top_stats['top1'])
        max_median = max(top_stats['median'])
        textstr = f'Peak Top-1: {max_top1:.3f}\nPeak Median: {max_median:.3f}\nLayers: {max_layers}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Add layer type annotations
    if layer_types:
        y_min, y_max = ax.get_ylim()
        annotation_y = y_min - (y_max - y_min) * 0.12  # Place annotations below plot
        
        # Create abbreviated layer type names for cleaner display
        abbreviations = {
            'Conv2d': 'C',
            'BatchNorm2d': 'BN', 
            'ReLU': 'R',
            'AvgPool2d': 'AP',
            'MaxPool2d': 'MP',
            'AdaptiveAvgPool2d': 'AAP',
            'Linear': 'L'
        }
        
        # Add annotations for every layer with tiny font
        for i, layer_type in enumerate(layer_types):
            abbrev = abbreviations.get(layer_type, layer_type[:2])  # Use first 2 chars if not in dict
            ax.text(i+1, annotation_y, abbrev, 
                   rotation=90, fontsize=4, ha='center', va='top',
                   color='#666666', alpha=0.7, weight='normal')
        
        # Adjust plot limits to accommodate annotations
        ax.set_ylim(annotation_y - (y_max - y_min) * 0.05, y_max)
    
    plt.tight_layout(pad=3.0)
    
    output_path = PLOT_DIR / f"{label}_top_layers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  [OK] Layer-wise top activations plot saved to: {output_path}")

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    print("""
    ═══════════════════════════════════════════════════════════════════════
    DenseNet vs ResNet vs Plain Network Comparison
    ═══════════════════════════════════════════════════════════════════════
    
    DenseNet Model Details:
    ----------------------
    
    DenseNet-14: 
    - Block config: (3, 4, 4, 3) = 14 dense layers total
    - Growth rate: 64 (optimized for ~11M parameters)
    - Compression: 0.8 (reduced compression to keep more parameters)
    - Target: ~11M parameters (similar to ResNet-14)
    - Dense connections within each block
    
    DenseNet-18:
    - Block config: (4, 4, 4, 6) = 18 dense layers total
    - Growth rate: 64 (optimized for ~11M parameters)
    - Compression: 0.8 (reduced compression to keep more parameters)
    - Target: ~11M parameters (similar to ResNet-18)
    - More balanced layer distribution
    
    DenseNet-34:
    - Block config: (6, 8, 12, 8) = 34 dense layers total  
    - Growth rate: 64 (optimized for ~21M parameters)
    - Compression: 0.75 (balanced compression for parameter control)
    - Target: ~21M parameters (similar to ResNet-34)
    - Deeper dense blocks matching ResNet-34 complexity
    
    Key Architectural Assumptions:
    -----------------------------
    1. Stem Layer: All models use identical 7x7 conv + maxpool stem for fair comparison
    2. Feature Maps: DenseNet uses concatenation vs ResNet's addition
    3. Compression: 0.5 compression ratio in transition layers (DenseNet standard)
    4. Growth Rate: Tuned to achieve similar parameter counts across architectures
    5. Bottleneck: 4x growth_rate bottleneck width (DenseNet-BC design)
    6. Spatial Reduction: Transition layers use 2x2 avg pooling (vs ResNet's stride-2 conv)
    
    Training Configuration:
    ----------------------
    - Same learning schedules across all architectures
    - Same data augmentation and normalization
    - Same optimization (SGD + momentum + weight decay)
    - Mixed precision training for efficiency
    
    ═══════════════════════════════════════════════════════════════════════
    """)
    
    # Print model parameter counts for comparison
    print("Model Parameter Counts:")
    print("-" * 40)
    for name, model in MODELS.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name:12s}: {param_count:,} parameters")
    print("-" * 40)

    # Training phase
    train_loader, val_loader = None, None
    if not args.no_train:
        train_loader, val_loader = make_loaders()
        for model_name, model_obj in MODELS.items():
            train(model_name, model_obj, train_loader, val_loader)

    # Analysis phase
    if not args.no_analysis:
        if train_loader is None or val_loader is None:
            _ = make_loaders()

        all_stats = {}
        for model_name, model_obj in MODELS.items():
            ck_path = CHK_DIR / f"{DATASET}_{model_name}.pth"
            if not ck_path.exists():
                print(f"[WARN] Checkpoint not found for {model_name}, skipping analysis")
                continue
            state = torch.load(ck_path, map_location="cpu")
            model_obj.load_state_dict(strip_prefix(state))
            analyse(model_name, model_obj, all_stats)

        # Professional Combined L2 plot across all models
        if all_stats:
            fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
            
            # Enhanced color scheme for three model types
            colors = {
                'resnet': '#2E86AB',    # Professional blue
                'plain': '#A23B72',     # Professional magenta  
                'densenet': '#F18F01'   # Professional orange
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
            
            markers = {
                14: 's',    # square
                18: 'o',    # circle
                34: '^'     # triangle
            }
            
            # Group and plot models
            legend_elements = []
            
            for tag, stats in all_stats.items():
                # Parse model type and depth
                if tag.startswith('resnet'):
                    model_type = 'resnet'
                elif tag.startswith('plain'):
                    model_type = 'plain'
                elif tag.startswith('densenet'):
                    model_type = 'densenet'
                else:
                    continue
                    
                depth = parse_depth_from_tag(tag)
                
                x_vals = np.array(range(1, len(stats["l2_mean"]) + 1))
                means = np.array(stats["l2_mean"])
                stds = np.array(stats["l2_std"])
                
                color = colors[model_type]
                linestyle = line_styles[depth]
                linewidth = line_widths[depth]
                marker = markers[depth]
                alpha_fill = 0.15
                
                # Plot mean line
                line = ax.plot(x_vals, means, 
                              color=color, 
                              linestyle=linestyle, 
                              linewidth=linewidth,
                              label=f'{model_type.upper()}-{depth}',
                              marker=marker,
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
            
            # Enhanced plot aesthetics
            ax.set_xlabel('Block Index', fontsize=14, fontweight='bold')
            ax.set_ylabel('Activation L2 Norm', fontsize=14, fontweight='bold')
            ax.set_title(f'Per-Block Activation Analysis: ResNet vs Plain vs DenseNet\n'
                        f'Dataset: {DATASET.upper()} | Error bands show ±1 standard deviation',
                        fontsize=18, fontweight='bold', pad=25)
            
            # Customize grid and styling
            ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.8)
            ax.set_axisbelow(True)
            
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#333333')
            
            ax.tick_params(axis='both', which='major', labelsize=12, 
                          colors='#333333', width=1.2, length=6)
            ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Set x-axis to show integer ticks only
            max_blocks = max([len(s["l2_mean"]) for s in all_stats.values()])
            ax.set_xticks(range(1, min(max_blocks + 1, 25)))  # Limit for readability
            
            # Organized legend by model type
            resnet_elements = [el for el in legend_elements if 'RESNET' in el.get_label()]
            plain_elements = [el for el in legend_elements if 'PLAIN' in el.get_label()]
            densenet_elements = [el for el in legend_elements if 'DENSENET' in el.get_label()]
            
            # Sort by depth
            for elements in [resnet_elements, plain_elements, densenet_elements]:
                elements.sort(key=lambda x: int(x.get_label().split('-')[1]))
            
            # Create multiple legends
            if resnet_elements:
                legend1 = ax.legend(resnet_elements, [el.get_label() for el in resnet_elements],
                                   title='ResNet Models', loc='upper left', 
                                   fontsize=11, title_fontsize=12, framealpha=0.9)
                legend1.get_title().set_fontweight('bold')
                ax.add_artist(legend1)
            
            if plain_elements:
                legend2 = ax.legend(plain_elements, [el.get_label() for el in plain_elements],
                                   title='Plain Models', loc='upper center',
                                   fontsize=11, title_fontsize=12, framealpha=0.9)
                legend2.get_title().set_fontweight('bold')
                ax.add_artist(legend2)
            
            if densenet_elements:
                legend3 = ax.legend(densenet_elements, [el.get_label() for el in densenet_elements],
                                   title='DenseNet Models', loc='upper right',
                                   fontsize=11, title_fontsize=12, framealpha=0.9)
                legend3.get_title().set_fontweight('bold')
            
            # Final styling
            ax.set_facecolor('#fafafa')
            plt.tight_layout(pad=2.0)
            
            # Save high-quality plot
            output_path = PLOT_DIR / f"combined_l2_rvd_professional_{DATASET}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"[INFO] Professional combined plot saved to: {output_path}")

        # ═══════════════════════ LAYER-WISE ANALYSIS ═══════════════════════
        print(f"\n{'='*60}")
        print("STARTING LAYER-WISE ANALYSIS")
        print(f"{'='*60}")
        
        # Run layer-wise analysis for all models
        print(f"\n=== Layer-wise Analysis Mode ===")
        all_layer_stats: Dict[str, Dict[str, List[float]]] = {}
        
        for model_name, model_obj in MODELS.items():
            ck_path = CHK_DIR / f"{DATASET}_{model_name}.pth"
            if not ck_path.exists():
                print(f"[WARN] Checkpoint not found for {model_name}, skipping layer-wise analysis")
                continue
            try:
                print(f"\n[INFO] Running layer-wise analysis for {model_name}")
                state = torch.load(ck_path, map_location="cpu")
                model_obj.load_state_dict(strip_prefix(state))
                analyse(model_name, model_obj, all_layer_stats, use_layers=True)
            except Exception as e:
                print(f"  [WARN] Skipped layer-wise analysis for {model_name}: {e}")
        
        if all_layer_stats:
            print(f"\n[INFO] Successfully analyzed {len(all_layer_stats)} models at layer level")
            
            # Generate combined layer-wise comparison plots
            fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
            
            # Use the same color scheme as block analysis for consistency
            colors = {
                'resnet': '#2E86AB',    # Professional blue
                'plain': '#A23B72',     # Professional magenta  
                'densenet': '#F18F01'   # Professional orange
            }
            
            line_styles = {
                14: '-',      # solid
                18: '--',     # dashed  
                34: '-.'      # dash-dot
            }
            
            line_widths = {
                14: 1.8,  # Slightly thinner for dense layer plots
                18: 2.0,
                34: 2.2
            }
            
            markers = {
                14: 's',    # square
                18: 'o',    # circle
                34: '^'     # triangle
            }
            
            legend_elements = []
            max_layers_count = 0
            
            for tag, stats in all_layer_stats.items():
                # Skip if this is a layers analysis (contains "_layers")
                if "_layers" not in tag:
                    continue
                    
                # Parse model type and depth from tag (remove "_layers" suffix)
                base_tag = tag.replace("_layers", "")
                if base_tag.startswith('resnet'):
                    model_type = 'resnet'
                elif base_tag.startswith('plain'):
                    model_type = 'plain'
                elif base_tag.startswith('densenet'):
                    model_type = 'densenet'
                else:
                    continue
                    
                depth = parse_depth_from_tag(base_tag)
                
                means = np.array(stats["l2_mean"])
                stds = np.array(stats["l2_std"])
                x_vals = np.array(range(1, len(means) + 1))
                max_layers_count = max(max_layers_count, len(means))
                
                color = colors[model_type]
                linestyle = line_styles[depth]
                linewidth = line_widths[depth]
                marker = markers[depth]
                
                # Use fewer markers for dense plots
                markevery = max(1, len(x_vals) // 15) if len(x_vals) > 30 else 1
                
                # Plot with professional styling adapted for layer analysis
                line = ax.plot(x_vals, means, 
                              color=color, linestyle=linestyle, linewidth=linewidth,
                              label=f"{model_type.upper()}-{depth} ({len(means)} layers)",
                              marker=marker, markersize=4,
                              markerfacecolor=color, markeredgecolor='white',
                              markeredgewidth=0.8, alpha=0.9, markevery=markevery)
                
                # Add lighter error band
                ax.fill_between(x_vals, means - stds, means + stds, 
                               color=color, alpha=0.08, linewidth=0)
                
                legend_elements.append(line[0])
            
            # Enhanced styling for layer-wise comparison
            ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold', color='#333333')
            ax.set_ylabel('Activation L2 Norm', fontsize=14, fontweight='bold', color='#333333')
            ax.set_title(f'Layer-wise Activation Analysis: ResNet vs Plain vs DenseNet\n'
                        f'Dataset: {DATASET.upper()} | Individual layer analysis with error bands',
                        fontsize=18, fontweight='bold', pad=25, color='#333333')
            
            # Grid and aesthetics
            ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray', linewidth=0.5)
            ax.minorticks_on()
            
            # Spines and ticks
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#333333')
            
            ax.tick_params(axis='both', which='major', labelsize=12, 
                          colors='#333333', width=1.2, length=6)
            ax.tick_params(axis='both', which='minor', width=1, length=3, colors='#666666')
            
            # Adjust x-axis for potentially many layers
            if max_layers_count > 30:
                step = max(1, max_layers_count // 15)
                ax.set_xticks(range(1, max_layers_count + 1, step))
            else:
                ax.set_xticks(range(1, max_layers_count + 1, max(1, max_layers_count // 10)))
            
            # Organized legend by model type
            resnet_elements = [el for el in legend_elements if 'RESNET' in el.get_label()]
            plain_elements = [el for el in legend_elements if 'PLAIN' in el.get_label()]
            densenet_elements = [el for el in legend_elements if 'DENSENET' in el.get_label()]
            
            # Sort by depth
            for elements in [resnet_elements, plain_elements, densenet_elements]:
                elements.sort(key=lambda x: int(x.get_label().split('-')[1].split(' ')[0]))
            
            # Create multiple legends for layer-wise analysis
            legend_y_positions = [0.85, 0.65, 0.45]
            legends = []
            
            if resnet_elements:
                legend1 = ax.legend(resnet_elements, [el.get_label() for el in resnet_elements],
                                   title='ResNet Models (Layer-wise)', loc='center right',
                                   bbox_to_anchor=(1.0, legend_y_positions[0]),
                                   fontsize=10, title_fontsize=11, framealpha=0.95)
                legend1.get_title().set_fontweight('bold')
                legend1.get_title().set_color('#333333')
                ax.add_artist(legend1)
                legends.append(legend1)
            
            if plain_elements:
                legend2 = ax.legend(plain_elements, [el.get_label() for el in plain_elements],
                                   title='Plain Models (Layer-wise)', loc='center right',
                                   bbox_to_anchor=(1.0, legend_y_positions[1]),
                                   fontsize=10, title_fontsize=11, framealpha=0.95)
                legend2.get_title().set_fontweight('bold')
                legend2.get_title().set_color('#333333')
                ax.add_artist(legend2)
                legends.append(legend2)
            
            if densenet_elements:
                legend3 = ax.legend(densenet_elements, [el.get_label() for el in densenet_elements],
                                   title='DenseNet Models (Layer-wise)', loc='center right',
                                   bbox_to_anchor=(1.0, legend_y_positions[2]),
                                   fontsize=10, title_fontsize=11, framealpha=0.95)
                legend3.get_title().set_fontweight('bold')
                legend3.get_title().set_color('#333333')
                legends.append(legend3)
            
            ax.set_facecolor('#fafafa')
            
            # Add summary annotation
            textstr = f'Max Layers: {max_layers_count}\nArchitectures: {len([tag for tag in all_layer_stats.keys() if "_layers" in tag])}'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, fontfamily='monospace')
            
            plt.tight_layout(pad=3.0)
            
            # Save layer-wise comparison plot
            output_path = PLOT_DIR / f"combined_l2_rvd_layers_{DATASET}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print(f"  [OK] Layer-wise comparison plot saved to: {output_path}")
        
        else:
            print("[WARN] No successful layer-wise analyses to combine")

    print("✓ Training and analysis complete!")
    print(f"✓ Checkpoints saved in: {CHK_DIR}")
    print(f"✓ Visualizations saved in: {PLOT_DIR}")
    
    # Create comparative training plots if training was performed
    if not args.no_train:
        create_comparative_training_plots()

    # ═══════════════════════ ACTIVATION ANALYSIS EXPLANATION ═══════════════════════
    print(f"\n{'='*80}")
    print("ACTIVATION ANALYSIS METHODOLOGY")
    print(f"{'='*80}")
    print("""
ACTIVATION PARSING AND ANALYSIS METHODOLOGY:

This script implements two complementary levels of activation analysis to provide both
macro-level architectural insights and micro-level layer-specific understanding:

1. BLOCK-LEVEL ANALYSIS (Standard "_top" plots):
   • Analyzes activations at the architectural block level (e.g., ResNet BasicBlocks, 
     DenseNet DenseBlocks, Plain CNN blocks)
   • Each data point represents aggregated statistics from an entire functional block
   • Captures high-level architectural patterns and skip connection effects
   • Provides coarse-grained comparison between different architectural paradigms
   • Ideal for understanding overall network behavior and architectural trade-offs

2. LAYER-WISE ANALYSIS ("_layers" plots with type annotations):
   • Analyzes activations at individual layer granularity (Conv2d, BatchNorm2d, ReLU, etc.)
   • Each data point represents statistics from a single atomic operation
   • Reveals fine-grained activation dynamics within architectural blocks
   • Shows heterogeneous behavior across different layer types within the same block
   • Layer type annotations use abbreviations: C=Conv2d, BN=BatchNorm2d, R=ReLU, 
     AP=AvgPool2d, MP=MaxPool2d for every layer position
   • Enables detailed analysis of normalization effects, activation functions, and 
     feature transformation patterns

ACTIVATION COMPUTATION METHODOLOGY:
• Forward hooks capture intermediate feature maps at each analysis unit
• Raw activations are processed to compute L2 norms across spatial dimensions
• Top-k statistics (top1, top2, top3) identify peak activation magnitudes
• Median statistics provide robust central tendency measures
• Both per-channel and global statistics are computed for comprehensive analysis

STATISTICAL SIGNIFICANCE:
• Error bands represent ±1 standard deviation across feature maps
• Outlier detection identifies extreme activation values for anomaly analysis
• Cross-architectural comparisons use identical input stimuli for fair evaluation
• Layer-wise analysis reveals activation heterogeneity masked in block-level aggregation

This dual-resolution approach enables both architectural-level insights (ResNet vs 
DenseNet vs Plain CNN behavior) and implementation-level insights (layer-specific 
activation patterns, normalization effects, and activation function dynamics).
    """)
    print(f"{'='*80}")

