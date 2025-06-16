#!/usr/bin/env python3
# resnet_vs_densenet_layerwise_comparison.py
#
# Purpose:
#   Create stunning layer-wise raw activation comparison plots between ResNet and DenseNet architectures.
#   This script builds off train_and_analyze_resnet18_vs_plain18_v_dense.py to generate beautiful
#   professional visualizations comparing activation patterns across individual layers.
#
# Features:
#   - Layer-wise raw activation analysis for ResNet and DenseNet models
#   - Beautiful combined plots with professional styling
#   - Gradient color schemes and smooth line rendering
#   - Statistical annotations and performance metrics
#   - High-resolution publication-ready outputs

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
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
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

# Set beautiful style
plt.style.use('default')
sns.set_palette("husl")

# ───────────────────────── CONFIG ─────────────────────────
BASE = Path.cwd()
CHK_DIR = BASE / "checkpoints"
DATA_DIR = BASE / "datasets"

# Create beautiful output directory
PLOT_DIR = BASE / "viz" / "plots" / "dense_res_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "A CUDA-capable GPU is required."

# Dataset configuration
IMAGENET_DIR = os.getenv("IMAGENET_DIR")
if IMAGENET_DIR and Path(IMAGENET_DIR).is_dir():
    DATASET = "imagenet"
else:
    DATASET = "cifar100"

IMG_RES = 224
NUM_CLASSES = 1000 if DATASET in {"imagenet", "tiny"} else 100
print(f"[INFO] Dataset: {DATASET.upper()}, Classes: {NUM_CLASSES}")

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

# ────────────── DENSENET IMPLEMENTATION ──────────────
class _DenseLayer(nn.Module):
    """DenseNet Layer: BatchNorm + ReLU + Conv1x1 + BatchNorm + ReLU + Conv3x3"""
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
    """Dense Block: Contains multiple DenseLayers with feature concatenation."""
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
    """Transition layer: BatchNorm + ReLU + Conv1x1 + AvgPool2d"""
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
    """General ResNet with BasicBlock."""
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
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
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
    """Custom DenseNet implementation comparable to ResNet architectures."""
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

def densenet14(num_classes=NUM_CLASSES):
    return DenseNet(block_config=(3, 4, 4, 3), growth_rate=64, compression=0.8, num_classes=num_classes)

def densenet18(num_classes=NUM_CLASSES):
    return DenseNet(block_config=(4, 4, 4, 6), growth_rate=64, compression=0.8, num_classes=num_classes)

def densenet34(num_classes=NUM_CLASSES):
    return DenseNet(block_config=(6, 8, 12, 8), growth_rate=64, compression=0.75, num_classes=num_classes)

# Model configurations
MODELS = {
    "resnet14": resnet14(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "densenet14": densenet14(),
    "densenet18": densenet18(),
    "densenet34": densenet34(),
}

# ───────── LAYER EXTRACTION ─────────
def get_layers_for_analysis(tag: str, model: nn.Module):
    """Extract individual layers for fine-grained analysis."""
    layers = []
    
    if tag.startswith("resnet"):
        # For ResNet: extract all individual layers including conv, bn, relu
        def extract_resnet_layers(layer_group):
            layer_list = []
            for block in layer_group:
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
        if hasattr(model, 'pool'):
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

# ───────── ANALYSIS FUNCTION ─────────
@torch.no_grad()
def analyze_model_layerwise(tag: str, model: nn.Module):
    """Analyze layer-wise raw activations for a single model."""
    model.eval().to(DEVICE)
    
    # Get layers for analysis
    analysis_units = get_layers_for_analysis(tag, model)
    print(f"  [INFO] Analyzing {len(analysis_units)} layers for {tag}")
    
    # Get test image
    if DATASET == "cifar100":
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
        dummy = torch.zeros((1,3,IMG_RES,IMG_RES), dtype=torch.float32)
        dummy[:,0,:,:] = 0.485 / 0.229
        dummy[:,1,:,:] = 0.456 / 0.224
        dummy[:,2,:,:] = 0.406 / 0.225
        dummy = dummy.to(DEVICE)

    activations = []
    raw_means = []
    
    # Extract actual modules
    modules = [layer for layer, layer_type in analysis_units]
    
    # Activation hooking
    def hook_fn(_, __, out):
        outf = out.detach().float()
        
        # Handle different activation shapes from different layer types
        if outf.ndim == 4:  # Spatial features: [B,C,H,W]
            outf = outf.squeeze(0)  # Remove batch dimension -> [C,H,W]
            activations.append(outf.cpu().numpy())
            
            # Raw mean across all spatial locations and channels
            raw_means.append(outf.mean().item())
            
        elif outf.ndim == 2:  # Linear layer features: [B,Features]
            outf = outf.squeeze(0)  # Remove batch dimension -> [Features]
            activations.append(outf.cpu().numpy())
            
            # Raw mean for 1D features
            raw_means.append(outf.mean().item())
            
        else:  # Fallback for other shapes
            if outf.shape[0] == 1:
                outf = outf.squeeze(0)
            activations.append(outf.cpu().numpy())
            
            # Raw mean of flattened tensor
            raw_means.append(outf.mean().item())

    # Register hooks
    hdls = [unit.register_forward_hook(hook_fn) for unit in modules]
    
    # Forward pass
    _ = model(dummy)
    
    # Clean up hooks
    for h in hdls:
        h.remove()
    
    return raw_means

# ───────── BEAUTIFUL PLOTTING ─────────
def create_stunning_comparison_plot():
    """Create a beautiful ResNet vs DenseNet layer-wise comparison plot."""
    print("\n" + "="*80)
    print("CREATING STUNNING RESNET VS DENSENET COMPARISON")
    print("="*80)
    
    # Analyze all models
    all_results = {}
    
    for model_name, model_obj in MODELS.items():
        # Skip if not ResNet or DenseNet
        if not (model_name.startswith("resnet") or model_name.startswith("densenet")):
            continue
            
        ck_path = CHK_DIR / f"{DATASET}_{model_name}.pth"
        if not ck_path.exists():
            print(f"[WARN] Checkpoint not found for {model_name}, skipping")
            continue
            
        print(f"\n[INFO] Loading and analyzing {model_name}")
        state = torch.load(ck_path, map_location="cpu")
        model_obj.load_state_dict(state)
        
        raw_means = analyze_model_layerwise(model_name, model_obj)
        all_results[model_name] = raw_means
        
        # Cleanup
        del model_obj
        torch.cuda.empty_cache()
        gc.collect()
    
    if not all_results:
        print("[ERROR] No models to analyze!")
        return
    
    # Create the stunning plot
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    
    # Define beautiful color schemes with gradients
    resnet_colors = {
        'resnet14': '#FF6B6B',    # Coral red
        'resnet18': '#4ECDC4',    # Turquoise
        'resnet34': '#45B7D1',    # Sky blue
    }
    
    densenet_colors = {
        'densenet14': '#96CEB4',  # Mint green
        'densenet18': '#FFEAA7',  # Warm yellow
        'densenet34': '#DDA0DD',  # Plum
    }
    
    # Define elegant line styles
    line_styles = {
        14: '-',      # solid
        18: '--',     # dashed
        34: '-.',     # dash-dot
    }
    
    # Plot settings for stunning visuals
    line_width = 3.5
    marker_size = 8
    alpha = 0.85
    
    # Create gradient background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.1, 
              extent=[0, max([len(vals) for vals in all_results.values()]), 
                     min([min(vals) for vals in all_results.values()]), 
                     max([max(vals) for vals in all_results.values()])])
    
    legend_elements = []
    
    # Plot ResNet models with stunning effects
    for model_name, raw_means in all_results.items():
        if model_name.startswith('resnet'):
            depth = int(model_name.replace('resnet', ''))
            x_vals = np.arange(1, len(raw_means) + 1)
            
            color = resnet_colors[model_name]
            linestyle = line_styles[depth]
            
            # Create smooth interpolation for elegance
            if len(x_vals) > 3:
                from scipy import interpolate
                f = interpolate.interp1d(x_vals, raw_means, kind='cubic')
                x_smooth = np.linspace(x_vals[0], x_vals[-1], len(x_vals) * 3)
                y_smooth = f(x_smooth)
            else:
                x_smooth, y_smooth = x_vals, raw_means
            
            # Main line with shadow effect
            line = ax.plot(x_smooth, y_smooth,
                          color=color, linestyle=linestyle, linewidth=line_width,
                          label=f'ResNet-{depth} ({len(raw_means)} layers)',
                          alpha=alpha, zorder=10)
            
            # Add subtle shadow
            ax.plot(x_smooth, y_smooth,
                   color='black', linestyle=linestyle, linewidth=line_width + 1,
                   alpha=0.1, zorder=5)
            
            # Add marker highlights at key points
            highlight_indices = np.linspace(0, len(x_vals)-1, min(10, len(x_vals))).astype(int)
            ax.scatter(x_vals[highlight_indices], np.array(raw_means)[highlight_indices],
                      color=color, s=marker_size*15, alpha=0.8, 
                      edgecolor='white', linewidth=2, zorder=15)
            
            legend_elements.append(line[0])
    
    # Plot DenseNet models with stunning effects
    for model_name, raw_means in all_results.items():
        if model_name.startswith('densenet'):
            depth = int(model_name.replace('densenet', ''))
            x_vals = np.arange(1, len(raw_means) + 1)
            
            color = densenet_colors[model_name]
            linestyle = line_styles[depth]
            
            # Create smooth interpolation for elegance
            if len(x_vals) > 3:
                from scipy import interpolate
                f = interpolate.interp1d(x_vals, raw_means, kind='cubic')
                x_smooth = np.linspace(x_vals[0], x_vals[-1], len(x_vals) * 3)
                y_smooth = f(x_smooth)
            else:
                x_smooth, y_smooth = x_vals, raw_means
            
            # Main line with glow effect
            line = ax.plot(x_smooth, y_smooth,
                          color=color, linestyle=linestyle, linewidth=line_width,
                          label=f'DenseNet-{depth} ({len(raw_means)} layers)',
                          alpha=alpha, zorder=10)
            
            # Add glow effect
            for width in [line_width+4, line_width+2]:
                ax.plot(x_smooth, y_smooth,
                       color=color, linestyle=linestyle, linewidth=width,
                       alpha=0.1, zorder=5)
            
            # Add diamond markers at key points
            highlight_indices = np.linspace(0, len(x_vals)-1, min(10, len(x_vals))).astype(int)
            ax.scatter(x_vals[highlight_indices], np.array(raw_means)[highlight_indices],
                      color=color, s=marker_size*15, alpha=0.8, marker='D',
                      edgecolor='white', linewidth=2, zorder=15)
            
            legend_elements.append(line[0])
    
    # Stunning axis styling
    ax.set_xlabel('Layer Index', fontsize=18, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Raw Activation Mean', fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Create beautiful title with gradient effect
    title_text = f'ResNet vs DenseNet: Layer-wise Raw Activation Analysis\n' \
                f'Dataset: {DATASET.upper()} | Architectural Comparison of Deep Learning Models'
    ax.text(0.5, 1.08, title_text, transform=ax.transAxes, 
           fontsize=22, fontweight='bold', ha='center', va='center',
           color='#2C3E50', bbox=dict(boxstyle="round,pad=0.5", 
           facecolor='white', edgecolor='#BDC3C7', alpha=0.9))
    
    # Beautiful grid with subtle transparency
    ax.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D', linewidth=1)
    ax.set_axisbelow(True)
    
    # Add subtle minor grid
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#95A5A6', linewidth=0.5)
    ax.minorticks_on()
    
    # Elegant spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#34495E')
    
    # Beautiful ticks
    ax.tick_params(axis='both', which='major', labelsize=14, 
                  colors='#2C3E50', width=2, length=8)
    ax.tick_params(axis='both', which='minor', width=1, length=4, colors='#7F8C8D')
    
    # Stunning legend with multiple columns
    resnet_legend = [el for el in legend_elements if 'ResNet' in el.get_label()]
    densenet_legend = [el for el in legend_elements if 'DenseNet' in el.get_label()]
    
    # Create beautiful dual legends
    legend1 = ax.legend(resnet_legend, [el.get_label() for el in resnet_legend],
                       title='ResNet Architectures', loc='upper left', 
                       fontsize=13, title_fontsize=15, framealpha=0.95,
                       fancybox=True, shadow=True, borderpad=1.5)
    legend1.get_frame().set_facecolor('#FADBD8')
    legend1.get_frame().set_edgecolor('#E74C3C')
    legend1.get_frame().set_linewidth(2)
    legend1.get_title().set_fontweight('bold')
    legend1.get_title().set_color('#E74C3C')
    
    legend2 = ax.legend(densenet_legend, [el.get_label() for el in densenet_legend],
                       title='DenseNet Architectures', loc='upper right',
                       fontsize=13, title_fontsize=15, framealpha=0.95,
                       fancybox=True, shadow=True, borderpad=1.5)
    legend2.get_frame().set_facecolor('#D5F4E6')
    legend2.get_frame().set_edgecolor('#27AE60')
    legend2.get_frame().set_linewidth(2)
    legend2.get_title().set_fontweight('bold')
    legend2.get_title().set_color('#27AE60')
    
    ax.add_artist(legend1)
    
    # Beautiful background with subtle texture
    ax.set_facecolor('#FDFEFE')
    
    # Add stunning annotation box with statistics
    max_layers = max([len(vals) for vals in all_results.values()])
    total_models = len(all_results)
    
    stats_text = f'Analysis Summary:\n' \
                f'• Total Models: {total_models}\n' \
                f'• Max Layers: {max_layers}\n' \
                f'• Dataset: {DATASET.upper()}\n' \
                f'• Architecture Types: 2'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=11, ha='left', va='bottom',
           bbox=dict(boxstyle="round,pad=0.7", facecolor='white', 
                    edgecolor='#3498DB', alpha=0.9, linewidth=2),
           fontfamily='monospace', color='#2C3E50')
    
    # Adjust layout with extra padding for beauty
    plt.tight_layout(pad=4.0)
    
    # Save with multiple formats for maximum quality
    output_base = PLOT_DIR / f"stunning_resnet_vs_densenet_layerwise_{DATASET}"
    
    # Ultra high-quality PNG
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', format='png',
               metadata={'Title': 'ResNet vs DenseNet Layer-wise Analysis',
                        'Description': 'Beautiful activation comparison'})
    
    # Vector PDF for publications
    plt.savefig(f"{output_base}.pdf", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', format='pdf')
    
    # SVG for web use
    plt.savefig(f"{output_base}.svg", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', format='svg')
    
    plt.close()
    
    print(f"\n✨ STUNNING PLOTS CREATED:")
    print(f"  📊 PNG: {output_base}.png")
    print(f"  📄 PDF: {output_base}.pdf") 
    print(f"  🌐 SVG: {output_base}.svg")
    print(f"  📁 Directory: {PLOT_DIR}")
    
    # Create summary statistics
    create_model_statistics(all_results)

def create_model_statistics(all_results):
    """Create beautiful statistical summary."""
    print(f"\n{'='*80}")
    print("BEAUTIFUL STATISTICAL SUMMARY")
    print(f"{'='*80}")
    
    stats_data = []
    for model_name, raw_means in all_results.items():
        architecture = "ResNet" if model_name.startswith("resnet") else "DenseNet"
        depth = int(model_name.replace('resnet', '').replace('densenet', ''))
        
        stats_data.append({
            'Model': model_name.upper(),
            'Architecture': architecture,
            'Depth': depth,
            'Layers': len(raw_means),
            'Min Activation': f"{min(raw_means):.4f}",
            'Max Activation': f"{max(raw_means):.4f}",
            'Mean Activation': f"{np.mean(raw_means):.4f}",
            'Std Activation': f"{np.std(raw_means):.4f}",
        })
    
    # Sort by architecture and depth
    stats_data.sort(key=lambda x: (x['Architecture'], x['Depth']))
    
    # Beautiful console output
    print(f"\n{'Model':<12} {'Arch':<8} {'Layers':<7} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 70)
    for stat in stats_data:
        print(f"{stat['Model']:<12} {stat['Architecture']:<8} {stat['Layers']:<7} "
              f"{stat['Min Activation']:<8} {stat['Max Activation']:<8} {stat['Mean Activation']:<8} {stat['Std Activation']:<8}")
    
    # Save statistics as JSON
    stats_path = PLOT_DIR / f"model_statistics_{DATASET}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"\n📊 Statistics saved to: {stats_path}")

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    STUNNING RESNET VS DENSENET COMPARISON            ║
    ║                           Layer-wise Analysis                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    🎨 Creating beautiful layer-wise activation comparison plots
    🔬 Analyzing ResNet and DenseNet architectures 
    📊 Generating publication-ready visualizations
    ✨ Professional styling with gradient effects
    
    Dataset: {DATASET.upper()}
    Models: ResNet-14/18/34, DenseNet-14/18/34
    Output: {PLOT_DIR}
    """)
    
    try:
        create_stunning_comparison_plot()
        print(f"\n🎉 SUCCESS! Beautiful comparison plots created!")
        print(f"🔍 Check {PLOT_DIR} for your stunning visualizations")
        
    except Exception as e:
        print(f"\n❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc() 