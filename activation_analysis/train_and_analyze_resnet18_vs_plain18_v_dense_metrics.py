#!/usr/bin/env python3
# train_and_analyze_resnet18_vs_plain18_v_dense_metrics.py
#
# Purpose:
#   Modified version that focuses on Top 1%, Top 5%, and Top 10% activation metrics
#   analysis across multiple test images. Utilizes existing trained checkpoints.
#   CRITICAL: Analyzes INPUT activations to architectural layers (properly includes residuals/concatenations).
#
# Key Features:
#   - ResNet: Hooks at BasicBlock level to capture inputs AFTER residual additions
#   - PlainNet: Hooks at PlainBlock level (no residuals, but consistent with ResNet structure)
#   - DenseNet: Hooks at DenseBlock/Transition level to capture inputs AFTER concatenations
#
# Usage example:
#   python train_and_analyze_resnet18_vs_plain18_v_dense_metrics.py --no_train
#
# Key changes:
#   - Uses multiple test images from test_images/ directory
#   - Computes Top 1%, Top 5%, and Top 10% percentile activation statistics
#   - Generates beautiful plots showing mean ± std of these metrics
#   - Saves to separate _metrics directory to avoid conflicts

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
CHK_DIR = BASE / "checkpoints"  # Use existing checkpoints
CHK_DIR.mkdir(exist_ok=True)

CURVE_DIR = BASE / "viz_metrics"  # Modified output directory
CURVE_DIR.mkdir(exist_ok=True)

PLOT_DIR = BASE / "viz" / "plots" / "act_analysis_rvd_metrics"  # ResNet vs Dense Metrics
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE / "datasets"
DATA_DIR.mkdir(exist_ok=True)

TEST_IMAGES_DIR = BASE / "test_images"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "A CUDA-capable GPU is required."

torch.backends.cudnn.benchmark = True

# ───────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["auto", "imagenet", "cifar100", "tiny"], default="auto")
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--subset", type=float, default=1.0, help="subset ratio for ImageNet training data")
parser.add_argument("--no_train", action="store_true", default=True, help="skip training (default for metrics)")
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
# Copy all the architecture definitions from the original file...
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

# [Include DenseNet classes here - _DenseLayer, _DenseBlock, _Transition]
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            concat_features = x
        else:
            concat_features = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(concat_features)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class _DenseBlock(nn.Module):
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

# [Include ResNet and DenseNet classes...]
class ResNet(nn.Module):
    def __init__(self, block, layers: List[int], num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, block_config: tuple, growth_rate: int = 12, 
                 num_init_features: int = 64, bn_size: int = 4, 
                 compression: float = 0.5, drop_rate: float = 0.0, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                   num_output_features=int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# Model factory functions
def resnet14(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes)

def resnet18(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def plain14(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [1, 1, 1, 1], num_classes)

def plain18(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [2, 2, 2, 2], num_classes)

def plain34(num_classes=NUM_CLASSES):
    return ResNet(PlainBlock, [3, 4, 6, 3], num_classes)

def densenet14(num_classes=NUM_CLASSES):
    return DenseNet((2, 2, 2, 2), growth_rate=12, num_init_features=24, num_classes=num_classes)

def densenet18(num_classes=NUM_CLASSES):
    return DenseNet((3, 3, 3, 3), growth_rate=12, num_init_features=24, num_classes=num_classes)

def densenet34(num_classes=NUM_CLASSES):
    return DenseNet((6, 6, 6, 6), growth_rate=12, num_init_features=24, num_classes=num_classes)

# ───────────────────────── METRICS HELPERS ─────────────────────────
def _get_test_images():
    """Get list of test images from test_images directory."""
    image_paths = list(TEST_IMAGES_DIR.glob("*.png")) + list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No test images found in {TEST_IMAGES_DIR}")
    return sorted(image_paths)[:8]  # Use up to 8 images

def _compute_top_percentage_stats_multi_image(all_activations_per_image):
    """
    Compute Top 1%, Top 5%, and Top 10% activation statistics across multiple images.
    
    Args:
        all_activations_per_image: List[List[np.ndarray]] - [image][layer][activation_data]
        
    Returns:
        Dict with percentile statistics (mean and std across images)
    """
    if not all_activations_per_image:
        return {}
    
    num_layers = len(all_activations_per_image[0])
    
    # Initialize containers for each layer
    top1_means, top1_stds = [], []
    top5_means, top5_stds = [], []
    top10_means, top10_stds = [], []
    
    for layer_idx in range(num_layers):
        # Collect activations for this layer across all images
        layer_activations = []
        
        for img_activations in all_activations_per_image:
            if layer_idx < len(img_activations):
                act = img_activations[layer_idx]
                if act.size > 0:
                    flat = act.flatten()
                    layer_activations.append(flat)
        
        if not layer_activations:
            top1_means.append(0.0)
            top1_stds.append(0.0)
            top5_means.append(0.0)
            top5_stds.append(0.0)
            top10_means.append(0.0)
            top10_stds.append(0.0)
            continue
        
        # Calculate percentile statistics for each image
        top1_vals, top5_vals, top10_vals = [], [], []
        
        for flat in layer_activations:
            if flat.size == 0:
                continue
                
            # Calculate percentile cutoffs
            top1_cutoff = np.percentile(flat, 99)  # Top 1%
            top5_cutoff = np.percentile(flat, 95)  # Top 5%
            top10_cutoff = np.percentile(flat, 90)  # Top 10%
            
            # Get values above cutoffs
            top1_mask = flat >= top1_cutoff
            top5_mask = flat >= top5_cutoff
            top10_mask = flat >= top10_cutoff
            
            # Calculate means of top percentages
            top1_mean = np.mean(flat[top1_mask]) if np.any(top1_mask) else 0.0
            top5_mean = np.mean(flat[top5_mask]) if np.any(top5_mask) else 0.0
            top10_mean = np.mean(flat[top10_mask]) if np.any(top10_mask) else 0.0
            
            top1_vals.append(float(top1_mean))  # Convert to Python float
            top5_vals.append(float(top5_mean))  # Convert to Python float
            top10_vals.append(float(top10_mean))  # Convert to Python float
        
        # Calculate mean and std across images for this layer
        top1_means.append(float(np.mean(top1_vals)) if top1_vals else 0.0)
        top1_stds.append(float(np.std(top1_vals)) if top1_vals else 0.0)
        top5_means.append(float(np.mean(top5_vals)) if top5_vals else 0.0)
        top5_stds.append(float(np.std(top5_vals)) if top5_vals else 0.0)
        top10_means.append(float(np.mean(top10_vals)) if top10_vals else 0.0)
        top10_stds.append(float(np.std(top10_vals)) if top10_vals else 0.0)
    
    return {
        'top1_mean': top1_means,
        'top1_std': top1_stds,
        'top5_mean': top5_means,
        'top5_std': top5_stds,
        'top10_mean': top10_means,
        'top10_std': top10_stds
    }

def _plot_percentile_metrics_multi_with_layers(percentile_stats, label, safe_label, layer_info):
    """
    Create a single beautiful plot showing Top 1%, 5%, and 10% activation metrics per architectural layer
    with error bars for std, layer type annotations, and clear block boundaries.
    """
    if not percentile_stats or not percentile_stats.get('top1_mean'):
        return
        
    # Create a single larger plot with better proportions
    fig, ax = plt.subplots(1, 1, figsize=(24, 10), dpi=300)
    
    x_vals = np.array(range(1, len(percentile_stats['top1_mean']) + 1))
    
    # Professional color scheme with better contrast
    colors = {
        'top1': '#D32F2F',    # Deep red for top 1%
        'top5': '#FF8F00',    # Vibrant orange for top 5%
        'top10': '#FBC02D',   # Golden yellow for top 10%
    }
    
    markers = {
        'top1': 'o',     # Circle for top1%
        'top5': 's',     # Square for top5%
        'top10': '^',    # Triangle for top10%
    }
    
    # Plot mean values with error bars (std as shaded regions + error bars)
    for metric, color in colors.items():
        means = percentile_stats[f'{metric}_mean']
        stds = percentile_stats[f'{metric}_std']
        
        # Main line plot with markers
        line = ax.plot(x_vals, means, 
                      color=color, marker=markers[metric], 
                      linewidth=3, markersize=8, 
                      label=f'{metric.replace("top", "Top ").replace("1", "1%").replace("5", "5%").replace("10", "10%")}',
                      alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)
        
        # Shaded error region (mean ± std)
        ax.fill_between(x_vals, 
                       np.array(means) - np.array(stds), 
                       np.array(means) + np.array(stds),
                       color=color, alpha=0.15, linewidth=0)
        
        # Error bars for precise std indication
        ax.errorbar(x_vals, means, yerr=stds,
                   color=color, capsize=4, capthick=1.5, 
                   alpha=0.6, linewidth=0, elinewidth=1.5)
    
    # Add layer type annotations and block boundaries with improved clarity
    _add_layer_annotations_multi_beautiful(ax, layer_info, x_vals)
    
    # Styling
    ax.set_xlabel('Architectural Layer Index', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Input Activation Value', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_title(f'{label} - Top Percentile Input Activation Analysis\n'
                 f'Mean ± Standard Deviation across multiple test images\n'
                 f'(Input activations to architectural layers including residuals/concatenations)',
                 fontsize=18, fontweight='bold', pad=30, color='#2C3E50')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, color='#7F8C8D', linewidth=1)
    ax.set_axisbelow(True)
    
    # Legend styling
    legend1 = ax.legend(loc='upper left', fontsize=14, frameon=True, 
                       fancybox=True, shadow=True, framealpha=0.9,
                       edgecolor='#BDC3C7', facecolor='white')
    legend1.get_frame().set_linewidth(1.5)
    
    # Spine styling
    for spine_name, spine in ax.spines.items():
        if spine_name in ['top', 'right']:
            spine.set_visible(False)
        else:
            spine.set_color('#BDC3C7')
            spine.set_linewidth(2)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13, colors='#34495E',
                   length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    
    # Background color
    ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('white')
    
    # Tight layout with padding
    plt.tight_layout(pad=2.0)
    
    # Save plot with high quality
    plot_path = PLOT_DIR / f"{safe_label}_architectural_layers_percentile_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                pad_inches=0.2)
    plt.close()
    print(f"  [OK] Beautiful architectural layer percentile metrics plot saved to {plot_path}")

def _add_layer_annotations_multi_beautiful(ax, layer_info, x_vals):
    """Add beautiful layer type annotations and clear block boundaries to the plot."""
    if not layer_info:
        return
    
    # Enhanced layer type colors with better contrast
    type_colors = {
        'Conv': '#3498DB',          # Bright blue
        'BN': '#E74C3C',            # Bright red
        'ReLU': '#F39C12',          # Bright orange
        'MaxPool': '#9B59B6',       # Purple
        'AvgPool': '#1ABC9C',       # Teal
        'Linear': '#27AE60',        # Green
        'Dropout': '#95A5A6',       # Gray
        'ResidualBlock': '#E67E22', # Orange (for ResNet blocks)
        'PlainBlock': '#D35400',    # Dark orange (for Plain blocks)
        'DenseBlock': '#8E44AD',    # Purple (for DenseNet blocks)
        'Transition': '#16A085'     # Dark teal (for DenseNet transitions)
    }
    
    # Get y-axis limits for positioning
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Add colored markers for layer types at the top
    marker_y = y_max + y_range * 0.08
    
    for i, (layer_type, block_info) in enumerate(layer_info):
        if i < len(x_vals):
            color = type_colors.get(layer_type, '#7F8C8D')
            # Larger, more visible markers
            ax.scatter(x_vals[i], marker_y, color=color, s=120, 
                      marker='s', alpha=0.9, edgecolors='white', linewidths=1)
            
            # Add text annotation for layer type (rotated for clarity)
            if i % 2 == 0:  # Annotate every 2nd layer since we have fewer architectural layers
                ax.text(x_vals[i], marker_y + y_range * 0.04, 
                       layer_type, rotation=45, ha='left', va='bottom',
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor=color, alpha=0.8, linewidth=1))
    
    # Add block boundaries with clearer visual separation
    current_block = None
    block_positions = []
    
    for i, (layer_type, block_info) in enumerate(layer_info):
        if block_info != current_block and current_block is not None:
            # Store block transition position
            block_positions.append((x_vals[i] - 0.5, current_block, block_info))
        current_block = block_info
    
    # Draw block boundaries
    for pos, prev_block, next_block in block_positions:
        # Vertical line for block boundary
        ax.axvline(x=pos, color='#E67E22', linestyle='-', alpha=0.8, linewidth=3)
        
        # Block transition label with better positioning
        label_y = y_max + y_range * 0.18
        ax.text(pos, label_y, f'{prev_block} → {next_block}', 
               rotation=45, ha='left', va='bottom',
               fontsize=10, color='#E67E22', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#E67E22', alpha=0.8))
    
    # Create a beautiful legend for layer types
    legend_elements = []
    unique_types = sorted(list(set([info[0] for info in layer_info])))
    
    for layer_type in unique_types:
        if layer_type in type_colors:
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', 
                          markerfacecolor=type_colors[layer_type], 
                          markersize=12, label=layer_type, alpha=0.9,
                          markeredgecolor='white', markeredgewidth=1)
            )
    
    if legend_elements:
        # Position legend on the right side
        legend2 = ax.legend(handles=legend_elements, loc='center left', 
                           bbox_to_anchor=(1.02, 0.5), fontsize=12, 
                           title='Layer Types', title_fontsize=14,
                           frameon=True, fancybox=True, shadow=True,
                           framealpha=0.9, edgecolor='#BDC3C7')
        legend2.get_frame().set_linewidth(1.5)
        legend2.get_title().set_fontweight('bold')
        legend2.get_title().set_color('#2C3E50')
    
    # Adjust y-axis limits to accommodate annotations
    ax.set_ylim(y_min, y_max + y_range * 0.28)

# ───────────────────────── ANALYSIS FUNCTIONS ─────────────────────────
def get_architectural_layers_with_info(tag: str, model: nn.Module):
    """
    Extract architectural layers that properly capture inputs including residuals/concatenations.
    Returns list of (layer, layer_type, block_info) tuples.
    """
    layers_info = []
    
    if "resnet" in tag or "plain" in tag:
        # ResNet/Plain: Hook at architectural points where residuals are included
        for name, module in model.named_children():
            if name == "conv1":
                layers_info.append((module, "Conv", "stem"))
            elif name == "bn1":
                layers_info.append((module, "BN", "stem"))
            elif name == "relu":
                layers_info.append((module, "ReLU", "stem"))
            elif name == "maxpool":
                layers_info.append((module, "MaxPool", "stem"))
            elif name.startswith("layer"):
                # Hook each BasicBlock/PlainBlock - these capture inputs AFTER residual additions (for ResNet)
                block_idx = 0
                for block_name, block in module.named_children():
                    block_idx += 1
                    block_type = "ResidualBlock" if "resnet" in tag else "PlainBlock"
                    layers_info.append((block, block_type, f"{name}.{block_idx}"))
            elif name == "avgpool":
                layers_info.append((module, "AvgPool", "head"))
            elif name == "fc":
                layers_info.append((module, "Linear", "head"))
                
    elif "densenet" in tag:
        # DenseNet: Hook at points where concatenations are included
        for name, module in model.named_children():
            if name == "features":
                for feat_name, feat_module in module.named_children():
                    if feat_name.startswith("conv") or feat_name.startswith("norm") or feat_name.startswith("relu") or feat_name.startswith("pool"):
                        layer_type = "Conv" if "conv" in feat_name else ("BN" if "norm" in feat_name else ("ReLU" if "relu" in feat_name else "MaxPool"))
                        layers_info.append((feat_module, layer_type, "stem"))
                    elif feat_name.startswith("denseblock"):
                        # Hook the entire DenseBlock - captures inputs AFTER concatenations
                        layers_info.append((feat_module, "DenseBlock", feat_name))
                    elif feat_name.startswith("transition"):
                        # Hook transition layers - they receive concatenated inputs
                        layers_info.append((feat_module, "Transition", feat_name))
                    elif feat_name == "norm_final":
                        layers_info.append((feat_module, "BN", "head"))
            elif name == "classifier":
                layers_info.append((module, "Linear", "head"))
    
    return layers_info

def get_blocks_for_analysis(tag: str, model: nn.Module):
    """Extract blocks from model for analysis."""
    blocks = []
    if "resnet" in tag or "plain" in tag:
        # For ResNet/Plain: layer1, layer2, layer3, layer4
        for name, module in model.named_children():
            if name.startswith("layer"):
                blocks.append(module)
    elif "densenet" in tag:
        # For DenseNet: denseblock1, denseblock2, ...
        for name, module in model.features.named_children():
            if "denseblock" in name:
                blocks.append(module)
    return blocks

def get_layers_for_analysis(tag: str, model: nn.Module):
    """Extract individual layers from model for analysis."""
    layers = []
    
    def extract_resnet_layers(layer_group):
        layer_list = []
        for name, module in layer_group.named_children():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_type = "Conv" if isinstance(module, nn.Conv2d) else "Linear"
                layer_list.append((module, layer_type))
            elif hasattr(module, 'conv1'):  # BasicBlock or PlainBlock
                if hasattr(module, 'conv1'):
                    layer_list.append((module.conv1, "Conv"))
                if hasattr(module, 'conv2'):
                    layer_list.append((module.conv2, "Conv"))
        return layer_list
    
    def extract_densenet_layers(module, layers_list):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                layers_list.append((child, "Conv"))
            elif isinstance(child, nn.Linear):
                layers_list.append((child, "Linear"))
            elif hasattr(child, 'children'):
                extract_densenet_layers(child, layers_list)
    
    if "resnet" in tag or "plain" in tag:
        # Process each layer group
        for name, module in model.named_children():
            if name.startswith("layer"):
                layers.extend(extract_resnet_layers(module))
            elif name == "conv1":
                layers.append((module, "Conv"))
            elif name == "fc":
                layers.append((module, "Linear"))
    elif "densenet" in tag:
        extract_densenet_layers(model, layers)
    
    return layers

def strip_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_trained_model(tag: str):
    """Load a trained model from checkpoint."""
    # Model factory
    model_fns = {
        "resnet14": resnet14, "resnet18": resnet18, "resnet34": resnet34,
        "plain14": plain14, "plain18": plain18, "plain34": plain34,
        "densenet14": densenet14, "densenet18": densenet18, "densenet34": densenet34,
    }
    
    if tag not in model_fns:
        raise ValueError(f"Unknown model tag: {tag}")
    
    model = model_fns[tag]()
    
    # Load checkpoint
    ckpt_path = CHK_DIR / f"{tag}.pt"
    if ckpt_path.exists():
        print(f"  [INFO] Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        state_dict = strip_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"  [WARN] No checkpoint found at {ckpt_path}, using random weights")
    
    return model.to(DEVICE).eval()

@torch.no_grad()
def analyse_metrics_architectural(tag: str, model: nn.Module, results: Dict[str, Dict[str, List[float]]]):
    """
    Enhanced analysis function with multi-image percentile metrics for architectural layers.
    CRITICAL: Captures INPUT activations to architectural layers (includes residuals/concatenations).
    """
    model.eval().to(DEVICE)
    
    # Get architectural layers with detailed information
    layers_info = get_architectural_layers_with_info(tag, model)
    print(f"  [INFO] Found {len(layers_info)} architectural layers for {tag}")

    # Get test images
    try:
        image_paths = _get_test_images()
        print(f"  [INFO] Processing {len(image_paths)} test images")
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return

    # Prepare image transform
    if DATASET == "cifar100":
        n_mean, n_std = [0.5071,0.4865,0.4409], [0.2673,0.2564,0.2762]
    else:
        n_mean, n_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    transform = T.Compose([
        T.Resize((IMG_RES, IMG_RES)),
        T.ToTensor(),
        T.Normalize(n_mean, n_std),
    ])

    # Initialize containers
    aL2m, aL2s = [], []
    pL2m, pL2s = [], []
    all_activations_per_image = []  # List[List[np.ndarray]]

    # Extract actual modules and layer types
    modules = [layer for layer, layer_type, block_info in layers_info]

    # Parameter L2 stats (same for all images)
    for unit in modules:
        param_list = [p.detach().flatten() for p in unit.parameters() if p.requires_grad]
        if param_list:
            unit_l2s = torch.tensor([p.detach().norm() for p in unit.parameters() if p.requires_grad])
            pL2m.append(unit_l2s.mean().item())
            pL2s.append(unit_l2s.std().item())
        else:
            pL2m.append(0.0)
            pL2s.append(0.0)

    # Process each test image
    for img_idx, img_path in enumerate(image_paths):
        print(f"  [INFO] Processing image {img_idx + 1}/{len(image_paths)}: {img_path.name}")
        
        # Prepare containers for this image
        img_activations = []
        img_aL2m, img_aL2s = [], []

        # ARCHITECTURAL INPUT activation hooking for this image
        def architectural_input_hook_fn(module, input_tensor, output):
            """Hook that captures INPUT to architectural layers (includes residuals/concatenations)"""
            if isinstance(input_tensor, tuple):
                inpt = input_tensor[0].detach().float()
            else:
                inpt = input_tensor.detach().float()
            
            if inpt.ndim == 4:  # Spatial features: [B,C,H,W]
                inpt = inpt.squeeze(0)  # Remove batch dimension -> [C,H,W]
                img_activations.append(inpt.cpu().numpy())
                
                # L2 across channels
                fc = inpt.flatten(1)  # [C, H*W]
                norms_c = fc.norm(dim=1, p=2)
                img_aL2m.append(norms_c.mean().item())
                img_aL2s.append(norms_c.std().item())
                
            elif inpt.ndim == 2:  # Linear layer features: [B,Features]
                inpt = inpt.squeeze(0)  # Remove batch dimension -> [Features]
                img_activations.append(inpt.cpu().numpy())
                img_aL2m.append(inpt.norm(p=2).item())
                img_aL2s.append(0.0)
                
            else:  # Fallback for other shapes
                if inpt.shape[0] == 1:
                    inpt = inpt.squeeze(0)
                img_activations.append(inpt.cpu().numpy())
                flat = inpt.flatten()
                img_aL2m.append(flat.norm(p=2).item())
                img_aL2s.append(0.0)

        # Register INPUT hooks for each architectural layer
        hdls = [unit.register_forward_hook(architectural_input_hook_fn) for unit in modules]

        # Forward pass for this image
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        _ = model(x)
        
        # Remove hooks
        for h in hdls:
            h.remove()
        
        # Store activations for this image
        all_activations_per_image.append(img_activations)
        
        # Accumulate L2 stats
        if img_idx == 0:
            aL2m = img_aL2m.copy()
            aL2s = img_aL2s.copy()
        else:
            # Update running averages
            for i in range(len(img_aL2m)):
                if i < len(aL2m):
                    aL2m[i] = (aL2m[i] * img_idx + img_aL2m[i]) / (img_idx + 1)
                    aL2s[i] = (aL2s[i] * img_idx + img_aL2s[i]) / (img_idx + 1)

    # Calculate percentile statistics across all images
    print("  [INFO] Computing percentile statistics across all images...")
    percentile_stats = _compute_top_percentage_stats_multi_image(all_activations_per_image)

    # Store stats
    stats = {
        "l2_mean": aL2m,
        "l2_std": aL2s,
        "param_l2_mean": pL2m,
        "param_l2_std": pL2s,
        **percentile_stats  # Add percentile statistics
    }
    
    # Generate appropriate label
    plot_label = f"{DATASET}_{tag}_architectural_layers_metrics"
    results[f"{tag}_architectural_layers_metrics"] = stats
    
    # Save stats to JSON
    (PLOT_DIR / f"{plot_label}.json").write_text(json.dumps(stats, indent=2))
    print(f"  [OK] Stats saved to {PLOT_DIR / f'{plot_label}.json'}")

    # Generate percentile metrics plot with layer information
    layer_type_info = [(layer_type, block_info) for layer, layer_type, block_info in layers_info]
    _plot_percentile_metrics_multi_with_layers(percentile_stats, f"{tag} architectural layers", plot_label, layer_type_info)

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    print("Starting ResNet vs Plain vs Dense Architectural Layer Metrics Analysis")
    print(f"Dataset: {DATASET}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {PLOT_DIR}")
    
    # Check for test images
    try:
        test_images = _get_test_images()
        print(f"Found {len(test_images)} test images")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    if not args.no_analysis:
        # Models to analyze
        model_tags = ["resnet14", "resnet18", "resnet34", 
                      "plain14", "plain18", "plain34",
                      "densenet14", "densenet18", "densenet34"]
        
        all_results = {}
        
        for tag in model_tags:
            try:
                print(f"\n{'='*60}")
                print(f"Processing {tag}")
                print(f"{'='*60}")
                
                # Load trained model
                model = load_trained_model(tag)
                
                # Analyze architectural layers (INPUT activations)
                print(f"Analyzing {tag} architectural layers (INPUT activations)...")
                analyse_metrics_architectural(tag, model, all_results)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {tag}: {e}")
                continue
        
        # Save combined results
        combined_results_path = PLOT_DIR / "all_architectural_layer_metrics_results.json"
        with open(combined_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_results_path}")
        
        print(f"\nArchitectural layer metrics analysis complete! Check {PLOT_DIR} for plots and results.")
    else:
        print("Analysis skipped (--no_analysis flag set)") 