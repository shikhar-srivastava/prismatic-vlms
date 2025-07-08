#!/usr/bin/env python3
# analyze_cnn_whisker_box_metrics.py
#
# Purpose:
#   Generate layer-wise activation & parameter statistics for multiple pretrained CNNs,
#   focusing on Top 1%, Top 5%, and Top 10% activation metrics across multiple test images.
#   Creates beautiful mean and std plots of these metrics for each architectural layer.
#   CRITICAL: Analyzes INPUT activations to architectural layers (properly includes residuals/concatenations).
#
# Key Features:
#   - ResNet: Hooks at BasicBlock level to capture inputs AFTER residual additions
#   - DenseNet: Hooks at DenseBlock/Transition level to capture inputs AFTER concatenations  
#   - VGG/CNN: Hooks at individual layer level (no residuals/concatenations)
#
# Usage example:
#   python analyze_cnn_whisker_box_metrics.py
#
# Requirements:
#   - PyTorch
#   - torchvision
#   - matplotlib
#   - Pillow (for Image I/O)
#   - numpy
#
# Note:
#   Make sure you have test images in "test_images/" directory in the working directory.

from __future__ import annotations
import gc
import json
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_cnn_box_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set matplotlib style more compatibly
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    try:
        plt.style.use("seaborn-whitegrid")
    except:
        plt.style.use("default")
        
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGES_DIR = BASE_DIR / "test_images"

# Models to analyze
MODELS: Dict[str, str] = {
    "resnet":   "resnet18",     # Residual net
    "densenet": "densenet121",  # Dense net
    "cnn":      "vgg16_bn",     # Plain-ish CNN
}

# ───────────────────────── HELPERS ─────────────────────────
def _get_test_images():
    """Get list of test images from test_images directory."""
    image_paths = list(TEST_IMAGES_DIR.glob("*.png")) + list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No test images found in {TEST_IMAGES_DIR}")
    return sorted(image_paths)[:8]  # Use up to 8 images

def _compute_top_percentage_stats(activations_list):
    """
    Compute Top 1%, Top 5%, and Top 10% activation statistics across multiple images.
    
    Args:
        activations_list: List of lists, where each inner list contains activations for each layer
        
    Returns:
        Dict with percentile statistics (mean and std across images)
    """
    if not activations_list:
        return {}
    
    num_layers = len(activations_list[0])
    
    # Initialize containers for each layer
    top1_means, top1_stds = [], []
    top5_means, top5_stds = [], []
    top10_means, top10_stds = [], []
    
    for layer_idx in range(num_layers):
        # Collect activations for this layer across all images
        layer_activations = []
        
        for img_activations in activations_list:
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

def _plot_percentile_metrics_with_layers(percentile_stats, label, safe_label, layer_info):
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
    _add_layer_annotations_beautiful(ax, layer_info, x_vals)
    
    # Styling
    ax.set_xlabel('Architectural Layer Index', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Input Activation Value', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_title(f'{label} - Top Percentile Input Activation Analysis\n'
                 f'Mean ± Standard Deviation across multiple test images\n'
                 f'(Input activations to each layer including residuals/concatenations)',
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
    plot_path = OUT_DIR / f"{safe_label}_architectural_layers_percentile_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                pad_inches=0.2)
    plt.close()
    print(f"  [OK] Beautiful architectural layer percentile metrics plot saved to {plot_path}")

def _add_layer_annotations_beautiful(ax, layer_info, x_vals):
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

def _transform(size: int = 224) -> T.Compose:
    """
    Standard ImageNet preprocessing pipeline.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _get_architectural_layers_with_info(model: torch.nn.Module, tag: str):
    """
    Extract architectural layers that properly capture inputs including residuals/concatenations.
    Returns list of (layer, layer_type, block_info) tuples.
    """
    layers_info = []
    
    if tag == "resnet":
        # ResNet: Hook at architectural points where residuals are included
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
                # Hook each BasicBlock/Bottleneck - these capture inputs AFTER residual additions
                block_idx = 0
                for block_name, block in module.named_children():
                    block_idx += 1
                    layers_info.append((block, "ResidualBlock", f"{name}.{block_idx}"))
            elif name == "avgpool":
                layers_info.append((module, "AvgPool", "head"))
            elif name == "fc":
                layers_info.append((module, "Linear", "head"))
                
    elif tag == "densenet":
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
                
    elif tag == "cnn":
        # VGG-like: No residuals/concatenations, so atomic layers are fine
        for name, module in model.named_children():
            if name == "features":
                layer_idx = 0
                for layer in module:
                    layer_idx += 1
                    if isinstance(layer, torch.nn.Conv2d):
                        layers_info.append((layer, "Conv", f"conv_block_{(layer_idx-1)//3 + 1}"))
                    elif isinstance(layer, torch.nn.BatchNorm2d):
                        layers_info.append((layer, "BN", f"conv_block_{(layer_idx-1)//3 + 1}"))
                    elif isinstance(layer, torch.nn.ReLU):
                        layers_info.append((layer, "ReLU", f"conv_block_{(layer_idx-1)//3 + 1}"))
                    elif isinstance(layer, torch.nn.MaxPool2d):
                        layers_info.append((layer, "MaxPool", f"conv_block_{(layer_idx-1)//3 + 1}"))
            elif name == "avgpool":
                layers_info.append((module, "AvgPool", "head"))
            elif name == "classifier":
                for cls_idx, cls_layer in enumerate(module):
                    if isinstance(cls_layer, torch.nn.Linear):
                        layers_info.append((cls_layer, "Linear", f"classifier.{cls_idx}"))
                    elif isinstance(cls_layer, torch.nn.ReLU):
                        layers_info.append((cls_layer, "ReLU", f"classifier.{cls_idx}"))
                    elif isinstance(cls_layer, torch.nn.Dropout):
                        layers_info.append((cls_layer, "Dropout", f"classifier.{cls_idx}"))
    
    return layers_info

def _param_stats(m: torch.nn.Module) -> Tuple[float, float, float, float]:
    """
    Compute parameter statistics for a module.
    Returns: (l2_mean, l2_std, raw_mean, raw_std)
    """
    params = list(m.parameters())
    if not params:
        return 0.0, 0.0, 0.0, 0.0
    
    all_params = torch.cat([p.flatten() for p in params])
    l2_norm = torch.norm(all_params, p=2).item()
    raw_mean = all_params.mean().item()
    raw_std = all_params.std().item()
    
    return l2_norm, 0.0, raw_mean, raw_std

def _load_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """
    Load a torchvision model with pretrained weights, handling different API versions.
    """
    fn = getattr(tvm, model_id, None)
    if fn is None:
        raise ValueError(f"Unknown model: {model_id}")

    # Try multiple approaches to load the model with pretrained weights
    try:
        # First, try the new weights API with DEFAULT weights
        weights_attr_name = f"{model_id.upper()}_Weights"
        if hasattr(tvm, weights_attr_name):
            weights_enum = getattr(tvm, weights_attr_name)
            if hasattr(weights_enum, 'DEFAULT'):
                model = fn(weights=weights_enum.DEFAULT)
                return model.to(device).eval()
    except Exception as e:
        print(f"  [DEBUG] New weights API failed for {model_id}: {e}")

    try:
        # Try the legacy weights API with string
        model = fn(weights='DEFAULT')
        return model.to(device).eval()
    except Exception as e:
        print(f"  [DEBUG] String weights API failed for {model_id}: {e}")

    try:
        # Fallback to old pretrained=True API
        model = fn(pretrained=True)
        return model.to(device).eval()
    except Exception as e:
        print(f"  [DEBUG] Legacy pretrained API failed for {model_id}: {e}")

    # Final fallback: load without pretrained weights
    print(f"  [WARN] Loading {model_id} without pretrained weights")
    model = fn()
    return model.to(device).eval()

# ───────────────────────── ANALYSIS ─────────────────────────
def analyse(tag: str, model_id: str, max_layers: Optional[int]) -> Dict[str, List[float]]:
    """
    Analyze the specified model across multiple test images using architectural layers:
      1. Identify architectural layers that capture inputs AFTER residuals/concatenations.
      2. Gather parameter L2 stats.
      3. Forward on multiple test images, hooking INPUT to each architectural layer.
      4. Calculate Top 1%, 5%, 10% percentile statistics across images.
      5. Generate plots with layer type annotations and block boundaries.

    Args:
        tag: A short label identifying the net (e.g. 'resnet', 'densenet', 'cnn').
        model_id: The torchvision constructor name (e.g. 'resnet18').
        max_layers: If not None, limit analysis to the first N layers.

    Returns: A dict with metric name -> list of per-layer stats.
    """
    analysis_type = "architectural_layers"
    label = f"{model_id}_{analysis_type}"
    print(f"\n=== Analyzing {label} ===")

    # Load model
    try:
        model = _load_model(model_id, DEVICE)
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower():
            print("  [!] GPU OOM – attempting CPU …")
            model = _load_model(model_id, torch.device("cpu"))
        else:
            raise

    # Get architectural layers with detailed information
    layers_info = _get_architectural_layers_with_info(model, tag)
    print(f"  [INFO] Found {len(layers_info)} architectural layers")
    
    if not layers_info:
        raise ValueError(f"No architectural layers found for {model_id}")
    if max_layers is not None and len(layers_info) > max_layers:
        layers_info = layers_info[:max_layers]

    # Get test images
    image_paths = _get_test_images()
    print(f"  [INFO] Processing {len(image_paths)} test images")

    # Prepare containers for stats
    act_L2_m, act_L2_s    = [], []
    act_raw_m, act_raw_s  = [], []
    par_L2_m, par_L2_s    = [], []
    par_raw_m, par_raw_s  = [], []
    all_activations_per_image = []  # List of lists: [image][layer]

    # Parameter stats
    for layer, layer_type, block_info in layers_info:
        l2m, l2s, rm, rs = _param_stats(layer)
        par_L2_m.append(l2m)
        par_L2_s.append(l2s)
        par_raw_m.append(rm)
        par_raw_s.append(rs)

    # Process each test image
    for img_idx, img_path in enumerate(image_paths):
        print(f"  [INFO] Processing image {img_idx + 1}/{len(image_paths)}: {img_path.name}")
        
        # Prepare containers for this image
        img_activations = []
        img_act_L2_m, img_act_L2_s = [], []
        img_act_raw_m, img_act_raw_s = [], []

        # ARCHITECTURAL INPUT activation stats: forward hooks that capture INPUTS with residuals/concatenations
        def _architectural_input_hook(module, input_tensor, output):
            """Hook that captures INPUT to architectural layers (includes residuals/concatenations)"""
            if isinstance(input_tensor, tuple):
                h = input_tensor[0].detach().float()
            else:
                h = input_tensor.detach().float()
                
            if h.ndim == 4:  # (B,C,H,W)
                # store raw input activations
                h = h.squeeze(0)  # Remove batch dimension
                img_activations.append(h.cpu().numpy())
                # L2 norms by channel
                l2vals = torch.norm(h.flatten(1), p=2, dim=1)
                img_act_L2_m.append(l2vals.mean().item())
                img_act_L2_s.append(l2vals.std().item())
                flat = h.flatten()
                img_act_raw_m.append(flat.mean().item())
                img_act_raw_s.append(flat.std().item())
            elif h.ndim == 2:  # (B,Features)
                h = h.squeeze(0)  # Remove batch dimension
                img_activations.append(h.cpu().numpy())
                img_act_L2_m.append(h.norm(p=2).item())
                img_act_L2_s.append(0.0)
                img_act_raw_m.append(h.mean().item())
                img_act_raw_s.append(h.std().item())
            else:
                # fallback for other shapes
                if h.shape[0] == 1:
                    h = h.squeeze(0)
                img_activations.append(h.cpu().numpy())
                flat = h.flatten()
                img_act_L2_m.append(flat.norm(p=2).item())
                img_act_L2_s.append(0.0)
                img_act_raw_m.append(flat.mean().item())
                img_act_raw_s.append(flat.std().item())

        # Register INPUT hooks for each architectural layer
        hdls = []
        for layer, layer_type, block_info in layers_info:
            hdls.append(layer.register_forward_hook(_architectural_input_hook))

        # Forward pass for this image
        img = Image.open(img_path).convert("RGB")
        x = _transform(224)(img).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for h in hdls:
            h.remove()
        
        # Store activations and stats for this image
        all_activations_per_image.append(img_activations)
        
        # Accumulate stats (will average later)
        if img_idx == 0:
            act_L2_m = img_act_L2_m.copy()
            act_L2_s = img_act_L2_s.copy()
            act_raw_m = img_act_raw_m.copy()
            act_raw_s = img_act_raw_s.copy()
        else:
            # Update running averages
            for i in range(len(img_act_L2_m)):
                if i < len(act_L2_m):
                    act_L2_m[i] = (act_L2_m[i] * img_idx + img_act_L2_m[i]) / (img_idx + 1)
                    act_L2_s[i] = (act_L2_s[i] * img_idx + img_act_L2_s[i]) / (img_idx + 1)
                    act_raw_m[i] = (act_raw_m[i] * img_idx + img_act_raw_m[i]) / (img_idx + 1)
                    act_raw_s[i] = (act_raw_s[i] * img_idx + img_act_raw_s[i]) / (img_idx + 1)

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Calculate percentile statistics across all images
    print("  [INFO] Computing percentile statistics across all images...")
    percentile_stats = _compute_top_percentage_stats(all_activations_per_image)

    # Prepare final stats dict
    stats = {
        "l2_mean":         act_L2_m,
        "l2_std":          act_L2_s,
        "raw_mean":        act_raw_m,
        "raw_std":         act_raw_s,
        "param_l2_mean":   par_L2_m,
        "param_l2_std":    par_L2_s,
        "param_raw_mean":  par_raw_m,
        "param_raw_std":   par_raw_s,
        **percentile_stats  # Add percentile statistics
    }

    # Save to JSON
    safe_label = re.sub(r"[/\\\\]", "__", label)
    out_json = OUT_DIR / f"{safe_label}_metrics.json"
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"  [OK] Stats saved to {out_json}")

    # Generate percentile metrics plot with layer information
    layer_type_info = [(layer_type, block_info) for layer, layer_type, block_info in layers_info]
    _plot_percentile_metrics_with_layers(percentile_stats, label, safe_label, layer_type_info)

    return stats

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    print("Starting CNN Architectural Layer Metrics Analysis")
    print(f"Output directory: {OUT_DIR}")
    print(f"Device: {DEVICE}")
    
    # Check for test images
    try:
        test_images = _get_test_images()
        print(f"Found {len(test_images)} test images")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    all_results = {}
    
    for tag, model_id in MODELS.items():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {tag}: {model_id}")
            print(f"{'='*60}")
            
            # Analyze architectural layers
            results = analyse(tag, model_id, max_layers=None)
            all_results[f"{tag}_architectural_layers"] = results
            
        except Exception as e:
            print(f"Error processing {tag} ({model_id}): {e}")
            continue
    
    # Save combined results
    combined_results_path = OUT_DIR / "all_architectural_layer_metrics_results.json"
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_results_path}")
    
    print(f"\nArchitectural layer analysis complete! Check {OUT_DIR} for plots and results.") 