#!/usr/bin/env python3
# analyze_cnn_whisker_box.py
#
# Purpose:
#   Generate layer-wise activation & parameter statistics for multiple pretrained CNNs,
#   and create whisker box plots of the raw activations (with outlier annotations).
#
#   This script is inspired by the "Residual vs Dense vs Plain CNN – dual-depth study"
#   reference, ensuring a whisker box plot of raw activations is generated.
#
# Usage example:
#   python analyze_cnn_whisker_box.py
#
# Requirements:
#   - PyTorch
#   - torchvision
#   - matplotlib
#   - Pillow (for Image I/O)
#
# Note:
#   Make sure you have a valid test image "test.png" (224×224) in the working directory,
#   or specify your own path below in IMAGE_PATH.

from __future__ import annotations
import gc
import json
import re
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
OUT_DIR = BASE_DIR / "viz" / "plots" / "act_analysis_cnn_box"
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
IMAGE_PATH = BASE_DIR / "test.png"  # Provide your own image here if needed.

# Models to analyze
MODELS: Dict[str, str] = {
    "resnet":   "resnet18",     # Residual net
    "densenet": "densenet121",  # Dense net
    "cnn":      "vgg16_bn",     # Plain-ish CNN
}

# ───────────────────────── HELPERS ─────────────────────────
def _create_test_image_if_needed():
    """Create a test image if it doesn't exist."""
    if not IMAGE_PATH.exists():
        print(f"[INFO] Creating test image at {IMAGE_PATH}")
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        img.save(IMAGE_PATH)

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
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Activation Value', fontsize=14, fontweight='bold', color='#333333') 
    ax.set_title(f'{label} - Top Activation Analysis\n'
                f'Layer-wise peak activations and median reference values',
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
    max_layers = len(top_stats['top1'])
    ax.set_xticks(range(1, max_layers + 1))
    
    # Add subtle shading to differentiate value ranges
    y_min, y_max = ax.get_ylim()
    
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
    if max_layers > 0:
        max_top1 = max(top_stats['top1'])
        max_median = max(top_stats['median'])
        textstr = f'Peak Top-1: {max_top1:.3f}\nPeak Median: {max_median:.3f}\nLayers: {max_layers}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Tight layout with enhanced padding
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    output_path = OUT_DIR / f"{safe_label}_top.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', 
               metadata={'Title': f'{label} Top Activations Analysis'})
    plt.close()
    
    print(f"  [OK] Top activations plot saved to {output_path}")

def _transform(size: int = 224) -> T.Compose:
    """
    Preprocessing pipeline for the input image.
    """
    return T.Compose([
        T.Resize(size + 32),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

def _get_blocks(model: torch.nn.Module, tag: str):
    """
    Extract the per-block or per-layer modules from each architecture.
    """
    if tag == "resnet":
        # For ResNet-18: the 4 major layers, each has multiple BasicBlocks.
        return [b for layer in [model.layer1, model.layer2, model.layer3, model.layer4]
                for b in layer]
    elif tag == "densenet":
        # For DenseNet-121: the 4 dense blocks, each with multiple DenseLayers.
        feats = model.features
        dbs = [feats.denseblock1, feats.denseblock2, feats.denseblock3, feats.denseblock4]
        blocks = []
        for db in dbs:
            # DenseNet blocks contain named children, we need to extract the actual modules
            for name, module in db.named_children():
                if hasattr(module, 'parameters'):  # Only add if it's actually a module
                    blocks.append(module)
        return blocks
    else:
        # For VGG16_bn: a sequential list with conv2d/batchnorm/relu layers.
        # We'll pick out only conv2d modules as the "blocks."
        return [m for m in model.features if isinstance(m, torch.nn.Conv2d)]

def _get_layers(model: torch.nn.Module, tag: str):
    """
    Extract individual layers (not blocks) from each architecture for fine-grained analysis.
    This provides more detailed layer-by-layer activation analysis.
    
    Returns:
        List of tuples (layer, layer_type_string) for annotation purposes
    """
    layers = []
    
    if tag == "resnet":
        # For ResNet: extract all individual layers including conv, bn, relu from each BasicBlock
        def extract_resnet_layers(layer_group):
            layer_list = []
            for block in layer_group:
                # Each BasicBlock typically has: conv1, bn1, relu, conv2, bn2, (optional downsample)
                for name, module in block.named_children():
                    if isinstance(module, torch.nn.Conv2d):
                        layer_list.append((module, "Conv2d"))
                    elif isinstance(module, torch.nn.BatchNorm2d):
                        layer_list.append((module, "BatchNorm2d"))
                    elif isinstance(module, torch.nn.ReLU):
                        layer_list.append((module, "ReLU"))
                    elif isinstance(module, torch.nn.Sequential):  # downsample
                        for sub_module in module:
                            if isinstance(sub_module, torch.nn.Conv2d):
                                layer_list.append((sub_module, "Conv2d"))
                            elif isinstance(sub_module, torch.nn.BatchNorm2d):
                                layer_list.append((sub_module, "BatchNorm2d"))
            return layer_list
        
        # Add initial layers
        if hasattr(model, 'conv1'):
            layers.append((model.conv1, "Conv2d"))
        if hasattr(model, 'bn1'):
            layers.append((model.bn1, "BatchNorm2d"))
        if hasattr(model, 'relu'):
            layers.append((model.relu, "ReLU"))
        if hasattr(model, 'maxpool'):  # Add missing maxpool layer
            layers.append((model.maxpool, "MaxPool2d"))
            
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
            
    elif tag == "densenet":
        # For DenseNet: extract individual layers from features
        def extract_densenet_layers(module, layers_list):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Conv2d):
                    layers_list.append((child, "Conv2d"))
                elif isinstance(child, torch.nn.BatchNorm2d):
                    layers_list.append((child, "BatchNorm2d"))
                elif isinstance(child, torch.nn.ReLU):
                    layers_list.append((child, "ReLU"))
                elif isinstance(child, torch.nn.AvgPool2d):
                    layers_list.append((child, "AvgPool2d"))
                elif hasattr(child, 'named_children'):  # Recurse into submodules
                    extract_densenet_layers(child, layers_list)
        
        extract_densenet_layers(model.features, layers)
        
        # Add missing final classifier layer
        if hasattr(model, 'classifier'):
            layers.append((model.classifier, "Linear"))
        
    else:  # VGG and other CNNs
        # For VGG: extract all layers from features (conv, bn, relu, maxpool)
        for module in model.features:
            if isinstance(module, torch.nn.Conv2d):
                layers.append((module, "Conv2d"))
            elif isinstance(module, torch.nn.BatchNorm2d):
                layers.append((module, "BatchNorm2d"))
            elif isinstance(module, torch.nn.ReLU):
                layers.append((module, "ReLU"))
            elif isinstance(module, torch.nn.MaxPool2d):
                layers.append((module, "MaxPool2d"))
    
    # Filter out any None values and ensure we have actual parameters
    layers = [(layer, layer_type) for layer, layer_type in layers if layer is not None and 
              (hasattr(layer, 'weight') or isinstance(layer, (torch.nn.ReLU, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)))]
    
    return layers

def _param_stats(m: torch.nn.Module) -> Tuple[float, float, float, float]:
    """
    Compute parameter-level stats:
       - param_l2_mean, param_l2_std : L2 norms of each parameter tensor
       - param_raw_mean, param_raw_std : raw param values across the entire block
    Returns (param_l2_mean, param_l2_std, param_raw_mean, param_raw_std).
    """
    vecs = [p.detach().float().reshape(-1) for p in m.parameters() if p.requires_grad]
    if not vecs:
        return 0.0, 0.0, 0.0, 0.0
    flat = torch.cat(vecs)
    l2s  = torch.stack([v.norm(2) for v in vecs])
    return (l2s.mean().item(), l2s.std().item(), flat.mean().item(), flat.std().item())

def _load_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """
    Load an ImageNet-pretrained torchvision model by ID, using either:
      - new Weights Enum API (if available),
      - or legacy pretrained=True approach.
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
def analyse(tag: str, model_id: str, max_blocks: Optional[int], use_layers: bool = False) -> Dict[str, List[float]]:
    """
    Analyze the specified model:
      1. Identify blocks or layers (based on use_layers parameter).
      2. Gather parameter L2 stats.
      3. Forward on a test image, hooking each block/layer's output to gather activation stats.
      4. Generate whisker box plot with outliers & store stats in JSON.

    Args:
        tag: A short label identifying the net (e.g. 'resnet', 'densenet', 'cnn').
        model_id: The torchvision constructor name (e.g. 'resnet18').
        max_blocks: If not None, limit analysis to the first N blocks/layers.
        use_layers: If True, analyze individual layers instead of blocks.

    Returns: A dict with metric name -> list of per-block/layer stats.
    """
    analysis_type = "layers" if use_layers else ("full" if max_blocks is None else f"top{max_blocks}")
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

    # Identify the blocks or layers
    if use_layers:
        blocks = _get_layers(model, tag)
        print(f"  [INFO] Found {len(blocks)} individual layers")
    else:
        blocks = _get_blocks(model, tag)
        print(f"  [INFO] Found {len(blocks)} blocks")
        
    if not blocks:
        raise ValueError(f"No {'layers' if use_layers else 'blocks'} found for {model_id}")
    if max_blocks is not None and len(blocks) > max_blocks:
        blocks = blocks[:max_blocks]

    # Prepare containers for stats
    act_L2_m, act_L2_s    = [], []
    act_raw_m, act_raw_s  = [], []
    par_L2_m, par_L2_s    = [], []
    par_raw_m, par_raw_s  = [], []
    activations           = []
    outliers              = []
    layer_types           = None  # Initialize layer_types to avoid scope issues

    # Parameter stats - handle both block and layer formats
    for item in blocks:
        if use_layers:
            layer, layer_type = item
            l2m, l2s, rm, rs = _param_stats(layer)
        else:
            l2m, l2s, rm, rs = _param_stats(item)
        par_L2_m.append(l2m)
        par_L2_s.append(l2s)
        par_raw_m.append(rm)
        par_raw_s.append(rs)

    # Activation stats: forward hooks
    def _hook(_, __, out):
        h = out.detach().float()
        if h.ndim == 4:  # (B,C,H,W)
            # store raw activations
            # we assume B=1
            h = h.squeeze(0)
            activations.append(h.cpu().numpy())
            # L2 norms by channel
            l2vals = torch.norm(h.flatten(1), p=2, dim=1)
            act_L2_m.append(l2vals.mean().item())
            act_L2_s.append(l2vals.std().item())
            flat = h.flatten()
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())
        else:
            # fallback for shapes not 4D
            flat = h.flatten()
            activations.append(h.cpu().numpy())
            act_L2_m.append(h.norm(2).item())
            act_L2_s.append(0.0)
            act_raw_m.append(flat.mean().item())
            act_raw_s.append(flat.std().item())

    # Register hooks - handle both block and layer formats
    if use_layers:
        hdls = [layer.register_forward_hook(_hook) for layer, layer_type in blocks]
        layer_types = [layer_type for layer, layer_type in blocks]  # Extract layer types for annotations
    else:
        hdls = [b.register_forward_hook(_hook) for b in blocks]
        layer_types = None

    # Single image forward pass
    img = Image.open(IMAGE_PATH).convert("RGB")
    x = _transform(224)(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        _ = model(x)
    for h in hdls:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Outlier listing
    # For each activation map, find top/bottom values
    for a in activations:
        if a.ndim == 3:  # (C,H,W)
            C, H, W = a.shape
            flat = a.reshape(-1)
            if flat.size > 40:
                # pick top/bottom ~20
                idxs = np.argsort(flat)
                sel = np.concatenate([idxs[:20], idxs[-20:]])
            else:
                sel = np.arange(flat.size)
            outs = []
            for i in sel:
                val = float(flat[i])
                c = i // (H*W)
                r = (i % (H*W)) // W
                w = (i % (H*W)) % W
                outs.append((val, r*W + w, c))
            outliers.append(outs)
        else:
            # can't parse easily
            outliers.append([])

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
    }

    # Save to JSON
    safe_label = re.sub(r"[/\\\\]", "__", label)
    out_json = OUT_DIR / f"{safe_label}.json"
    out_json.write_text(json.dumps(stats))
    print(f"  [OK] Stats saved to {out_json}")

    # ───────── PLOTTING ─────────
    def _plot(vals, stds, ttl, ylab, suffix):
        xvals = range(1, len(vals) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        lower = [v - s for v, s in zip(vals, stds)]
        upper = [v + s for v, s in zip(vals, stds)]
        plt.fill_between(xvals, lower, upper, alpha=0.25)
        plt.plot(xvals, vals, marker="o")
        plt.title(ttl, weight="bold")
        plt.xlabel("Block")
        plt.ylabel(ylab)
        plt.grid(ls="--", lw=0.4)
        plt.tight_layout()
        out_png = OUT_DIR / f"{safe_label}_{suffix}.png"
        plt.savefig(out_png)
        plt.close()
        print(f"  [OK] Plot saved to {out_png}")

    # Whisker box plot for raw activations
    def _plot_box(data, out_info):
        """
        data: list of 3D or 2D numpy arrays from each block's activation
        out_info: list of outliers in the form (value, paramIndex, channelIndex)
        """
        if not data:
            return

        # Flatten each block's activations
        block_vals = [d.flatten() for d in data]

        plt.figure(figsize=(12, 3 + 0.5 * len(block_vals)), dpi=300)
        bp = plt.boxplot(
            block_vals,
            vert=True,
            patch_artist=True,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "r",
                "alpha": 0.6,
            },
        )
        # Overplot specific outliers
        for i, outs in enumerate(out_info):
            if not outs:
                continue
            # x coordinate is i+1 for boxplot
            xs = np.full(len(outs), i+1)
            ys = [o[0] for o in outs]
            plt.scatter(xs, ys, color="red", marker="x", s=12)
            # add text if you want (comment out if too cluttered)
            for xpt, ypt, _, ch_id in zip(xs, ys, [o[1] for o in outs], [o[2] for o in outs]):
                plt.text(xpt + 0.2, ypt, f"d{ch_id}", fontsize=6, ha="left")

        plt.xlabel("Block")
        plt.ylabel("Activation values")
        plt.title(f"{label} raw activation distribution", weight="bold")
        plt.grid(True, ls="--", lw=0.4, axis="y")
        plt.tight_layout()
        out_png = OUT_DIR / f"{safe_label}_box.png"
        plt.savefig(out_png)
        plt.close()
        print(f"  [OK] Box plot saved to {out_png}")

    # Generate standard line plots
    _plot(act_L2_m, act_L2_s, f"{label} ||activation||₂", "L2 norm", "l2")
    _plot(act_raw_m, act_raw_s, f"{label} raw activation", "Activation", "raw")
    _plot(par_L2_m, par_L2_s, f"{label} ||params||₂", "Parameter L2", "param_l2")
    _plot(par_raw_m, par_raw_s, f"{label} raw params", "Parameter value", "param_raw")

    # Generate whisker box plot for raw activations with outliers
    _plot_box(activations, outliers)

    # Generate top activations plot (top1, top2, top3, median)
    top_stats = _compute_top_stats(activations)
    _plot_top_activations(top_stats, label, safe_label)

    # Generate layer-wise variants if analyzing layers
    if use_layers:
        # Generate box plot with "_layers" suffix
        def _plot_box_layers(data, out_info, safe_label_layers, layer_types):
            """Box plot variant for layer-wise analysis"""
            if not data:
                return

            # Flatten each layer's activations
            layer_vals = [d.flatten() for d in data]

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
            for i, outs in enumerate(out_info):
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
            out_png = OUT_DIR / f"{safe_label_layers}_box_layers.png"
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Layer-wise box plot saved to {out_png}")

        # Generate top activations plot with "_layers" suffix  
        def _plot_top_activations_layers(top_stats, label, safe_label_layers, layer_types):
            """Top activations plot variant for layer-wise analysis"""
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
            ax.set_title(f'{label} - Layer-wise Top Activation Analysis\n'
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
                              fancybox=True, shadow=True)
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
                    plt.text(i+1, annotation_y, abbrev, 
                            rotation=90, fontsize=4, ha='center', va='top',
                            color='#666666', alpha=0.7, weight='normal')
                
                # Adjust plot limits to accommodate annotations
                ax.set_ylim(annotation_y - (y_max - y_min) * 0.03, y_max)
            
            plt.tight_layout(pad=3.0)
            
            output_path = OUT_DIR / f"{safe_label_layers}_top_layers.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  [OK] Layer-wise top activations plot saved to {output_path}")

        safe_label_layers = re.sub(r"[/\\\\]", "__", label)
        _plot_box_layers(activations, outliers, safe_label_layers, layer_types)
        _plot_top_activations_layers(top_stats, label, safe_label_layers, layer_types)

    return stats


# ───────────────────────── DRIVER ─────────────────────────
if __name__ == "__main__":
    # Ensure test image exists
    _create_test_image_if_needed()
    
    # Example usage: we do "full" depth only. 
    for depth_mode in ("full",):
        print(f"\n=== Depth mode: {depth_mode} ===")
        max_b = None if depth_mode == "full" else 16
        all_stats: Dict[str, Dict[str, List[float]]] = {}
        for net_tag, net_id in MODELS.items():
            try:
                s = analyse(net_tag, net_id, max_blocks=max_b)
                all_stats[net_id] = s
            except Exception as e:
                print(f"  [WARN] Skipped {net_id}: {e}")

        # Combine a line plot across all networks
        if not all_stats:
            print(f"No successful runs for {depth_mode} mode.")
            continue

        # Professional Combined L2 Activation Plot - Similar to train_and_analyze_resnet18_vs_plain18_v2.py
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        # Define professional color scheme
        colors = {
            'resnet18': '#2E86AB',      # Professional blue
            'densenet121': '#A23B72',   # Professional magenta  
            'vgg16_bn': '#F18F01'       # Professional orange
        }
        
        line_styles = {
            'resnet18': '-',      # solid
            'densenet121': '--',  # dashed  
            'vgg16_bn': '-.'      # dash-dot
        }
        
        line_widths = {
            'resnet18': 2.2,
            'densenet121': 2.0,
            'vgg16_bn': 2.4
        }
        
        markers = {
            'resnet18': 'o',
            'densenet121': 's', 
            'vgg16_bn': '^'
        }
        
        # Plot each model with professional styling
        legend_elements = []
        
        for model_id, stats_dict in all_stats.items():
            # Get activation L2 data
            means = np.array(stats_dict["l2_mean"])
            stds = np.array(stats_dict["l2_std"])
            x_vals = np.array(range(1, len(means) + 1))
            
            color = colors.get(model_id, '#333333')
            linestyle = line_styles.get(model_id, '-')
            linewidth = line_widths.get(model_id, 2.0)
            marker = markers.get(model_id, 'o')
            
            # Plot mean line with error band
            line = ax.plot(x_vals, means, 
                          color=color, 
                          linestyle=linestyle, 
                          linewidth=linewidth,
                          label=model_id.upper().replace('_', '-'),
                          marker=marker,
                          markersize=6,
                          markerfacecolor=color,
                          markeredgecolor='white',
                          markeredgewidth=1,
                          alpha=0.9)
            
            # Add error band (mean ± std)
            ax.fill_between(x_vals, means - stds, means + stds, 
                           color=color, alpha=0.15, 
                           linewidth=0)
            
            legend_elements.append(line[0])
        
        # Enhance the plot aesthetics
        ax.set_xlabel('Block Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Activation L2 Norm', fontsize=14, fontweight='bold')
        ax.set_title(f'Per-Block Activation Analysis: CNN Architecture Comparison\n'
                    f'Mode: {depth_mode.upper()} | Error bands show ±1 standard deviation',
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
        max_blocks = max([len(s["l2_mean"]) for s in all_stats.values()])
        ax.set_xticks(range(1, max_blocks + 1))
        
        # Professional legend
        legend = ax.legend(legend_elements, [el.get_label() for el in legend_elements],
                          title='CNN Architectures', loc='best', 
                          fontsize=12, title_fontsize=13, framealpha=0.9,
                          fancybox=True, shadow=True)
        legend.get_title().set_fontweight('bold')
        
        # Add subtle background color
        ax.set_facecolor('#fafafa')
        
        # Tight layout with padding
        plt.tight_layout(pad=2.0)
        
        # Save with high quality
        output_path = OUT_DIR / f"combined_l2_professional_{depth_mode}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  [OK] Professional combined L2 plot saved to {output_path}")

        # Additional professional plots for other metrics
        metrics_config = [
            ("raw_mean", "raw_std", "Raw Activation Value", "raw_activation"),
            ("param_l2_mean", "param_l2_std", "Parameter L2 Norm", "param_l2"),
            ("param_raw_mean", "param_raw_std", "Raw Parameter Value", "param_raw")
        ]
        
        for mean_key, std_key, ylabel, suffix in metrics_config:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
            
            for model_id, stats_dict in all_stats.items():
                means = np.array(stats_dict[mean_key])
                stds = np.array(stats_dict[std_key])
                x_vals = np.array(range(1, len(means) + 1))
                
                color = colors.get(model_id, '#333333')
                linestyle = line_styles.get(model_id, '-')
                linewidth = line_widths.get(model_id, 2.0)
                marker = markers.get(model_id, 'o')
                
                # Plot with error band
                ax.plot(x_vals, means, 
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       label=model_id.upper().replace('_', '-'),
                       marker=marker, markersize=5,
                       markerfacecolor=color, markeredgecolor='white',
                       markeredgewidth=0.8, alpha=0.9)
                
                ax.fill_between(x_vals, means - stds, means + stds, 
                               color=color, alpha=0.12, linewidth=0)
            
            # Styling
            ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'Blockwise {ylabel} Comparison ({depth_mode.upper()})',
                        fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(fontsize=10, framealpha=0.9)
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            out_png = OUT_DIR / f"combined_{suffix}_{depth_mode}.png"
            plt.savefig(out_png, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print(f"  [OK] Combined {suffix} plot saved to {out_png}")

    # ═══════════════════════ LAYER-WISE ANALYSIS ═══════════════════════
    print(f"\n{'='*60}")
    print("STARTING LAYER-WISE ANALYSIS")
    print(f"{'='*60}")
    
    # Run layer-wise analysis for all models (no depth limit for layers)
    print(f"\n=== Layer-wise Analysis Mode ===")
    all_layer_stats: Dict[str, Dict[str, List[float]]] = {}
    
    for net_tag, net_id in MODELS.items():
        try:
            print(f"\n[INFO] Running layer-wise analysis for {net_id}")
            s = analyse(net_tag, net_id, max_blocks=None, use_layers=True)
            all_layer_stats[net_id] = s
        except Exception as e:
            print(f"  [WARN] Skipped layer-wise analysis for {net_id}: {e}")
    
    if all_layer_stats:
        print(f"\n[INFO] Successfully analyzed {len(all_layer_stats)} models at layer level")
        
        # Generate combined layer-wise plots for comparison
        layer_analysis_types = [
            ("l2_mean", "l2_std", "Activation L2 Norm", "l2_layers"),
            ("raw_mean", "raw_std", "Raw Activation Value", "raw_layers")
        ]
        
        for mean_key, std_key, ylabel, suffix in layer_analysis_types:
            fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
            
            # Use the same color scheme as block analysis for consistency
            colors = {
                'resnet18': '#2E86AB',      # Professional blue
                'densenet121': '#A23B72',   # Professional magenta  
                'vgg16_bn': '#F18F01'       # Professional orange
            }
            
            line_styles = {
                'resnet18': '-',      # solid
                'densenet121': '--',  # dashed  
                'vgg16_bn': '-.'      # dash-dot
            }
            
            line_widths = {
                'resnet18': 2.0,  # Slightly thinner for dense layer plots
                'densenet121': 1.8,
                'vgg16_bn': 2.2
            }
            
            markers = {
                'resnet18': 'o',
                'densenet121': 's', 
                'vgg16_bn': '^'
            }
            
            legend_elements = []
            max_layers_count = 0
            
            for model_id, stats_dict in all_layer_stats.items():
                means = np.array(stats_dict[mean_key])
                stds = np.array(stats_dict[std_key])
                x_vals = np.array(range(1, len(means) + 1))
                max_layers_count = max(max_layers_count, len(means))
                
                color = colors.get(model_id, '#333333')
                linestyle = line_styles.get(model_id, '-')
                linewidth = line_widths.get(model_id, 2.0)
                marker = markers.get(model_id, 'o')
                
                # Use fewer markers for dense plots
                markevery = max(1, len(x_vals) // 15) if len(x_vals) > 30 else 1
                
                # Plot with professional styling adapted for layer analysis
                line = ax.plot(x_vals, means, 
                              color=color, linestyle=linestyle, linewidth=linewidth,
                              label=f"{model_id.upper().replace('_', '-')} ({len(means)} layers)",
                              marker=marker, markersize=4,
                              markerfacecolor=color, markeredgecolor='white',
                              markeredgewidth=0.8, alpha=0.9, markevery=markevery)
                
                # Add lighter error band
                ax.fill_between(x_vals, means - stds, means + stds, 
                               color=color, alpha=0.08, linewidth=0)
                
                legend_elements.append(line[0])
            
            # Enhanced styling for layer-wise comparison
            ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold', color='#333333')
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='#333333')
            ax.set_title(f'Layer-wise {ylabel} Comparison Across CNN Architectures\n'
                        f'Individual layer analysis | Error bands show ±1 standard deviation',
                        fontsize=16, fontweight='bold', pad=25, color='#333333')
            
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
            
            # Professional legend
            legend = ax.legend(legend_elements, [el.get_label() for el in legend_elements],
                              title='CNN Architectures (Layer-wise)', loc='best', 
                              fontsize=11, title_fontsize=12, framealpha=0.95,
                              fancybox=True, shadow=True)
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_color('#333333')
            
            ax.set_facecolor('#fafafa')
            
            # Add summary annotation
            textstr = f'Max Layers: {max_layers_count}\nArchitectures: {len(all_layer_stats)}'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, fontfamily='monospace')
            
            plt.tight_layout(pad=3.0)
            
            # Save layer-wise comparison plot
            out_png = OUT_DIR / f"combined_{suffix}_comparison.png"
            plt.savefig(out_png, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            print(f"  [OK] Combined layer-wise {suffix} comparison saved to {out_png}")
    
    else:
        print("[WARN] No successful layer-wise analyses to combine")

    print("\nAnalysis complete.")
