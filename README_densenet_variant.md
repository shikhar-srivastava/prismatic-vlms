# DenseNet vs ResNet vs Plain Network Comparison

## Overview

This is a variant of the `train_and_analyze_resnet18_vs_plain18_v2.py` script with the suffix `v_dense` that adds **DenseNet models** to compare alongside ResNet and Plain models. The script now trains and analyzes **nine networks** total: ResNet-14/18/34, Plain-14/18/34, and DenseNet-14/18/34.

## Key Files

- **`train_and_analyze_resnet18_vs_plain18_v_dense.py`**: Main training and analysis script
- **Output Directory**: `viz/plots/act_analysis_rvd/` (ResNet vs Dense)
- **Checkpoints**: `checkpoints/` directory

## DenseNet Model Details

### Architecture Overview

All DenseNet models follow the **DenseNet-BC** (Bottleneck-Compression) design with the following key components:

1. **Identical Stem**: Same 7×7 conv + maxpool as ResNet/Plain for fair comparison
2. **Dense Blocks**: Feature concatenation instead of addition
3. **Transition Layers**: Compression and spatial reduction between dense blocks
4. **Bottleneck Design**: 1×1 conv followed by 3×3 conv in each dense layer

### Model Specifications

#### DenseNet-14
- **Block Config**: (3, 4, 4, 3) = **14 dense layers total**
- **Growth Rate**: 64 (optimized for parameter matching)
- **Compression**: 0.8 (reduced to keep more parameters)
- **Parameters**: ~4.0M (vs ResNet-14: ~11.3M)
- **Design Goal**: Comparable depth to ResNet-14

#### DenseNet-18  
- **Block Config**: (4, 4, 4, 6) = **18 dense layers total**
- **Growth Rate**: 64 (consistent across models)
- **Compression**: 0.8 (reduced compression)
- **Parameters**: ~5.5M (vs ResNet-18: ~11.2M)
- **Design Goal**: Comparable depth to ResNet-18 with balanced layer distribution

#### DenseNet-34
- **Block Config**: (6, 8, 12, 8) = **34 dense layers total**
- **Growth Rate**: 64 (maintained for consistency)
- **Compression**: 0.75 (balanced for parameter control)
- **Parameters**: ~14.6M (vs ResNet-34: ~21.3M)
- **Design Goal**: Comparable depth to ResNet-34 with deeper dense blocks

## Key Architectural Assumptions & Design Decisions

### 1. **Stem Layer Consistency**
All models use identical 7×7 conv + maxpool stem for fair comparison. This ensures the initial feature extraction is the same across all architectures.

### 2. **Feature Combination Strategy**
- **ResNet**: Addition of features (`x + F(x)`)
- **Plain**: Sequential transformation (`F(x)`)
- **DenseNet**: Concatenation of all previous features (`[x0, x1, x2, ..., xn]`)

### 3. **Spatial Reduction**
- **ResNet/Plain**: Stride-2 convolutions in residual blocks
- **DenseNet**: 2×2 average pooling in transition layers

### 4. **Parameter Efficiency**
DenseNet models are inherently more parameter-efficient due to feature reuse. Even with high growth rates (64), they achieve similar performance with fewer parameters.

### 5. **Compression Strategy**
- **Standard DenseNet**: 0.5 compression ratio
- **Our Models**: 0.75-0.8 compression to balance parameter count vs performance

### 6. **Bottleneck Design**
All DenseNet layers use 4× growth_rate bottleneck width (DenseNet-BC standard):
- 1×1 conv: `input_features → 4 × growth_rate`
- 3×3 conv: `4 × growth_rate → growth_rate`

## Training Configuration

### Common Settings
- **Learning Schedule**: Same across all architectures (multi-step LR)
- **Data Augmentation**: Identical preprocessing and augmentation
- **Optimization**: SGD + momentum (0.9) + weight decay (5e-4)
- **Mixed Precision**: PyTorch AMP for efficiency
- **Batch Size**: Default 128 (adjustable)

### Depth-Specific Schedules
- **14-layer models**: 100 epochs, LR steps at [50, 75]
- **18-layer models**: 150 epochs, LR steps at [75, 110] 
- **34-layer models**: 200 epochs, LR steps at [100, 150]

## Usage Examples

### Train Only DenseNet Models
```bash
python train_and_analyze_resnet18_vs_plain18_v_dense.py --train_dense_only
```

### Full Comparison (Use Existing ResNet/Plain Checkpoints)
```bash
python train_and_analyze_resnet18_vs_plain18_v_dense.py
```

### Skip Training, Only Analysis
```bash
python train_and_analyze_resnet18_vs_plain18_v_dense.py --no_train
```

### Custom Dataset and Batch Size
```bash
python train_and_analyze_resnet18_vs_plain18_v_dense.py --dataset cifar100 --batch 64
```

## Analysis Features

### 1. **Per-Block Activation Analysis**
- L2 norm statistics across blocks
- Raw activation distributions
- Whisker box plots with outlier annotations

### 2. **Parameter Analysis**
- Parameter L2 norms per block
- Distribution of parameter values
- Parameter efficiency comparisons

### 3. **Training Curve Visualization**
- **Individual Model Plots**: Loss, accuracy, and learning rate curves for each model
- **Comparative Plots**: Side-by-side validation accuracy comparison across all architectures
- **Summary Tables**: Best/final accuracies and losses with visual formatting
- **Automatic Annotations**: Best validation accuracy/loss points highlighted
- **High-Quality Output**: 300 DPI plots with professional styling

### 4. **Professional Visualizations**
- Three-way comparison plots (ResNet vs Plain vs DenseNet)
- Error bands showing ±1 standard deviation
- Color-coded by architecture type:
  - **Blue**: ResNet models
  - **Magenta**: Plain models  
  - **Orange**: DenseNet models
- Organized legends by model type and depth

### 5. **Individual Model Plots**
- Per-model activation and parameter analysis
- Detailed whisker box plots for each architecture
- JSON statistics for quantitative analysis

## Expected Outcomes

### Parameter Efficiency
DenseNet models should achieve competitive performance with significantly fewer parameters due to:
- Feature reuse through concatenation
- Efficient information flow
- Reduced redundancy

### Gradient Flow
DenseNet models should show:
- Better gradient flow (similar to ResNet but different mechanism)
- More stable training compared to Plain networks
- Potentially better convergence properties

### Activation Patterns
Expected differences in activation analysis:
- **ResNet**: Skip connections may show activation rescaling
- **Plain**: Potential degradation in deeper layers
- **DenseNet**: Growing feature maps with dense connectivity

## Technical Implementation Notes

### DenseNet Block Extraction for Analysis
The analysis extracts blocks differently for each architecture:
- **ResNet/Plain**: Individual BasicBlocks/PlainBlocks
- **DenseNet**: Both entire DenseBlocks and individual DenseLayers + Transition layers

### Memory Considerations
DenseNet models can be memory-intensive due to feature concatenation. The implementation includes:
- Mixed precision training for efficiency
- Reasonable batch sizes for GPU memory
- Efficient forward hook management

### Checkpoint Compatibility
The script reuses existing ResNet/Plain checkpoints from the original v2 version, only training new DenseNet models when needed.

## File Structure

```
prismatic-vlms/
├── train_and_analyze_resnet18_vs_plain18_v_dense.py  # Main script
├── checkpoints/                                      # Model checkpoints
│   ├── cifar100_resnet14.pth                        # Existing from v2
│   ├── cifar100_resnet18.pth                        # Existing from v2
│   ├── cifar100_resnet34.pth                        # Existing from v2
│   ├── cifar100_plain14.pth                         # Existing from v2
│   ├── cifar100_plain18.pth                         # Existing from v2
│   ├── cifar100_plain34.pth                         # Existing from v2
│   ├── cifar100_densenet14.pth                      # New DenseNet models
│   ├── cifar100_densenet18.pth                      # New DenseNet models
│   └── cifar100_densenet34.pth                      # New DenseNet models
├── viz/plots/act_analysis_rvd/                       # Analysis outputs
│   ├── combined_l2_rvd_professional_cifar100.png    # Main comparison plot
│   ├── cifar100_densenet14_box.png                  # Individual analyses
│   ├── cifar100_densenet18_box.png
│   ├── cifar100_densenet34_box.png
│   ├── training_curves/                              # Training visualization outputs
│   │   ├── comparative_validation_accuracy_cifar100.png  # Comparative training plots
│   │   ├── training_summary_table_cifar100.png          # Results summary table
│   │   ├── training_summary_cifar100.json               # Results summary data
│   │   ├── cifar100_densenet14_training_curves.png      # Individual training curves
│   │   ├── cifar100_densenet14_loss.png                 # High-quality loss plots
│   │   ├── cifar100_densenet14_accuracy.png             # High-quality accuracy plots
│   │   ├── cifar100_densenet14_learning_rate.png        # Learning rate schedules
│   │   ├── cifar100_densenet14_metrics.json             # Training metrics data
│   │   └── ... (similar files for all other models)     # All model variations
│   └── *.json                                        # Quantitative statistics
└── README_densenet_variant.md                       # This documentation
```

## Comparison to Standard DenseNet

Our custom DenseNet models differ from torchvision's standard models:
- **Matched Depths**: Exact layer counts to match ResNet (14/18/34)
- **Tuned Growth Rates**: Higher growth rates (64 vs standard 32) for parameter matching
- **Custom Compression**: Adjusted compression ratios for fair comparison
- **Same Stem**: Identical preprocessing for all architectures

This design ensures a fair three-way comparison between residual connections (ResNet), no connections (Plain), and dense connections (DenseNet) at equivalent depths and similar computational budgets. 