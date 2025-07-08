# Validation Set System for Prismatic VLMs

This document explains the new validation set system that replaces single-batch validation with proper validation sets for more robust rank tracking and metrics computation.

## Overview

The validation system provides:
- **Robust validation**: Uses larger validation sets instead of single saved batches
- **Stratified sampling**: Representative samples from both align and finetune datasets
- **Efficient loading**: Pre-computed validation sets loaded once during training
- **Comprehensive metrics**: Average, min, max ranks across validation sets
- **Backward compatibility**: Falls back to legacy single-batch validation if needed

## Quick Start

### 1. Create Validation Sets

First, create validation sets for your model and dataset:

```bash
python scripts/create_validation_sets.py \
  --model llava-v15-7b \
  --dataset llava-v15 \
  --align_val_size 100 \
  --finetune_val_size 200 \
  --output_dir validation_sets/llava-v15-7b/ \
  --seed 42
```

**Parameters:**
- `--model`: Model ID (e.g., `llava-v15-7b`, `llava-v15-13b`)
- `--dataset`: Dataset ID (e.g., `llava-v15`, `llava-v1`)
- `--align_val_size`: Number of samples for align validation (recommended: 50-200)
- `--finetune_val_size`: Number of samples for finetune validation (recommended: 100-500)
- `--output_dir`: Directory to save validation sets
- `--seed`: Random seed for reproducibility

### 2. Enable Validation Tracking in Training

Add these flags to your training command:

```bash
python scripts/pretrain.py \
  --enable_validation_tracking true \
  --validation_set_dir validation_sets/llava-v15-7b/ \
  --validation_frequency 100 \
  # ... other training parameters
```

**New Parameters:**
- `--enable_validation_tracking`: Enable the new validation system
- `--validation_set_dir`: Path to directory containing validation sets
- `--validation_frequency`: How often to run validation (in training steps)

## Detailed Usage

### Creating Validation Sets

The validation creation script supports various options:

```bash
# Basic usage
python scripts/create_validation_sets.py --model llava-v15-7b --dataset llava-v15

# Custom sizes and output location
python scripts/create_validation_sets.py \
  --model llava-v15-7b \
  --dataset llava-v15 \
  --align_val_size 150 \
  --finetune_val_size 300 \
  --output_dir /path/to/validation_sets/ \
  --seed 42

# Skip align validation (finetune only)
python scripts/create_validation_sets.py \
  --model llava-v15-7b \
  --dataset llava-v15 \
  --align_val_size 0 \
  --finetune_val_size 200
```

### Output Structure

The script creates the following structure:

```
validation_sets/
├── align_validation.pt              # Align stage validation batches
├── finetune_validation.pt           # Finetune stage validation batches
└── validation_metadata.json         # Metadata and configuration
```

### Validation Metrics

The new system logs the following metrics:

**Per Stage (align/finetune):**
- `avg_rank_{stage}`: Average rank across all validation samples
- `min_rank_{stage}`: Best (lowest) rank in validation set
- `max_rank_{stage}`: Worst (highest) rank in validation set
- `avg_min_rank_{stage}`: Average of best ranks per batch
- `avg_max_rank_{stage}`: Average of worst ranks per batch

## Integration with Training

### PretrainConfig Parameters

Add these to your training configuration:

```python
@dataclass
class PretrainConfig:
    # ... existing parameters ...
    
    # Validation Set Configuration
    validation_set_dir: Optional[str] = None        # Path to validation sets
    enable_validation_tracking: bool = False        # Enable validation tracking
    validation_frequency: int = 100                 # Validation frequency (steps)
```

### Backward Compatibility

The system maintains backward compatibility:

1. **Legacy mode**: If `enable_validation_tracking=False` and `track_avg_rank=True`, uses old single-batch validation
2. **New mode**: If `enable_validation_tracking=True`, uses validation sets
3. **Disabled**: If both are `False`, no validation tracking

## Best Practices

### Validation Set Sizes

**Recommended sizes based on dataset size:**

| Dataset Size | Align Val Size | Finetune Val Size |
|-------------|----------------|-------------------|
| < 100K      | 50-100         | 100-200          |
| 100K-500K   | 100-200        | 200-400          |
| > 500K      | 200-500        | 400-800          |

### Validation Frequency

**Recommended frequencies:**

- **Development**: Every 50-100 steps for detailed tracking
- **Production**: Every 200-500 steps to reduce overhead
- **Large models**: Every 500-1000 steps to minimize impact

### Storage Considerations

Validation sets require storage space:
- **Align validation**: ~2-10 MB per 100 samples
- **Finetune validation**: ~5-20 MB per 100 samples
- **Total**: Usually < 100 MB for reasonable validation sizes

## Testing

Test the validation pipeline:

```bash
# Run comprehensive tests
python scripts/test_validation_pipeline.py

# Test with GPU (if available)
python scripts/test_validation_pipeline.py --gpu
```

## Migration from Legacy System

### Step 1: Create Validation Sets

```bash
python scripts/create_validation_sets.py \
  --model <your_model> \
  --dataset <your_dataset> \
  --align_val_size 100 \
  --finetune_val_size 200 \
  --output_dir validation_sets/<your_model>/
```

### Step 2: Update Training Command

**Before (legacy):**
```bash
python scripts/pretrain.py \
  --track_avg_rank true \
  # ... other parameters
```

**After (new system):**
```bash
python scripts/pretrain.py \
  --enable_validation_tracking true \
  --validation_set_dir validation_sets/<your_model>/ \
  --validation_frequency 100 \
  # ... other parameters
```

### Step 3: Verify Metrics

Check that the new validation metrics appear in your logs:
- `avg_rank_align`, `avg_rank_finetune`
- `min_rank_align`, `min_rank_finetune`
- `max_rank_align`, `max_rank_finetune`

## Troubleshooting

### Common Issues

**1. "Validation metadata not found"**
- Ensure you ran `create_validation_sets.py` first
- Check the path in `--validation_set_dir`

**2. "Failed to load validation sets"**
- Verify the model/dataset compatibility
- Check file permissions in validation directory

**3. "No validation set found for stage"**
- Check if you set `--align_val_size 0` or `--finetune_val_size 0`
- Verify the stage exists in your training

### Performance Impact

The validation system has minimal performance impact:
- **Memory**: ~10-50 MB additional memory usage
- **Compute**: ~0.5-2% overhead during validation steps
- **Storage**: ~10-100 MB for validation files

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger('prismatic.preprocessing.validation').setLevel(logging.DEBUG)
```

## Advanced Usage

### Custom Validation Sampling

For advanced use cases, you can modify the sampling strategy in `create_validation_sets.py`:

```python
def create_stratified_sample(dataset, val_size: int, seed: int = 42) -> List[int]:
    """
    Custom sampling strategy - modify this function for specific needs
    """
    # Your custom sampling logic here
    pass
```

### Multiple Validation Sets

Create validation sets for different models/datasets:

```bash
# For different models
python scripts/create_validation_sets.py --model llava-v15-7b --dataset llava-v15 --output_dir validation_sets/7b/
python scripts/create_validation_sets.py --model llava-v15-13b --dataset llava-v15 --output_dir validation_sets/13b/

# For different datasets
python scripts/create_validation_sets.py --model llava-v15-7b --dataset llava-v15 --output_dir validation_sets/v15/
python scripts/create_validation_sets.py --model llava-v15-7b --dataset llava-v1 --output_dir validation_sets/v1/
```

## FAQ

**Q: How does this improve over single-batch validation?**
A: The new system provides more robust metrics by evaluating on multiple representative samples, reducing variance and providing better insights into model performance.

**Q: Can I use different validation sets for different experiments?**
A: Yes, create separate validation directories and specify the appropriate `--validation_set_dir` for each experiment.

**Q: What happens if validation fails during training?**
A: The system gracefully handles errors and continues training. Failed validation attempts are logged but don't stop training.

**Q: Can I inspect the validation data?**
A: Yes, validation sets are saved as PyTorch files that can be loaded and inspected manually for debugging.

**Q: How do I update validation sets when my dataset changes?**
A: Re-run `create_validation_sets.py` with the same parameters to regenerate validation sets with the updated dataset. 