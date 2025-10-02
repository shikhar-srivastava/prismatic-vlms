#!/usr/bin/env python3
"""
create_validation_sets.py

Script to create validation sets for both align and finetune stages by sampling representative
batches from the full datasets. These validation sets are used for tracking metrics like 
average rank during training instead of using single saved batches.

Usage:
    python scripts/create_validation_sets.py --model llava-v15-7b --dataset llava-v15 \
        --align_val_size 100 --finetune_val_size 200 --output_dir validation_sets/
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import draccus

from prismatic.conf import DatasetConfig, ModelConfig
from prismatic.conf.datasets import DatasetRegistry
from prismatic.conf.models import ModelRegistry
from prismatic.models.materialize import get_vision_backbone_and_transform, get_llm_backbone_and_tokenizer
from prismatic.preprocessing.materialize import get_dataset_and_collator
from prismatic.util.torch_utils import set_global_seed
from prismatic.overwatch import initialize_overwatch

# Initialize overwatch
overwatch = initialize_overwatch(__name__)


def create_stratified_sample(dataset, val_size: int, seed: int = 42) -> List[int]:
    """
    Create a stratified sample of indices from the dataset to ensure representation
    across different types of examples (if applicable).
    
    For now, we use simple random sampling, but this can be extended to be more sophisticated.
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices[:val_size]


def save_validation_batch(dataloader: DataLoader, indices: List[int], output_path: Path, stage: str):
    """
    Save validation batches to disk for later use during training.
    """
    validation_batches = []
    
    # Create subset dataset with the selected indices
    subset_dataset = Subset(dataloader.dataset, indices)
    subset_dataloader = DataLoader(
        subset_dataset, 
        batch_size=dataloader.batch_size,
        collate_fn=dataloader.collate_fn,
        shuffle=False,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )
    
    overwatch.info(f"Collecting {len(indices)} samples for {stage} validation set...")
    
    for batch_idx, batch in enumerate(subset_dataloader):
        # Convert tensors to CPU and detach to save memory
        cpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cpu_batch[key] = value.detach().cpu()
            else:
                cpu_batch[key] = value
        
        validation_batches.append(cpu_batch)
        
        if (batch_idx + 1) % 10 == 0:
            overwatch.info(f"  Processed {batch_idx + 1} batches...")
    
    # Save the validation batches
    torch.save(validation_batches, output_path)
    overwatch.info(f"Saved {len(validation_batches)} validation batches to {output_path}")
    
    return len(validation_batches)


def create_validation_sets(
    model_id: str,
    dataset_id: str,
    align_val_size: int,
    finetune_val_size: int,
    output_dir: Path,
    seed: int = 42,
    hf_token: str = None,
    llm_checkpoint_path: str = None
):
    """
    Create validation sets for both align and finetune stages.
    """
    overwatch.info(f"Creating validation sets for model: {model_id}, dataset: {dataset_id}")
    
    # Set seed for reproducibility
    set_global_seed(seed)
    
    # Get model and dataset configs
    model_cfg = ModelConfig.get_choice_class(model_id)()
    dataset_cfg = DatasetConfig.get_choice_class(dataset_id)()
    
    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vision backbone and image transform
    overwatch.info(f"Loading vision backbone: {model_cfg.vision_backbone_id}")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        image_resize_strategy=model_cfg.image_resize_strategy
    )
    
    # Load LLM backbone and tokenizer
    overwatch.info(f"Loading LLM backbone: {model_cfg.llm_backbone_id}")
    
    # Build config dict to pass checkpoint path override if provided
    cfg_dict = {}
    if llm_checkpoint_path is not None:
        cfg_dict['llm_checkpoint_path'] = llm_checkpoint_path
        overwatch.info(f"Using LLM checkpoint path override: {llm_checkpoint_path}")
    
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        cfg=cfg_dict if cfg_dict else None
    )
    
    # Create validation sets for both stages
    validation_info = {}
    
    for stage, val_size in [("align", align_val_size), ("finetune", finetune_val_size)]:
        if val_size <= 0:
            overwatch.info(f"Skipping {stage} validation set (size = {val_size})")
            continue
            
        overwatch.info(f"Creating {stage} validation set with {val_size} samples")
        
        # Get dataset and collator for this stage
        dataset, collator = get_dataset_and_collator(
            stage=stage,
            dataset_cfg=dataset_cfg,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            padding_side=tokenizer.padding_side
        )
        
        overwatch.info(f"Full {stage} dataset size: {len(dataset)}")
        
        # Sample validation indices
        val_indices = create_stratified_sample(dataset, val_size, seed)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=min(8, val_size),  # Use reasonable batch size
            collate_fn=collator,
            shuffle=False,
            num_workers=0
        )
        
        # Save validation batches
        output_path = output_dir / f"{stage}_validation.pt"
        num_batches = save_validation_batch(dataloader, val_indices, output_path, stage)
        
        # Store metadata
        validation_info[stage] = {
            "num_samples": val_size,
            "num_batches": num_batches,
            "indices": val_indices,
            "output_path": str(output_path),
            "dataset_size": len(dataset)
        }
    
    # Save metadata
    metadata_path = output_dir / "validation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "seed": seed,
            "validation_info": validation_info
        }, f, indent=2)
    
    overwatch.info(f"Validation set creation complete. Metadata saved to {metadata_path}")
    return validation_info


def main():
    parser = argparse.ArgumentParser(description="Create validation sets for Prismatic VLM training")
    parser.add_argument("--model", required=True, help="Model ID (e.g., llava-v15-7b)")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., llava-v15)")
    parser.add_argument("--align_val_size", type=int, default=100, 
                       help="Number of samples for align validation set")
    parser.add_argument("--finetune_val_size", type=int, default=200,
                       help="Number of samples for finetune validation set")
    parser.add_argument("--output_dir", type=Path, default=Path("validation_sets"),
                       help="Output directory for validation sets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token (if needed)")
    parser.add_argument("--llm_checkpoint_path", type=str, default=None,
                       help="Override LLM checkpoint path (for custom models)")
    
    args = parser.parse_args()
    
    # Validate model and dataset IDs
    try:
        ModelConfig.get_choice_class(args.model)
    except KeyError:
        available_models = [model.value.model_id for model in ModelRegistry]
        raise ValueError(f"Invalid model ID '{args.model}'. Available: {available_models}")
    
    try:
        DatasetConfig.get_choice_class(args.dataset)
    except KeyError:
        available_datasets = [dataset.value.dataset_id for dataset in DatasetRegistry]
        raise ValueError(f"Invalid dataset ID '{args.dataset}'. Available: {available_datasets}")
    
    # Create validation sets
    validation_info = create_validation_sets(
        model_id=args.model,
        dataset_id=args.dataset,
        align_val_size=args.align_val_size,
        finetune_val_size=args.finetune_val_size,
        output_dir=args.output_dir,
        seed=args.seed,
        hf_token=args.hf_token,
        llm_checkpoint_path=args.llm_checkpoint_path
    )
    
    print("=" * 60)
    print("VALIDATION SET CREATION SUMMARY")
    print("=" * 60)
    for stage, info in validation_info.items():
        print(f"{stage.upper()} VALIDATION:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Dataset Coverage: {info['num_samples']}/{info['dataset_size']} "
              f"({100 * info['num_samples'] / info['dataset_size']:.1f}%)")
        print(f"  Output: {info['output_path']}")
        print()


if __name__ == "__main__":
    main() 