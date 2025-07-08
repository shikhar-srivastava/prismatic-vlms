"""
validation.py

Validation utilities for loading and managing pre-created validation sets during training.
These are used for metrics like average rank tracking instead of single saved batches.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from prismatic.overwatch import initialize_overwatch

# Initialize overwatch
overwatch = initialize_overwatch(__name__)


class ValidationDataset(Dataset):
    """
    A dataset wrapper that loads pre-created validation batches from disk.
    """
    
    def __init__(self, validation_batches: List[Dict]):
        """
        Initialize with a list of pre-created validation batches.
        
        Args:
            validation_batches: List of batches, where each batch is a dict with keys like
                              'input_ids', 'attention_mask', 'pixel_values', 'labels', etc.
        """
        self.validation_batches = validation_batches
        
    def __len__(self) -> int:
        return len(self.validation_batches)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a validation batch. Since batches are pre-created, we return the batch directly.
        """
        return self.validation_batches[idx]


class ValidationSetManager:
    """
    Manager class for loading and providing access to validation sets.
    """
    
    def __init__(self, validation_set_dir: str, device_id: int = 0):
        """
        Initialize the validation set manager.
        
        Args:
            validation_set_dir: Directory containing validation sets and metadata
            device_id: CUDA device ID to move tensors to
        """
        self.validation_set_dir = Path(validation_set_dir)
        self.device_id = device_id
        self.metadata = None
        self.validation_datasets = {}
        
        # Load metadata
        self._load_metadata()
        
        # Load validation sets
        self._load_validation_sets()
    
    def _load_metadata(self):
        """Load validation set metadata."""
        metadata_path = self.validation_set_dir / "validation_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Validation metadata not found at {metadata_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        overwatch.info(f"Loaded validation metadata for model: {self.metadata['model_id']}, "
                      f"dataset: {self.metadata['dataset_id']}")
    
    def _load_validation_sets(self):
        """Load validation sets from disk."""
        for stage in ["align", "finetune"]:
            stage_info = self.metadata["validation_info"].get(stage)
            if stage_info is None:
                overwatch.info(f"No validation set found for stage: {stage}")
                continue
            
            validation_path = Path(stage_info["output_path"])
            
            # If path is relative, resolve it relative to validation_set_dir
            if not validation_path.is_absolute():
                validation_path = self.validation_set_dir / validation_path.name
            
            if not validation_path.exists():
                overwatch.error(f"Validation file not found: {validation_path}")
                continue
            
            # Load validation batches
            overwatch.info(f"Loading {stage} validation set from {validation_path}")
            validation_batches = torch.load(validation_path, map_location="cpu")
            
            # Move tensors to the specified device
            device_batches = []
            for batch in validation_batches:
                device_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        device_batch[key] = value.to(f"cuda:{self.device_id}")
                    else:
                        device_batch[key] = value
                device_batches.append(device_batch)
            
            # Create validation dataset
            self.validation_datasets[stage] = ValidationDataset(device_batches)
            
            overwatch.info(f"Loaded {len(device_batches)} validation batches for {stage} stage")
    
    def get_validation_dataset(self, stage: str) -> Optional[ValidationDataset]:
        """
        Get the validation dataset for a specific stage.
        
        Args:
            stage: Training stage ("align" or "finetune")
            
        Returns:
            ValidationDataset or None if not available
        """
        return self.validation_datasets.get(stage)
    
    def get_validation_batch(self, stage: str, batch_idx: int = 0) -> Optional[Dict]:
        """
        Get a specific validation batch for a stage.
        
        Args:
            stage: Training stage ("align" or "finetune")
            batch_idx: Index of the batch to retrieve
            
        Returns:
            Validation batch dict or None if not available
        """
        dataset = self.validation_datasets.get(stage)
        if dataset is None or batch_idx >= len(dataset):
            return None
        
        return dataset[batch_idx]
    
    def get_all_validation_batches(self, stage: str) -> Optional[List[Dict]]:
        """
        Get all validation batches for a stage.
        
        Args:
            stage: Training stage ("align" or "finetune")
            
        Returns:
            List of validation batch dicts or None if not available
        """
        dataset = self.validation_datasets.get(stage)
        if dataset is None:
            return None
        
        return [dataset[i] for i in range(len(dataset))]
    
    def has_validation_set(self, stage: str) -> bool:
        """Check if a validation set exists for the given stage."""
        return stage in self.validation_datasets
    
    def get_validation_info(self, stage: str) -> Optional[Dict]:
        """Get metadata information about a validation set."""
        if self.metadata is None:
            return None
        return self.metadata["validation_info"].get(stage)
    
    def validate_compatibility(self, model_id: str, dataset_id: str) -> bool:
        """
        Validate that the loaded validation sets are compatible with the current training setup.
        
        Args:
            model_id: Current model ID
            dataset_id: Current dataset ID
            
        Returns:
            True if compatible, False otherwise
        """
        if self.metadata is None:
            return False
        
        return (self.metadata["model_id"] == model_id and 
                self.metadata["dataset_id"] == dataset_id)


def load_validation_manager(validation_set_dir: str, device_id: int = 0) -> Optional[ValidationSetManager]:
    """
    Convenience function to load a validation set manager.
    
    Args:
        validation_set_dir: Directory containing validation sets
        device_id: CUDA device ID
        
    Returns:
        ValidationSetManager instance or None if loading fails
    """
    try:
        return ValidationSetManager(validation_set_dir, device_id)
    except Exception as e:
        overwatch.error(f"Failed to load validation sets: {e}")
        return None 