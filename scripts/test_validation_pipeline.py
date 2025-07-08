#!/usr/bin/env python3
"""
test_validation_pipeline.py

Test script to validate that the validation pipeline works correctly.
This script creates small validation sets and tests the loading mechanism.
"""

import argparse
import tempfile
from pathlib import Path

import torch

from prismatic.preprocessing.validation import load_validation_manager
from scripts.create_validation_sets import create_validation_sets
from prismatic.overwatch import initialize_overwatch

# Initialize overwatch
overwatch = initialize_overwatch(__name__)


def test_validation_creation_and_loading():
    """Test the complete validation pipeline."""
    
    overwatch.info("Testing validation set creation and loading pipeline...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        overwatch.info(f"Using temporary directory: {temp_path}")
        
        # Test 1: Create validation sets with small sizes
        overwatch.info("Step 1: Creating small validation sets...")
        try:
            validation_info = create_validation_sets(
                model_id="reproduction-llava-v15+7b",
                dataset_id="llava-v15", 
                align_val_size=10,  # Small for testing
                finetune_val_size=15,  # Small for testing
                output_dir=temp_path,
                seed=42
            )
            overwatch.info("✓ Validation set creation successful")
        except Exception as e:
            overwatch.error(f"✗ Validation set creation failed: {e}")
            return False
        
        # Test 2: Load validation manager
        overwatch.info("Step 2: Loading validation manager...")
        try:
            validation_manager = load_validation_manager(str(temp_path), device_id=0)
            if validation_manager is None:
                overwatch.error("✗ Failed to load validation manager")
                return False
            overwatch.info("✓ Validation manager loading successful")
        except Exception as e:
            overwatch.error(f"✗ Validation manager loading failed: {e}")
            return False
        
        # Test 3: Verify validation sets
        overwatch.info("Step 3: Verifying validation sets...")
        try:
            for stage in ["align", "finetune"]:
                if validation_manager.has_validation_set(stage):
                    val_info = validation_manager.get_validation_info(stage)
                    overwatch.info(f"  {stage}: {val_info['num_samples']} samples, {val_info['num_batches']} batches")
                    
                    # Test getting a single batch
                    batch = validation_manager.get_validation_batch(stage, 0)
                    if batch is None:
                        overwatch.error(f"✗ Failed to get {stage} validation batch")
                        return False
                    
                    # Verify batch contents
                    required_keys = ['input_ids', 'attention_mask', 'labels']
                    for key in required_keys:
                        if key not in batch:
                            overwatch.error(f"✗ Missing key '{key}' in {stage} batch")
                            return False
                        if not isinstance(batch[key], torch.Tensor):
                            overwatch.error(f"✗ Key '{key}' is not a tensor in {stage} batch")
                            return False
                    
                    overwatch.info(f"  ✓ {stage} validation set verified")
                else:
                    overwatch.info(f"  No {stage} validation set found (expected for small test)")
            
            overwatch.info("✓ Validation set verification successful")
        except Exception as e:
            overwatch.error(f"✗ Validation set verification failed: {e}")
            return False
        
        # Test 4: Compatibility check
        overwatch.info("Step 4: Testing compatibility check...")
        try:
            is_compatible = validation_manager.validate_compatibility("reproduction-llava-v15+7b", "llava-v15")
            if not is_compatible:
                overwatch.error("✗ Compatibility check failed")
                return False
            overwatch.info("✓ Compatibility check successful")
        except Exception as e:
            overwatch.error(f"✗ Compatibility check failed: {e}")
            return False
        
        overwatch.info("🎉 All validation pipeline tests passed!")
        return True


def test_validation_manager_edge_cases():
    """Test edge cases for validation manager."""
    
    overwatch.info("Testing validation manager edge cases...")
    
    # Test loading from non-existent directory
    try:
        manager = load_validation_manager("/non/existent/path", device_id=0)
        if manager is not None:
            overwatch.error("✗ Expected None for non-existent path")
            return False
        overwatch.info("✓ Handled non-existent path correctly")
    except Exception as e:
        overwatch.error(f"✗ Exception handling non-existent path: {e}")
        return False
    
    overwatch.info("✓ Edge case tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test validation pipeline")
    parser.add_argument("--gpu", action="store_true", help="Test with GPU if available")
    args = parser.parse_args()
    
    # Set device
    if args.gpu and torch.cuda.is_available():
        overwatch.info("Testing with GPU")
        device_id = 0
    else:
        overwatch.info("Testing with CPU only")
        device_id = 0  # CPU fallback
    
    print("=" * 60)
    print("VALIDATION PIPELINE TEST")
    print("=" * 60)
    
    success = True
    
    # Run main pipeline test
    if not test_validation_creation_and_loading():
        success = False
    
    print()
    
    # Run edge case tests
    if not test_validation_manager_edge_cases():
        success = False
    
    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The validation pipeline is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 