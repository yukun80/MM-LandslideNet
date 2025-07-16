#!/usr/bin/env python3
"""
Test script to verify MM-InternImage-TNF implementation.

This script performs basic functionality tests to ensure all components work correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    from mm_intern_image_src.config import config

    print(f"✓ Device: {config.DEVICE}")
    print(f"✓ Model name: {config.MODEL_NAME}")
    print(f"✓ Batch size: {config.BATCH_SIZE}")
    print(f"✓ Data directory: {config.DATA_DIR}")
    print(f"✓ Optical channels: {config.OPTICAL_CHANNELS}")
    print(f"✓ SAR channels: {config.SAR_CHANNELS}")
    print(f"✓ SAR diff channels: {config.SAR_DIFF_CHANNELS}")


def test_model_creation():
    """Test model creation and forward pass."""
    print("\nTesting model creation...")
    from mm_intern_image_src.models import create_model

    try:
        # Create model (this might fail if InternImage is not properly set up)
        model = create_model(device="cpu")  # Use CPU for testing
        print("✓ Model created successfully")

        # Test forward pass with dummy data
        batch_size = 2
        test_batch = {
            "optical": torch.randn(batch_size, 5, 64, 64),
            "sar": torch.randn(batch_size, 4, 64, 64),
            "sar_diff": torch.randn(batch_size, 4, 64, 64),
        }

        with torch.no_grad():
            output = model(test_batch)
            print(f"✓ Forward pass successful, output shape: {output.shape}")

            # Test feature extraction
            features = model.get_feature_maps(test_batch)
            print(f"✓ Feature extraction successful")
            for name, feat in features.items():
                print(f"  - {name}: {feat.shape}")

    except Exception as e:
        print(f"✗ Model creation/testing failed: {e}")
        print("  This might be due to InternImage not being properly compiled")
        return False

    return True


def test_loss_functions():
    """Test loss functions."""
    print("\nTesting loss functions...")
    from mm_intern_image_src.utils import FocalLoss, DiceLoss, CombinedLoss

    # Create dummy data
    logits = torch.randn(10, 1)
    labels = torch.randint(0, 2, (10,)).float()

    # Test individual loss functions
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()

    focal_output = focal_loss(logits.squeeze(), labels)
    dice_output = dice_loss(logits.squeeze(), labels)
    combined_output = combined_loss(logits.squeeze(), labels)

    print(f"✓ Focal loss: {focal_output.item():.4f}")
    print(f"✓ Dice loss: {dice_output.item():.4f}")
    print(f"✓ Combined loss: {combined_output['total_loss'].item():.4f}")


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    from mm_intern_image_src.utils import calculate_metrics

    # Create dummy predictions and labels
    y_pred = torch.rand(100)  # Random probabilities
    y_true = torch.randint(0, 2, (100,)).float()

    metrics = calculate_metrics(y_pred, y_true)

    print("✓ Metrics calculated successfully:")
    for name, value in metrics.items():
        print(f"  - {name}: {value:.4f}")


def test_data_preprocessing():
    """Test data preprocessing functions."""
    print("\nTesting data preprocessing...")

    try:
        from mm_intern_image_src.dataset import MultiModalLandslideDataset

        print("✓ Dataset class imported successfully")

        # Test NDVI calculation
        from mm_intern_image_src.predict import calculate_ndvi, split_and_process_modalities

        # Create dummy 12-channel data
        dummy_data = np.random.rand(64, 64, 12).astype(np.float32)

        # Test modality splitting
        optical, sar, sar_diff = split_and_process_modalities(dummy_data)

        print(f"✓ Modality splitting successful:")
        print(f"  - Optical: {optical.shape}")
        print(f"  - SAR: {sar.shape}")
        print(f"  - SAR diff: {sar_diff.shape}")

        # Test NDVI calculation
        red = dummy_data[:, :, 0]
        nir = dummy_data[:, :, 3]
        ndvi = calculate_ndvi(red, nir)

        print(f"✓ NDVI calculation successful, range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        return False

    return True


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    from mm_intern_image_src.utils import seed_everything, format_metrics, get_class_weights

    # Test seeding
    seed_everything(42)
    print("✓ Random seed set successfully")

    # Test metrics formatting
    dummy_metrics = {"accuracy": 0.85, "f1_score": 0.73, "precision": 0.80}
    formatted = format_metrics(dummy_metrics)
    print(f"✓ Metrics formatting: {formatted}")

    # Test class weights
    labels = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1])
    weights = get_class_weights(labels)
    print(f"✓ Class weights: {weights}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MM-InternImage-TNF Implementation Test")
    print("=" * 60)

    tests = [
        test_config,
        test_loss_functions,
        test_metrics,
        test_data_preprocessing,
        test_utilities,
        test_model_creation,  # This might fail if InternImage is not set up
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            result = test_func()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        if passed >= total - 1:  # Allow model test to fail
            print("Note: Model test failure might be due to InternImage compilation issues.")
            print("This is expected if DCNv3 operators are not properly compiled.")

    print("=" * 60)


if __name__ == "__main__":
    main()
