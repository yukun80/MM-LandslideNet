#!/usr/bin/env python3
"""
MM-TNF Model Testing and Validation Script

This script tests the TNF model architecture, data loading, and training pipeline
to ensure everything works correctly before starting full training.

Usage:
    python test_tnf_model.py [--quick] [--debug] [--gpu-only]

Examples:
    # Full test suite
    python test_tnf_model.py

    # Quick tests only
    python test_tnf_model.py --quick

    # Debug mode with detailed output
    python test_tnf_model.py --debug

    # Skip tests that require GPU
    python test_tnf_model.py --gpu-only
"""

import sys
import argparse
import logging
import traceback
import time
from pathlib import Path
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )


class TNFModelTester:
    """Comprehensive tester for TNF model components."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.passed_tests = []
        self.failed_tests = []
        self.warnings = []

    def log_test(self, test_name: str, success: bool, message: str = "", warning: bool = False):
        """Log test result."""
        if warning:
            self.warnings.append(f"{test_name}: {message}")
            print(f"⚠️ {test_name}: {message}")
        elif success:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}" + (f": {message}" if message else ""))
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {message}")

    def test_imports(self):
        """Test all required imports."""
        print("\n📦 Testing imports...")

        # Core imports
        try:
            import torch
            import numpy as np
            import pandas as pd

            self.log_test("Core packages (torch, numpy, pandas)", True)
        except ImportError as e:
            self.log_test("Core packages", False, str(e))
            return False

        # Computer vision imports
        try:
            import cv2
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            self.log_test("Computer vision packages (opencv, albumentations)", True)
        except ImportError as e:
            self.log_test("Computer vision packages", False, str(e))
            return False

        # ML packages
        try:
            import timm
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score, precision_score, recall_score

            self.log_test("ML packages (timm, sklearn)", True)
        except ImportError as e:
            self.log_test("ML packages", False, str(e))
            return False

        # Project modules
        try:
            from mm_intern_image_src import config
            from mm_intern_image_src.models import create_tnf_model
            from mm_intern_image_src.dataset import MultiModalLandslideDataset
            from mm_intern_image_src.utils import CombinedLoss

            self.log_test("Project modules", True)
        except ImportError as e:
            self.log_test("Project modules", False, str(e))
            return False

        return True

    def test_config(self):
        """Test configuration."""
        print("\n⚙️ Testing configuration...")

        try:
            from mm_intern_image_src.config import config

            # Test config validation
            config.validate_config()
            self.log_test("Config validation", True)

            # Test essential config values
            assert config.MODEL_CONFIG["optical_channels"] == 5
            assert config.MODEL_CONFIG["sar_channels"] == 8
            assert len(config.MODEL_CONFIG["branch_weights"]) == 3
            self.log_test("Config values", True)

            # Test path setup
            config.setup_run_paths("test_run")
            self.log_test("Path setup", True)

            return True

        except Exception as e:
            self.log_test("Configuration", False, str(e))
            return False

    def test_data_loading(self):
        """Test data loading functionality."""
        print("\n📊 Testing data loading...")

        try:
            from mm_intern_image_src.config import config
            from mm_intern_image_src.dataset import MultiModalLandslideDataset, get_augmentations
            import pandas as pd

            # Check if data files exist
            if not config.TRAIN_CSV_PATH.exists():
                self.log_test("Data files", False, f"Train CSV not found: {config.TRAIN_CSV_PATH}")
                return False

            if not config.TRAIN_DATA_DIR.exists():
                self.log_test("Data files", False, f"Train data dir not found: {config.TRAIN_DATA_DIR}")
                return False

            # Load sample data
            df = pd.read_csv(config.TRAIN_CSV_PATH)
            sample_df = df.head(5)  # Use only 5 samples for testing

            # Test dataset creation
            dataset = MultiModalLandslideDataset(
                df=sample_df,
                data_dir=config.TRAIN_DATA_DIR,
                augmentations=get_augmentations("val"),  # Use val (no augmentation) for testing
                mode="train",
            )

            self.log_test("Dataset creation", True, f"Created with {len(dataset)} samples")

            # Test data loading
            sample = dataset[0]

            # Check sample format
            expected_keys = ["optical", "sar", "label", "id"]
            for key in expected_keys:
                if key not in sample:
                    self.log_test("Sample format", False, f"Missing key: {key}")
                    return False

            # Check tensor shapes
            optical_shape = sample["optical"].shape
            sar_shape = sample["sar"].shape

            if optical_shape != (5, 64, 64):
                self.log_test("Optical shape", False, f"Expected (5,64,64), got {optical_shape}")
                return False

            if sar_shape != (8, 64, 64):
                self.log_test("SAR shape", False, f"Expected (8,64,64), got {sar_shape}")
                return False

            self.log_test("Data loading", True, f"Optical: {optical_shape}, SAR: {sar_shape}")

            # Test augmentations
            aug_dataset = MultiModalLandslideDataset(
                df=sample_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("train"), mode="train"
            )

            aug_sample = aug_dataset[0]
            self.log_test("Data augmentation", True)

            return True

        except Exception as e:
            self.log_test("Data loading", False, str(e))
            if self.debug:
                traceback.print_exc()
            return False

    def test_model_creation(self):
        """Test model creation and forward pass."""
        print("\n🏗️ Testing model creation...")

        try:
            import torch
            from mm_intern_image_src.models import create_tnf_model

            # Create model
            model = create_tnf_model(
                pretrained=False,  # Faster for testing
                optical_channels=5,
                sar_channels=8,
                optical_feature_dim=512,
                sar_feature_dim=512,
                fusion_dim=512,
            )

            self.log_test("Model creation", True)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.log_test("Parameter counting", True, f"Total: {total_params:,}, Trainable: {trainable_params:,}")

            # Test model forward pass
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()

            # Create dummy input
            batch_size = 2
            optical_input = torch.randn(batch_size, 5, 64, 64).to(device)
            sar_input = torch.randn(batch_size, 8, 64, 64).to(device)

            with torch.no_grad():
                outputs = model(optical_input, sar_input)

            # Check output format
            expected_keys = ["optical_logits", "sar_logits", "fusion_logits", "final_logits", "fusion_weights"]
            for key in expected_keys:
                if key not in outputs:
                    self.log_test("Model output format", False, f"Missing key: {key}")
                    return False

            # Check output shapes
            for key in ["optical_logits", "sar_logits", "fusion_logits", "final_logits"]:
                expected_shape = (batch_size, 1)
                actual_shape = outputs[key].shape
                if actual_shape != expected_shape:
                    self.log_test("Output shapes", False, f"{key}: expected {expected_shape}, got {actual_shape}")
                    return False

            self.log_test("Model forward pass", True, f"All outputs have correct shapes")

            # Test loss computation with different loss functions
            targets = torch.randint(0, 2, (batch_size,)).float().to(device)  # Shape: (B,)

            # Test with standard BCEWithLogitsLoss
            try:
                import torch.nn as nn

                bce_loss = nn.BCEWithLogitsLoss()
                loss_dict_bce = model.compute_loss(outputs, targets, bce_loss)

                expected_loss_keys = ["total_loss", "optical_loss", "sar_loss", "fusion_loss"]
                for key in expected_loss_keys:
                    if key not in loss_dict_bce:
                        self.log_test("BCE Loss computation", False, f"Missing loss key: {key}")
                        return False

                self.log_test("BCE Loss computation", True)
            except Exception as e:
                self.log_test("BCE Loss computation", False, str(e))
                return False

            # Test with CombinedLoss
            try:
                from mm_intern_image_src.utils import CombinedLoss

                combined_loss = CombinedLoss()
                loss_dict_combined = model.compute_loss(outputs, targets, combined_loss)

                for key in expected_loss_keys:
                    if key not in loss_dict_combined:
                        self.log_test("Combined Loss computation", False, f"Missing loss key: {key}")
                        return False

                self.log_test("Combined Loss computation", True)
            except Exception as e:
                self.log_test("Combined Loss computation", False, str(e))
                return False

            return True

        except Exception as e:
            self.log_test("Model creation", False, str(e))
            if self.debug:
                traceback.print_exc()
            return False

    def test_training_components(self, quick: bool = False):
        """Test training pipeline components."""
        print("\n🎯 Testing training components...")

        try:
            import torch
            from torch.utils.data import DataLoader
            from mm_intern_image_src.config import config
            from mm_intern_image_src.dataset import MultiModalLandslideDataset, get_augmentations
            from mm_intern_image_src.models import create_tnf_model
            from mm_intern_image_src.utils import CombinedLoss, calculate_metrics
            import pandas as pd
            import numpy as np

            # Use small dataset for testing
            df = pd.read_csv(config.TRAIN_CSV_PATH)
            small_df = df.head(10)  # Very small for testing

            # Create dataset and dataloader
            dataset = MultiModalLandslideDataset(
                df=small_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("train"), mode="train"
            )

            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            self.log_test("DataLoader creation", True)

            # Create model and loss
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = create_tnf_model(pretrained=False).to(device)
            criterion = CombinedLoss(**config.LOSS_CONFIG)

            self.log_test("Model and loss setup", True)

            if quick:
                print("  Skipping training loop test (quick mode)")
                return True

            # Test training loop (one batch)
            model.train()
            batch = next(iter(dataloader))

            optical_data = batch["optical"].to(device)
            sar_data = batch["sar"].to(device)
            # 修复：确保targets的维度和类型正确
            targets = batch["label"].to(device)  # Keep as (B,) for now

            # Forward pass
            outputs = model(optical_data, sar_data)

            # 修复：使用模型的compute_loss方法，它会正确处理targets维度
            loss_dict = model.compute_loss(outputs, targets, criterion)
            loss = loss_dict["total_loss"]

            # Backward pass
            loss.backward()

            # Check gradients
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if not has_gradients:
                self.log_test("Gradient computation", False, "No gradients computed")
                return False

            self.log_test("Training loop", True, f"Loss: {loss.item():.4f}")

            # Test metrics calculation
            with torch.no_grad():
                probs = torch.sigmoid(outputs["final_logits"]).cpu().numpy()
                # 修复：确保targets格式正确用于metrics计算
                if targets.dim() == 1:
                    targets_np = targets.cpu().numpy()
                else:
                    targets_np = targets.squeeze().cpu().numpy()

                metrics = calculate_metrics(probs.squeeze(), targets_np, threshold=0.5)
                self.log_test("Metrics calculation", True, f"F1: {metrics['f1_score']:.3f}")

            return True

        except Exception as e:
            self.log_test("Training components", False, str(e))
            if self.debug:
                traceback.print_exc()
            return False

    def test_gpu_compatibility(self):
        """Test GPU compatibility and performance."""
        print("\n🎮 Testing GPU compatibility...")

        try:
            import torch

            if not torch.cuda.is_available():
                self.log_test("GPU availability", False, "CUDA not available", warning=True)
                return False

            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            self.log_test("GPU detection", True, f"{gpu_name} ({gpu_memory:.1f}GB)")

            # Test memory allocation
            from mm_intern_image_src.models import create_tnf_model

            model = create_tnf_model(pretrained=False).to(device)

            # Test with larger batch to check memory
            batch_size = 8
            optical_input = torch.randn(batch_size, 5, 64, 64).to(device)
            sar_input = torch.randn(batch_size, 8, 64, 64).to(device)

            # Measure memory usage
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()

            with torch.no_grad():
                outputs = model(optical_input, sar_input)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - start_memory) / 1e6  # MB

            self.log_test("GPU memory test", True, f"Used {memory_used:.1f}MB for batch size {batch_size}")

            # Test training memory usage
            model.train()
            targets = torch.randint(0, 2, (batch_size, 1)).float().to(device)

            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()

            outputs = model(optical_input, sar_input)
            loss_dict = model.compute_loss(outputs, targets)
            loss = loss_dict["total_loss"]
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated()
            training_memory = (peak_memory - start_memory) / 1e6  # MB

            self.log_test("GPU training memory", True, f"Used {training_memory:.1f}MB for training")

            # Memory recommendations
            if gpu_memory < 4.0:
                self.log_test(
                    "GPU memory warning",
                    False,
                    f"Low GPU memory ({gpu_memory:.1f}GB). Consider reducing batch size.",
                    warning=True,
                )
            elif gpu_memory < 8.0:
                self.log_test(
                    "GPU memory recommendation",
                    True,
                    "Moderate GPU memory. Batch size 32-64 recommended.",
                    warning=True,
                )
            else:
                self.log_test("GPU memory", True, "Sufficient GPU memory for large batches")

            return True

        except Exception as e:
            self.log_test("GPU compatibility", False, str(e))
            if self.debug:
                traceback.print_exc()
            return False

    def test_prediction_pipeline(self):
        """Test prediction pipeline."""
        print("\n🔮 Testing prediction pipeline...")

        try:
            import torch
            from mm_intern_image_src.dataset import get_tta_augmentations
            from mm_intern_image_src.models import create_tnf_model

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = create_tnf_model(pretrained=False).to(device)
            model.eval()

            # Test standard prediction
            batch_size = 4
            optical_input = torch.randn(batch_size, 5, 64, 64).to(device)
            sar_input = torch.randn(batch_size, 8, 64, 64).to(device)

            with torch.no_grad():
                outputs = model(optical_input, sar_input)
                probs = torch.sigmoid(outputs["final_logits"])
                predictions = (probs > 0.5).float()

            self.log_test("Standard prediction", True, f"Shape: {predictions.shape}")

            # Test TTA transforms
            tta_transforms = get_tta_augmentations()
            self.log_test("TTA transforms", True, f"{len(tta_transforms)} transforms")

            # Test prediction methods
            prediction_results = model.predict(optical_input, sar_input, use_ensemble=True)
            expected_keys = ["probabilities", "predictions", "logits", "all_outputs"]

            for key in expected_keys:
                if key not in prediction_results:
                    self.log_test("Prediction methods", False, f"Missing key: {key}")
                    return False

            self.log_test("Prediction methods", True)

            return True

        except Exception as e:
            self.log_test("Prediction pipeline", False, str(e))
            if self.debug:
                traceback.print_exc()
            return False

    def run_all_tests(self, quick: bool = False, gpu_only: bool = False):
        """Run all tests."""
        print("🧪 MM-TNF Model Test Suite")
        print("=" * 50)

        start_time = time.time()

        # Core tests
        if not self.test_imports():
            print("\n❌ Import tests failed. Cannot continue.")
            return False

        if not self.test_config():
            print("\n❌ Configuration tests failed. Cannot continue.")
            return False

        if not self.test_data_loading():
            print("\n❌ Data loading tests failed. Cannot continue.")
            return False

        if not self.test_model_creation():
            print("\n❌ Model creation tests failed. Cannot continue.")
            return False

        # Optional tests
        if not gpu_only:
            self.test_training_components(quick=quick)
            self.test_prediction_pipeline()

        # GPU tests
        if torch.cuda.is_available():
            self.test_gpu_compatibility()
        else:
            self.log_test("GPU tests", False, "CUDA not available", warning=True)

        # Summary
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n📊 Test Summary ({duration:.1f}s)")
        print("=" * 50)
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")

        if self.failed_tests:
            print(f"\n❌ Failed tests:")
            for test in self.failed_tests:
                print(f"  - {test}")

        if self.warnings:
            print(f"\n⚠️ Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        success = len(self.failed_tests) == 0

        if success:
            print(f"\n🎉 All tests passed! The TNF model is ready for training.")
        else:
            print(f"\n❌ Some tests failed. Please fix the issues before training.")

        return success


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test MM-TNF model components", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--quick", action="store_true", help="Skip time-consuming tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--gpu-only", action="store_true", help="Only run GPU-specific tests")

    args = parser.parse_args()

    setup_logging(args.debug)

    tester = TNFModelTester(debug=args.debug)
    success = tester.run_all_tests(quick=args.quick, gpu_only=args.gpu_only)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
