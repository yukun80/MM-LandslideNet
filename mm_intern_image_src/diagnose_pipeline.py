#!/usr/bin/env python3
"""
MM-InternImage-TNF Pipeline Diagnostic Script

This script diagnoses common issues in the training and prediction pipeline.
It checks for:
- Missing dependencies
- Configuration issues
- Data integrity
- Model import problems
- GPU availability

Usage:
    python diagnose_pipeline.py
"""

import sys
import os
from pathlib import Path
import importlib
import traceback


"""
python -m mm_intern_image_src.diagnose_pipeline
"""

class PipelineDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def log_issue(self, message):
        """Log a critical issue."""
        self.issues.append(message)
        print(f"❌ {message}")

    def log_warning(self, message):
        """Log a warning."""
        self.warnings.append(message)
        print(f"⚠️ {message}")

    def log_pass(self, message):
        """Log a passed check."""
        self.passed_checks.append(message)
        print(f"✅ {message}")

    def check_project_structure(self):
        """Check if project structure is correct."""
        print("\n📁 Checking project structure...")

        required_dirs = ["mm_intern_image_src", "InternImage", "dataset", "outputs"]

        required_files = [
            "mm_intern_image_src/config.py",
            "mm_intern_image_src/models.py",
            "mm_intern_image_src/dataset.py",
            "mm_intern_image_src/train.py",
            "mm_intern_image_src/utils.py",
            "mm_intern_image_src/__init__.py",
            "dataset/Train.csv",
        ]

        # Check directories
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                self.log_pass(f"Directory exists: {dir_path}")
            else:
                self.log_issue(f"Missing directory: {dir_path}")

        # Check files
        for file_path in required_files:
            path = Path(file_path)
            if path.exists() and path.is_file():
                self.log_pass(f"File exists: {file_path}")
            else:
                self.log_issue(f"Missing file: {file_path}")

    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("\n📦 Checking dependencies...")

        required_packages = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("albumentations", "Albumentations"),
            ("opencv-cv2", "OpenCV", "cv2"),
            ("scikit-learn", "Scikit-learn", "sklearn"),
            ("tqdm", "TQDM"),
            ("timm", "TIMM"),
            ("pathlib", "Pathlib"),
        ]

        for package_info in required_packages:
            package_name = package_info[0]
            display_name = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name

            try:
                importlib.import_module(import_name)
                self.log_pass(f"{display_name} is available")
            except ImportError:
                self.log_issue(f"{display_name} is not installed (pip install {package_name})")

    def check_gpu_availability(self):
        """Check GPU availability and CUDA setup."""
        print("\n🖥️ Checking GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.log_pass(f"CUDA is available with {gpu_count} GPU(s)")

                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.log_pass(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.log_warning("CUDA is not available - training will be very slow on CPU")

        except ImportError:
            self.log_issue("PyTorch is not installed")

    def check_config_completeness(self):
        """Check if config.py has all required attributes."""
        print("\n⚙️ Checking configuration...")

        try:
            sys.path.insert(0, str(Path.cwd()))
            from mm_intern_image_src.config import config

            required_configs = [
                "AUGMENTATION_CONFIG",
                "TRAIN_CSV_PATH",
                "TRAIN_DATA_DIR",
                "BATCH_SIZE",
                "NUM_EPOCHS",
                "LEARNING_RATE",
                "LOSS_CONFIG",
            ]

            for config_name in required_configs:
                if hasattr(config, config_name):
                    self.log_pass(f"Config has {config_name}")
                else:
                    self.log_issue(f"Config missing {config_name}")

            # Check AUGMENTATION_CONFIG structure
            if hasattr(config, "AUGMENTATION_CONFIG"):
                aug_config = config.AUGMENTATION_CONFIG
                required_aug_keys = ["horizontal_flip_prob", "vertical_flip_prob", "rotation_limit"]

                for key in required_aug_keys:
                    if key in aug_config:
                        self.log_pass(f"Augmentation config has {key}")
                    else:
                        self.log_issue(f"Augmentation config missing {key}")

        except ImportError as e:
            self.log_issue(f"Cannot import config: {e}")
        except Exception as e:
            self.log_issue(f"Config error: {e}")

    def check_data_integrity(self):
        """Check data file integrity."""
        print("\n📊 Checking data integrity...")

        try:
            # Check if Train.csv exists and is readable
            csv_path = Path("dataset/Train.csv")
            if csv_path.exists():
                import pandas as pd

                df = pd.read_csv(csv_path)
                self.log_pass(f"Train.csv loaded successfully ({len(df)} samples)")

                # Check required columns
                required_columns = ["ID", "label"]
                for col in required_columns:
                    if col in df.columns:
                        self.log_pass(f"CSV has required column: {col}")
                    else:
                        self.log_issue(f"CSV missing required column: {col}")

                # Check data directory
                data_dir = Path("dataset/train_data")
                if data_dir.exists():
                    npy_files = list(data_dir.glob("*.npy"))
                    self.log_pass(f"Found {len(npy_files)} .npy files in train_data")

                    # Check if sample IDs match files
                    sample_ids = set(df["ID"].astype(str))
                    file_ids = set(f.stem for f in npy_files)

                    missing_files = sample_ids - file_ids
                    extra_files = file_ids - sample_ids

                    if not missing_files:
                        self.log_pass("All CSV samples have corresponding data files")
                    else:
                        self.log_issue(f"{len(missing_files)} samples missing data files")

                    if extra_files:
                        self.log_warning(f"{len(extra_files)} extra data files found")

                else:
                    self.log_issue("train_data directory not found")
            else:
                self.log_issue("Train.csv not found")

        except Exception as e:
            self.log_issue(f"Data integrity check failed: {e}")

    def check_model_imports(self):
        """Check if model components can be imported."""
        print("\n🤖 Checking model imports...")

        try:
            sys.path.insert(0, str(Path.cwd()))

            # Test config import
            from mm_intern_image_src.config import config

            self.log_pass("Config import successful")

            # Test model import
            from mm_intern_image_src.models import create_optical_dominated_model

            self.log_pass("Model import successful")

            # Test dataset import
            from mm_intern_image_src.dataset import MultiModalLandslideDataset

            self.log_pass("Dataset import successful")

            # Test utils import
            from mm_intern_image_src.utils import CombinedLoss

            self.log_pass("Utils import successful")

            # Test InternImage import
            try:
                from mm_intern_image_src.intern_image_import import InternImage

                self.log_pass("InternImage import successful")
            except Exception as e:
                self.log_warning(f"InternImage import issue: {e}")

        except ImportError as e:
            self.log_issue(f"Import error: {e}")
            traceback.print_exc()
        except Exception as e:
            self.log_issue(f"Unexpected error in imports: {e}")

    def check_augmentation_functions(self):
        """Check if augmentation functions work correctly."""
        print("\n🔄 Checking augmentation functions...")

        try:
            sys.path.insert(0, str(Path.cwd()))
            from mm_intern_image_src.dataset import get_augmentations

            # Test train augmentations
            train_augs = get_augmentations("train")
            if train_augs is not None:
                self.log_pass("Train augmentations created successfully")
            else:
                self.log_issue("Train augmentations returned None")

            # Test validation augmentations
            val_augs = get_augmentations("val")
            if val_augs is not None:
                self.log_pass("Validation augmentations created successfully")
            else:
                self.log_issue("Validation augmentations returned None")

        except Exception as e:
            self.log_issue(f"Augmentation function error: {e}")

    def run_all_checks(self):
        """Run all diagnostic checks."""
        print("🔍 MM-InternImage-TNF Pipeline Diagnostic")
        print("=" * 60)

        self.check_project_structure()
        self.check_dependencies()
        self.check_gpu_availability()
        self.check_config_completeness()
        self.check_data_integrity()
        self.check_model_imports()
        self.check_augmentation_functions()

        # Summary
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print(f"✅ Passed checks: {len(self.passed_checks)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"❌ Critical issues: {len(self.issues)}")

        if self.issues:
            print(f"\n🚨 Critical Issues to Fix:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.warnings:
            print(f"\n⚠️ Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.issues:
            print("\n🎉 No critical issues found! Pipeline should work correctly.")
            print("\n🚀 You can run training with:")
            print("   python -m mm_intern_image_src.train")
        else:
            print(f"\n🔧 Please fix the {len(self.issues)} critical issues before training.")
            print("\n💡 Run 'python fix_training_pipeline.py' to apply automatic fixes.")


def main():
    """Main function."""
    diagnostic = PipelineDiagnostic()
    diagnostic.run_all_checks()


if __name__ == "__main__":
    main()
