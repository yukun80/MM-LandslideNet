#!/usr/bin/env python3
"""
Environment Check Script for Linux
Check if all paths and configurations are working correctly in Linux environment
"""

import os
import sys
import platform
import torch
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from optical_src.config import OpticalBaselineConfig


def check_system_info():
    """Check system information"""
    print("🖥️ System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Python Version: {sys.version}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Current Working Directory: {os.getcwd()}")
    print()


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("🔥 PyTorch & CUDA Check:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("   Using CPU")
    print()


def check_project_structure():
    """Check project directory structure"""
    print("📁 Project Structure Check:")

    config = Config()
    print(f"   Project Root: {config.PROJECT_ROOT}")
    print(f"   Exists: {config.PROJECT_ROOT.exists()}")

    # Check key directories
    directories = [
        ("Dataset Root", config.DATASET_ROOT),
        ("Train Data", config.TRAIN_DATA_DIR),
        ("Test Data", config.TEST_DATA_DIR),
        ("Output Root", config.OUTPUT_ROOT),
        ("Checkpoint Dir", config.CHECKPOINT_DIR),
        ("Log Dir", config.LOG_DIR),
    ]

    for name, path in directories:
        print(f"   {name}: {path}")
        print(f"     Exists: {path.exists()}")
        if path.exists():
            if path.is_dir():
                try:
                    items = list(path.iterdir())
                    print(f"     Items: {len(items)}")
                except PermissionError:
                    print(f"     Permission denied")
            else:
                print(f"     Size: {path.stat().st_size} bytes")
    print()


def check_data_files():
    """Check data files existence"""
    print("📄 Data Files Check:")

    config = Config()

    # Check CSV files
    csv_files = [
        ("Train CSV", config.TRAIN_CSV),
        ("Test CSV", config.TEST_CSV),
        ("Sample Submission", config.SAMPLE_SUBMISSION),
    ]

    for name, path in csv_files:
        print(f"   {name}: {path}")
        print(f"     Exists: {path.exists()}")
        if path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(path)
                print(f"     Shape: {df.shape}")
            except Exception as e:
                print(f"     Error reading: {e}")

    # Check sample data files
    if config.TRAIN_DATA_DIR.exists():
        npy_files = list(config.TRAIN_DATA_DIR.glob("*.npy"))
        print(f"   Train .npy files: {len(npy_files)}")
        if npy_files:
            # Check first file
            try:
                import numpy as np

                sample_data = np.load(npy_files[0])
                print(f"     Sample shape: {sample_data.shape}")
                print(f"     Sample dtype: {sample_data.dtype}")
            except Exception as e:
                print(f"     Error loading sample: {e}")

    if config.TEST_DATA_DIR.exists():
        npy_files = list(config.TEST_DATA_DIR.glob("*.npy"))
        print(f"   Test .npy files: {len(npy_files)}")

    print()


def check_optical_config():
    """Check optical baseline configuration"""
    print("🔧 Optical Baseline Configuration:")

    try:
        config = OpticalBaselineConfig()
        print(f"   Model Name: {config.MODEL_NAME}")
        print(f"   Device: {config.DEVICE}")
        print(f"   Batch Size: {config.BATCH_SIZE}")
        print(f"   Num Workers: {config.NUM_WORKERS}")
        print(f"   Mixed Precision: {config.MIXED_PRECISION}")

        # Check optical-specific directories
        optical_dirs = [
            ("Optical Log Dir", config.OPTICAL_LOG_DIR),
            ("Optical Checkpoint Dir", config.OPTICAL_CHECKPOINT_DIR),
            ("Optical Output Dir", config.OPTICAL_OUTPUT_DIR),
        ]

        for name, path in optical_dirs:
            print(f"   {name}: {path}")
            print(f"     Exists: {path.exists()}")

    except Exception as e:
        print(f"   Error initializing optical config: {e}")

    print()


def check_dependencies():
    """Check required dependencies"""
    print("📦 Dependencies Check:")

    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "albumentations",
        "opencv-python",
        "timm",
        "tqdm",
        "tensorboard",
        "pillow",
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT FOUND")

    print()


def create_missing_directories():
    """Create missing directories"""
    print("📁 Creating Missing Directories:")

    config = Config()
    optical_config = OpticalBaselineConfig()

    # Create base directories
    config.create_dirs()
    print(f"   ✅ Created base directories")

    # Create optical directories
    optical_config._create_optical_dirs()
    print(f"   ✅ Created optical directories")

    # Create data_check directory
    data_check_dir = config.DATASET_ROOT / "data_check"
    data_check_dir.mkdir(exist_ok=True)
    print(f"   ✅ Created data_check directory: {data_check_dir}")

    print()


def main():
    """Main function"""
    print("🔍 Linux Environment Check for MM-LandslideNet")
    print("=" * 60)

    check_system_info()
    check_pytorch_cuda()
    check_project_structure()
    check_data_files()
    check_optical_config()
    check_dependencies()
    create_missing_directories()

    print("✅ Environment check completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
