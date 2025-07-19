#!/usr/bin/env python3
"""
MM-TNF Model Training Launcher

This script provides a convenient way to start training the TNF model
with proper error handling and configuration validation.

Usage:
    python run_tnf_training.py [--resume CHECKPOINT_PATH] [--debug]

Examples:
    # Start fresh training
    python run_tnf_training.py

    # Resume from checkpoint
    python run_tnf_training.py --resume outputs/checkpoints/tnf_run_20241219_143022/best_model.pth

    # Debug mode
    python run_tnf_training.py --debug
"""

import sys
import argparse
import logging
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def validate_environment():
    """Validate the environment before training."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")

    # Check PyTorch
    try:
        import torch

        if not torch.cuda.is_available():
            issues.append("CUDA not available - training will be very slow on CPU")
        else:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check required packages
    required_packages = ["numpy", "pandas", "scikit-learn", "albumentations", "opencv-python", "timm", "tqdm"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            issues.append(f"Required package not installed: {package}")

    # Check data directories
    data_dir = project_root / "dataset"
    required_paths = [data_dir / "Train.csv", data_dir / "train_data", data_dir / "test_data"]

    for path in required_paths:
        if not path.exists():
            issues.append(f"Required data path missing: {path}")

    # Check project structure
    required_modules = [
        "mm_intern_image_src/__init__.py",
        "mm_intern_image_src/config.py",
        "mm_intern_image_src/models.py",
        "mm_intern_image_src/dataset.py",
        "mm_intern_image_src/train.py",
        "mm_intern_image_src/utils.py",
    ]

    for module_path in required_modules:
        if not (project_root / module_path).exists():
            issues.append(f"Required module missing: {module_path}")

    return issues


def main():
    """Main training launcher."""
    parser = argparse.ArgumentParser(
        description="Launch MM-TNF model training", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment, don't start training")
    parser.add_argument("--config-override", type=str, help="JSON string to override config parameters")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    print("🚀 MM-TNF Model Training Launcher")
    print("=" * 50)

    # Validate environment
    print("🔍 Validating environment...")
    issues = validate_environment()

    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running training.")
        return 1
    else:
        print("✅ Environment validation passed")

    if args.validate_only:
        print("Validation complete. Exiting.")
        return 0

    try:
        # Import project modules
        print("📦 Importing project modules...")
        from mm_intern_image_src import config, run_training

        # Apply config overrides if provided
        if args.config_override:
            import json

            overrides = json.loads(args.config_override)
            print(f"🔧 Applying config overrides: {overrides}")
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"  - {key}: {value}")
                else:
                    print(f"  - Warning: Unknown config key '{key}'")

        # Validate resume checkpoint
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                print(f"❌ Resume checkpoint not found: {resume_path}")
                return 1
            print(f"🔄 Resuming from checkpoint: {resume_path}")

        # Print configuration summary
        config.print_config()

        # Start training
        print("🎯 Starting TNF model training...")
        run_training(args)

        print("🎉 Training completed successfully!")
        return 0

    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.error("Make sure you're running from the project root directory")
        if args.debug:
            traceback.print_exc()
        return 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
