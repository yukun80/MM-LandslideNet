#!/usr/bin/env python3
"""
MM-TNF Model Prediction Launcher

This script provides a convenient way to run predictions with the trained TNF model
with proper error handling and flexible options.

Usage:
    python run_tnf_prediction.py --model PATH_TO_MODEL [OPTIONS]

Examples:
    # Basic prediction
    python run_tnf_prediction.py --model outputs/checkpoints/tnf_run_20241219_143022/best_model.pth

    # Prediction with TTA disabled
    python run_tnf_prediction.py --model best_model.pth --no-tta

    # Custom threshold and output directory
    python run_tnf_prediction.py --model best_model.pth --threshold 0.3 --output custom_predictions

    # Use specific branch for submission
    python run_tnf_prediction.py --model best_model.pth --branch fusion --threshold 0.4
"""

import sys
import argparse
import logging
from pathlib import Path
import traceback
from datetime import datetime

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


def validate_model_checkpoint(model_path: Path) -> bool:
    """Validate model checkpoint file."""
    if not model_path.exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        return False

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Check if it's a proper checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                print(f"✅ Valid checkpoint found (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                print("✅ Valid state dict found")
        else:
            print("⚠️ Checkpoint format might be unusual, but proceeding...")

        return True

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def validate_test_data():
    """Validate test data availability."""
    data_dir = project_root / "dataset"
    test_csv = data_dir / "Test.csv"
    test_data_dir = data_dir / "test_data"

    issues = []

    if not test_csv.exists():
        issues.append(f"Test CSV not found: {test_csv}")

    if not test_data_dir.exists():
        issues.append(f"Test data directory not found: {test_data_dir}")
    elif not any(test_data_dir.glob("*.npy")):
        issues.append(f"No .npy files found in test data directory: {test_data_dir}")

    return issues


def estimate_prediction_time(num_samples: int, use_tta: bool, batch_size: int) -> str:
    """Estimate prediction time."""
    # Rough estimates based on typical GPU performance
    samples_per_second = 50 if not use_tta else 10  # TTA is much slower

    total_seconds = num_samples / samples_per_second

    if total_seconds < 60:
        return f"~{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"~{total_seconds/60:.1f} minutes"
    else:
        return f"~{total_seconds/3600:.1f} hours"


def main():
    """Main prediction launcher."""
    parser = argparse.ArgumentParser(
        description="Launch MM-TNF model prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")

    # Optional arguments
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: outputs/predictions/TIMESTAMP)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument(
        "--branch",
        type=str,
        default="final",
        choices=["optical", "sar", "fusion", "final"],
        help="Which branch to use for submission (default: final)",
    )
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable Test-Time Augmentation (faster but potentially less accurate)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--validate-only", action="store_true", help="Only validate setup, don't run prediction")
    parser.add_argument("--force", action="store_true", help="Force prediction even if validation warnings exist")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    print("🔮 MM-TNF Model Prediction Launcher")
    print("=" * 50)

    # Validate model checkpoint
    print("🔍 Validating model checkpoint...")
    model_path = Path(args.model)
    if not validate_model_checkpoint(model_path):
        return 1

    # Validate test data
    print("🔍 Validating test data...")
    test_issues = validate_test_data()
    if test_issues:
        print("❌ Test data validation failed:")
        for issue in test_issues:
            print(f"  - {issue}")
        if not args.force:
            print("Use --force to proceed anyway.")
            return 1
        else:
            print("⚠️ Proceeding with --force flag...")
    else:
        print("✅ Test data validation passed")

    if args.validate_only:
        print("Validation complete. Exiting.")
        return 0

    try:
        # Import project modules
        print("📦 Importing project modules...")
        from mm_intern_image_src import config
        from mm_intern_image_src.predict import run_prediction

        # Setup output directory
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "outputs" / "predictions" / f"tnf_prediction_{timestamp}"
        else:
            output_dir = Path(args.output)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load test data info for time estimation
        try:
            import pandas as pd

            test_csv = project_root / "dataset" / "Test.csv"
            test_df = pd.read_csv(test_csv)
            num_samples = len(test_df)

            print(f"📊 Test dataset: {num_samples} samples")
            estimated_time = estimate_prediction_time(num_samples, not args.no_tta, args.batch_size)
            print(f"⏱️ Estimated time: {estimated_time}")

        except Exception as e:
            logger.warning(f"Could not estimate prediction time: {e}")
            num_samples = "unknown"

        # Print configuration
        print(f"\n🔧 Prediction Configuration:")
        print(f"  Model: {model_path}")
        print(f"  Output: {output_dir}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Branch: {args.branch}")
        print(f"  TTA: {'Disabled' if args.no_tta else 'Enabled'}")
        print(f"  Samples: {num_samples}")

        # Confirm before starting
        if not args.force:
            response = input("\n📝 Start prediction? [y/N]: ")
            if response.lower() not in ["y", "yes"]:
                print("Prediction cancelled.")
                return 0

        # Start prediction
        print(f"\n🎯 Starting prediction...")
        run_prediction(
            model_path=model_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            threshold=args.threshold,
            use_branch=args.branch,
        )

        print(f"\n🎉 Prediction completed successfully!")
        print(f"📁 Results saved to: {output_dir}")

        # List output files
        output_files = list(output_dir.glob("*.csv"))
        if output_files:
            print(f"📄 Output files:")
            for file in output_files:
                print(f"  - {file.name}")

        return 0

    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.error("Make sure you're running from the project root directory")
        if args.debug:
            traceback.print_exc()
        return 1

    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
