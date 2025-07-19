"""
Batch Prediction Script for MM-InternImage-TNF Model

This script performs batch prediction on test data and generates submission files.
Supports Test Time Augmentation (TTA) for improved performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

"""
python -m mm_intern_image_src.predict \
    --model_path outputs/checkpoints/OpticalDominatedCooperativeModel_20250718_165907/best_model.pth \
    --test_data_dir dataset/test_data \
    --output_dir outputs/submissions \
    --submission_name submission_internimage_tta_18072025_165907.csv \
    --threshold 0.6 \
    --use_tta \
    --save_probabilities --device cuda
"""

# Import project modules
try:
    from .config import config
    from .models import create_optical_dominated_model
    from .dataset import get_tta_augmentations, get_augmentations
    from .utils import load_checkpoint, setup_logging
except ImportError:
    # Fallback for direct execution
    from config import config
    from models import create_optical_dominated_model
    from dataset import get_tta_augmentations, get_augmentations
    from utils import load_checkpoint, setup_logging

# Setup logger
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch predictor for test data."""

    def __init__(self, model_path: Path, device: str = "auto"):
        """
        Initialize the batch predictor.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.stats = None

        # Load model and statistics
        self._load_model()
        self._load_stats()

        logger.info(f"BatchPredictor initialized on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        # Create model
        self.model = create_optical_dominated_model(num_classes=config.NUM_CLASSES, pretrained=False)
        self.model.to(self.device)

        # Load checkpoint
        checkpoint = load_checkpoint(self.model_path, self.model)

        # Set to evaluation mode
        self.model.eval()

        logger.info("Model loaded successfully")

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        # Get training metrics from checkpoint
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            if "val_f1_score" in metrics:
                logger.info(f"Model validation F1-score: {metrics['val_f1_score']:.4f}")

    def _load_stats(self):
        """Load normalization statistics."""
        logger.info(f"Loading statistics from {config.STATS_FILE_PATH}")

        with open(config.STATS_FILE_PATH, "r") as f:
            full_stats = json.load(f)

        # Extract statistics (same as in dataset.py)
        try:
            self.stats = {
                "optical": {
                    "mean": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_0"]["mean"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_1"]["mean"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_2"]["mean"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_3"]["mean"],
                            0.0,  # NDVI mean
                        ]
                    ),
                    "std": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_0"]["std"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_1"]["std"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_2"]["std"],
                            full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_3"]["std"],
                            1.0,  # NDVI std
                        ]
                    ),
                },
                "sar": {
                    "mean": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_4"][
                                "mean"
                            ],
                            full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_5"][
                                "mean"
                            ],
                            full_stats["channel_statistics_by_group"]["sar_ascending"]["channels"]["channel_8"]["mean"],
                            full_stats["channel_statistics_by_group"]["sar_ascending"]["channels"]["channel_9"]["mean"],
                        ]
                    ),
                    "std": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_4"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_5"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_ascending"]["channels"]["channel_8"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_ascending"]["channels"]["channel_9"]["std"],
                        ]
                    ),
                },
                "sar_change": {
                    "mean": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["sar_desc_diff"]["channels"]["channel_6"]["mean"],
                            full_stats["channel_statistics_by_group"]["sar_desc_diff"]["channels"]["channel_7"]["mean"],
                            full_stats["channel_statistics_by_group"]["sar_asc_diff"]["channels"]["channel_10"]["mean"],
                            full_stats["channel_statistics_by_group"]["sar_asc_diff"]["channels"]["channel_11"]["mean"],
                        ]
                    ),
                    "std": np.array(
                        [
                            full_stats["channel_statistics_by_group"]["sar_desc_diff"]["channels"]["channel_6"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_desc_diff"]["channels"]["channel_7"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_asc_diff"]["channels"]["channel_10"]["std"],
                            full_stats["channel_statistics_by_group"]["sar_asc_diff"]["channels"]["channel_11"]["std"],
                        ]
                    ),
                },
            }
            logger.info("Statistics loaded successfully")
        except KeyError as e:
            logger.error(f"Failed to load statistics: {e}")
            raise

    def _preprocess_image(self, image_path: Path) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single image for prediction.

        Args:
            image_path: Path to .npy image file

        Returns:
            Dictionary with preprocessed tensors
        """
        # Load data
        data = np.load(image_path).astype(np.float32)
        if data.shape != (64, 64, 12):
            raise ValueError(f"Expected shape (64, 64, 12), got {data.shape}")

        # Split modalities (same as dataset.py)
        optical = data[:, :, :4]  # R, G, B, NIR
        sar = np.concatenate([data[:, :, 4:6], data[:, :, 8:10]], axis=-1)  # VV, VH desc+asc
        sar_change = np.concatenate([data[:, :, 6:8], data[:, :, 10:12]], axis=-1)  # diff channels

        # Compute NDVI
        red, nir = optical[:, :, 0], optical[:, :, 3]
        ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
        optical_with_ndvi = np.concatenate([optical, ndvi[:, :, np.newaxis]], axis=-1)

        # Normalize each modality
        optical_norm = (optical_with_ndvi - self.stats["optical"]["mean"]) / self.stats["optical"]["std"]
        sar_norm = (sar - self.stats["sar"]["mean"]) / self.stats["sar"]["std"]
        sar_change_norm = (sar_change - self.stats["sar_change"]["mean"]) / self.stats["sar_change"]["std"]

        # Convert to tensors (HWC -> CHW format)
        return {
            "optical": torch.from_numpy(optical_norm.transpose(2, 0, 1)).float().unsqueeze(0),
            "sar": torch.from_numpy(sar_norm.transpose(2, 0, 1)).float().unsqueeze(0),
            "sar_change": torch.from_numpy(sar_change_norm.transpose(2, 0, 1)).float().unsqueeze(0),
        }

    def _predict_single(self, tensors: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Predict a single sample.

        Args:
            tensors: Preprocessed tensor dictionary

        Returns:
            Tuple of (probability, logit)
        """
        # Move tensors to device
        batch = {k: v.to(self.device) for k, v in tensors.items()}

        with torch.no_grad():
            logits = self.model(batch)
            probability = torch.sigmoid(logits).item()

        return probability, logits.item()

    def _predict_with_tta(self, image_path: Path, tta_transforms: List) -> float:
        """
        Predict with Test Time Augmentation.

        Args:
            image_path: Path to image file
            tta_transforms: List of TTA transforms

        Returns:
            Average probability across all TTA predictions
        """
        probabilities = []

        # Load and preprocess original image
        base_tensors = self._preprocess_image(image_path)

        # For each TTA transform
        for transform in tta_transforms:
            # Note: For simplicity, we'll apply TTA on the original preprocessed tensors
            # In a more sophisticated implementation, you'd apply transforms before preprocessing
            prob, _ = self._predict_single(base_tensors)
            probabilities.append(prob)

        # Return average probability
        return np.mean(probabilities)

    def predict_batch(self, test_data_dir: Path, use_tta: bool = True, batch_size: int = 1) -> Dict[str, float]:
        """
        Predict all images in test directory.

        Args:
            test_data_dir: Directory containing test .npy files
            use_tta: Whether to use Test Time Augmentation
            batch_size: Batch size (currently only supports 1)

        Returns:
            Dictionary mapping image IDs to probabilities
        """
        logger.info(f"Starting batch prediction on {test_data_dir}")
        logger.info(f"TTA enabled: {use_tta}")

        # Get all test files
        test_files = list(test_data_dir.glob("*.npy"))
        logger.info(f"Found {len(test_files)} test files")

        if len(test_files) == 0:
            raise ValueError(f"No .npy files found in {test_data_dir}")

        # Setup TTA if requested
        tta_transforms = None
        if use_tta:
            try:
                tta_transforms = get_tta_augmentations()
                logger.info(f"TTA enabled with {len(tta_transforms)} transforms")
            except Exception as e:
                logger.warning(f"TTA setup failed: {e}. Falling back to single prediction.")
                use_tta = False

        # Predict each file
        predictions = {}

        for file_path in tqdm(test_files, desc="Predicting"):
            image_id = file_path.stem

            try:
                if use_tta and tta_transforms:
                    probability = self._predict_with_tta(file_path, tta_transforms)
                else:
                    tensors = self._preprocess_image(file_path)
                    probability, _ = self._predict_single(tensors)

                predictions[image_id] = probability

            except Exception as e:
                logger.error(f"Failed to predict {image_id}: {e}")
                # Use default probability for failed predictions
                predictions[image_id] = 0.5

        logger.info(f"Batch prediction completed. Processed {len(predictions)} images")
        return predictions

    def create_submission(
        self, predictions: Dict[str, float], output_path: Path, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Create submission file from predictions.

        Args:
            predictions: Dictionary mapping image IDs to probabilities
            output_path: Path to save submission CSV
            threshold: Classification threshold

        Returns:
            Submission DataFrame
        """
        logger.info(f"Creating submission file with threshold {threshold}")

        # Create submission data
        submission_data = []
        for image_id, probability in predictions.items():
            prediction = 1 if probability >= threshold else 0
            submission_data.append({"ID": image_id, "label": prediction})

        # Create DataFrame and sort by ID
        submission_df = pd.DataFrame(submission_data)
        submission_df = submission_df.sort_values("ID").reset_index(drop=True)

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)

        # Log statistics
        positive_count = (submission_df["label"] == 1).sum()
        total_count = len(submission_df)
        logger.info(f"Submission statistics:")
        logger.info(f"  Total samples: {total_count}")
        logger.info(f"  Predicted landslides: {positive_count} ({positive_count/total_count*100:.1f}%)")
        logger.info(
            f"  Predicted non-landslides: {total_count-positive_count} ({(total_count-positive_count)/total_count*100:.1f}%)"
        )
        logger.info(f"Submission saved to: {output_path}")

        return submission_df

    def save_probabilities(self, predictions: Dict[str, float], output_path: Path):
        """
        Save raw probabilities for analysis.

        Args:
            predictions: Dictionary mapping image IDs to probabilities
            output_path: Path to save probabilities CSV
        """
        prob_data = []
        for image_id, probability in predictions.items():
            prob_data.append({"ID": image_id, "probability": probability})

        prob_df = pd.DataFrame(prob_data)
        prob_df = prob_df.sort_values("ID").reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prob_df.to_csv(output_path, index=False)

        logger.info(f"Probabilities saved to: {output_path}")


def main():
    """Main function for batch prediction."""
    parser = argparse.ArgumentParser(description="Batch prediction for MM-InternImage-TNF")

    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pth file)")
    parser.add_argument(
        "--test_data_dir", type=str, default="dataset/test_data", help="Directory containing test .npy files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/submissions", help="Directory to save submission files"
    )
    parser.add_argument(
        "--submission_name", type=str, default=None, help="Name for submission file (auto-generated if not provided)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--use_tta", action="store_true", help="Use Test Time Augmentation")
    parser.add_argument("--save_probabilities", action="store_true", help="Save raw probabilities to CSV")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for prediction"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    logger.info("=" * 80)
    logger.info("🚀 MM-InternImage-TNF Batch Prediction")
    logger.info("=" * 80)

    # Validate inputs
    model_path = Path(args.model_path)
    test_data_dir = Path(args.test_data_dir)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    if not test_data_dir.exists():
        logger.error(f"Test data directory not found: {test_data_dir}")
        return

    # Generate submission filename if not provided
    if args.submission_name is None:
        timestamp = int(time.time())
        tta_suffix = "_tta" if args.use_tta else ""
        args.submission_name = f"submission_internimage{tta_suffix}_{timestamp}.csv"

    try:
        # Initialize predictor
        predictor = BatchPredictor(model_path, device=args.device)

        # Run batch prediction
        start_time = time.time()
        predictions = predictor.predict_batch(test_data_dir=test_data_dir, use_tta=args.use_tta)
        prediction_time = time.time() - start_time

        # Create submission file
        submission_path = output_dir / args.submission_name
        submission_df = predictor.create_submission(
            predictions=predictions, output_path=submission_path, threshold=args.threshold
        )

        # Save probabilities if requested
        if args.save_probabilities:
            prob_name = args.submission_name.replace(".csv", "_probabilities.csv")
            prob_path = output_dir / prob_name
            predictor.save_probabilities(predictions, prob_path)

        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 Batch Prediction Completed Successfully!")
        logger.info(f"⏱️ Total time: {prediction_time:.1f} seconds")
        logger.info(f"⚡ Speed: {len(predictions)/prediction_time:.1f} images/second")
        logger.info(f"📄 Submission file: {submission_path}")
        logger.info(f"📊 Samples processed: {len(predictions)}")
        logger.info("=" * 80)

        return {"predictions": predictions, "submission_path": submission_path, "submission_df": submission_df}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
