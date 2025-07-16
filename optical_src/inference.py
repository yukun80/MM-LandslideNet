#!/usr/bin/env python3
"""
MM-LandslideNet Optical Baseline Inference Script

This script loads the trained optical baseline model and generates predictions
for the test dataset, creating a submission file in the required format.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import optical baseline modules
from optical_src.config import OpticalBaselineConfig
from optical_src.dataset import LandslideDataset
from optical_src.model import BaselineOpticalModel
from optical_src.utils import setup_logging, get_device, ensure_dir

# Setup logger for this module
logger = logging.getLogger("optical_baseline.inference")


class OpticalInference:
    """
    Optical baseline model inference class for generating predictions on test data.
    """

    def __init__(self, config: OpticalBaselineConfig = None, model_path: str = None):
        """
        Initialize the inference class.

        Args:
            config: OpticalBaselineConfig instance
            model_path: Path to the trained model checkpoint
        """
        self.config = config if config is not None else OpticalBaselineConfig()
        self.device = get_device()

        # Model path selection
        if model_path is None:
            self.model_path = self.config.OPTICAL_CHECKPOINT_DIR / "best_model.pth"
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        logger.info(f"Using model checkpoint: {self.model_path}")

        # Initialize model
        self._load_model()

        # Load test dataset
        self._setup_test_dataset()

    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        logger.info("Loading trained model...")

        try:
            # Create model
            self.model = BaselineOpticalModel.from_config(self.config, variant="swin_tiny")

            # Load checkpoint (set weights_only=False for compatibility with PyTorch 2.6+)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Move to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Log model info
            val_metrics = checkpoint.get("val_metrics", {})
            best_f1 = checkpoint.get("best_f1", "N/A")
            epoch = checkpoint.get("epoch", "N/A")

            logger.info(f"Model loaded successfully from epoch {epoch}")
            logger.info(f"Best validation F1: {best_f1}")
            if val_metrics:
                logger.info(f"Validation metrics: {val_metrics}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _setup_test_dataset(self) -> None:
        """Setup test dataset and dataloader."""
        logger.info("Setting up test dataset...")

        try:
            # Load channel statistics for normalization (same as training)
            channel_stats_path = self.config.PROJECT_ROOT / "dataset" / "data_check" / "channel_stats.json"
            if channel_stats_path.exists():
                import json

                with open(channel_stats_path, "r") as f:
                    channel_stats = json.load(f)
                logger.info("Loaded channel statistics for normalization")
            else:
                channel_stats = None
                logger.warning("Channel statistics not found, using raw values")

            # Get test transforms (no augmentation)
            test_transforms = LandslideDataset.get_transforms(is_training=False, config=self.config)

            # Create test dataset
            self.test_dataset = LandslideDataset(
                csv_file=self.config.TEST_CSV,
                data_dir=self.config.TEST_DATA_DIR,
                exclude_ids=None,  # No exclusions for test data
                channel_stats=channel_stats,
                transforms=test_transforms,
                is_test=True,  # Important: test mode returns image_id instead of label
            )

            # Create test dataloader
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,  # Important: maintain order for submission
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,  # Keep all samples
            )

            logger.info(f"Test dataset size: {len(self.test_dataset)}")
            logger.info(f"Test loader batches: {len(self.test_loader)}")

        except Exception as e:
            logger.error(f"Error setting up test dataset: {str(e)}")
            raise

    def predict(self, use_tta: bool = False) -> Dict[str, float]:
        """
        Generate predictions for all test samples.

        Args:
            use_tta: Whether to use Test Time Augmentation

        Returns:
            Dictionary mapping image IDs to prediction probabilities
        """
        logger.info("Starting inference on test dataset...")

        predictions = {}
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (data, image_ids) in enumerate(tqdm(self.test_loader, desc="Inference")):
                data = data.to(self.device)

                if use_tta:
                    # Test Time Augmentation
                    outputs = self._predict_with_tta(data)
                else:
                    # Standard prediction
                    outputs = self.model(data)

                # Convert to probabilities
                probabilities = torch.sigmoid(outputs).squeeze()

                # Handle single sample case
                if probabilities.dim() == 0:
                    probabilities = probabilities.unsqueeze(0)
                    image_ids = [image_ids]

                # Store predictions
                for i, image_id in enumerate(image_ids):
                    predictions[image_id] = probabilities[i].cpu().item()

                # Log progress
                if batch_idx % 50 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(self.test_loader)}")

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Generated predictions for {len(predictions)} samples")

        return predictions

    def _predict_with_tta(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict with Test Time Augmentation.

        Args:
            data: Input tensor

        Returns:
            Averaged predictions
        """
        predictions = []

        # Original
        predictions.append(self.model(data))

        # Horizontal flip
        predictions.append(self.model(torch.flip(data, dims=[3])))

        # Vertical flip
        predictions.append(self.model(torch.flip(data, dims=[2])))

        # Both flips
        predictions.append(self.model(torch.flip(data, dims=[2, 3])))

        # Average all predictions
        return torch.stack(predictions).mean(dim=0)

    def create_submission(self, predictions: Dict[str, float], threshold: float = 0.5) -> pd.DataFrame:
        """
        Create submission dataframe from predictions.

        Args:
            predictions: Dictionary mapping image IDs to probabilities
            threshold: Classification threshold

        Returns:
            Submission dataframe
        """
        logger.info("Creating submission file...")

        # Load test CSV to get the correct order of IDs
        test_df = pd.read_csv(self.config.TEST_CSV)

        # Create submission data
        submission_data = []
        missing_predictions = []

        for _, row in test_df.iterrows():
            image_id = row["ID"]
            if image_id in predictions:
                probability = predictions[image_id]
                label = 1 if probability > threshold else 0
                submission_data.append({"ID": image_id, "label": label})
            else:
                # Handle missing predictions (shouldn't happen)
                missing_predictions.append(image_id)
                submission_data.append({"ID": image_id, "label": 0})  # Default to 0

        if missing_predictions:
            logger.warning(f"Missing predictions for {len(missing_predictions)} samples")

        submission_df = pd.DataFrame(submission_data)

        # Statistics
        total_samples = len(submission_df)
        positive_predictions = sum(submission_df["label"])
        positive_rate = positive_predictions / total_samples * 100

        logger.info(f"Submission statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Positive predictions: {positive_predictions} ({positive_rate:.2f}%)")
        logger.info(f"  Negative predictions: {total_samples - positive_predictions} ({100-positive_rate:.2f}%)")

        return submission_df

    def save_submission(self, submission_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save submission file.

        Args:
            submission_df: Submission dataframe
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"optical_baseline_submission_{timestamp}.csv"

        # Ensure submissions directory exists
        submission_dir = self.config.OUTPUT_ROOT / "submissions"
        ensure_dir(submission_dir)

        output_path = submission_dir / filename
        submission_df.to_csv(output_path, index=False)

        logger.info(f"Submission saved to: {output_path}")
        return str(output_path)

    def run_inference(self, use_tta: bool = False, threshold: float = 0.5, save_probabilities: bool = True) -> str:
        """
        Complete inference pipeline.

        Args:
            use_tta: Whether to use Test Time Augmentation
            threshold: Classification threshold
            save_probabilities: Whether to save raw probabilities

        Returns:
            Path to submission file
        """
        logger.info("=" * 60)
        logger.info("Starting MM-LandslideNet Optical Baseline Inference")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Generate predictions
            predictions = self.predict(use_tta=use_tta)

            # Create submission
            submission_df = self.create_submission(predictions, threshold=threshold)

            # Save submission
            submission_path = self.save_submission(submission_df)

            # Optionally save raw probabilities
            if save_probabilities:
                prob_data = [{"ID": image_id, "probability": prob} for image_id, prob in predictions.items()]
                prob_df = pd.DataFrame(prob_data)

                timestamp = int(time.time())
                prob_filename = f"optical_baseline_probabilities_{timestamp}.csv"
                prob_path = self.config.OUTPUT_ROOT / "submissions" / prob_filename
                prob_df.to_csv(prob_path, index=False)
                logger.info(f"Probabilities saved to: {prob_path}")

            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"Inference completed successfully in {total_time:.2f} seconds")
            logger.info(f"Submission file: {submission_path}")
            logger.info("=" * 60)

            return submission_path

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise


def main():
    """Main function to run inference."""
    # Setup logging
    config = OpticalBaselineConfig()
    log_file = config.OPTICAL_LOG_DIR / "inference.log"
    setup_logging(log_level="INFO", log_file=log_file)

    logger.info("Starting MM-LandslideNet Optical Baseline Inference")

    try:
        # Create inference instance
        inference = OpticalInference(config=config)

        # Run inference with TTA for better performance
        submission_path = inference.run_inference(
            use_tta=True,  # Use Test Time Augmentation
            threshold=0.5,  # Classification threshold
            save_probabilities=True,  # Save raw probabilities
        )

        print(f"\n🎉 Inference completed successfully!")
        print(f"📄 Submission file: {submission_path}")
        print(f"📊 Check logs at: {log_file}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        print(f"\n❌ Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
