"""
Modified Prediction Script for MM-TNF Model

Key Changes:
1. Adapted for dual-branch TNF model inference
2. Updated data loading format (optical + sar)
3. Enhanced ensemble prediction with branch weights
4. Test-Time Augmentation (TTA) support for TNF model
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import argparse
from tqdm import tqdm
import json
from datetime import datetime

"""
python -m mm_intern_image_src.predict --model outputs/checkpoints/tnf_run_20250720_200449/best_model.pth
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import project modules
from .config import config, init_inference
from .dataset import MultiModalLandslideDataset, get_tta_augmentations
from .models import create_tnf_model
from .utils import load_checkpoint, seed_everything


class TNFPredictor:
    """
    Prediction class for TNF landslide detection model.

    Supports:
    - Standard inference
    - Test-Time Augmentation (TTA)
    - Ensemble predictions from multiple branches
    - Uncertainty estimation
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        use_tta: bool = True,
        tta_confidence_threshold: float = 0.9,
    ):
        """
        Initialize TNF predictor.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            use_tta: Whether to use Test-Time Augmentation
            tta_confidence_threshold: Threshold for TTA consensus
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = use_tta
        self.tta_confidence_threshold = tta_confidence_threshold

        # Load model
        self.model = self._load_model()
        logger.info(f"TNF Predictor initialized on {self.device}")

    def _load_model(self) -> nn.Module:
        """Load trained TNF model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        # Create model
        model = create_tnf_model(
            pretrained=False,  # We're loading trained weights
            optical_channels=5,
            sar_channels=8,
            optical_feature_dim=512,
            sar_feature_dim=512,
            fusion_dim=512,
        )

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model state dict directly")

        model = model.to(self.device)
        model.eval()

        return model

    def predict_batch(self, optical_data: torch.Tensor, sar_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict on a batch of data.

        Args:
            optical_data: Optical tensor (B, 5, 64, 64)
            sar_data: SAR tensor (B, 8, 64, 64)

        Returns:
            Dictionary with predictions and intermediate outputs
        """
        with torch.no_grad():
            # Standard prediction
            outputs = self.model(optical_data, sar_data)

            # Convert logits to probabilities
            predictions = {
                "optical_probs": torch.sigmoid(outputs["optical_logits"]),
                "sar_probs": torch.sigmoid(outputs["sar_logits"]),
                "fusion_probs": torch.sigmoid(outputs["fusion_logits"]),
                "final_probs": torch.sigmoid(outputs["final_logits"]),
                "fusion_weights": outputs["fusion_weights"],
            }

            return predictions

    def predict_with_tta(self, optical_data: torch.Tensor, sar_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with Test-Time Augmentation.

        Args:
            optical_data: Optical tensor (B, 5, 64, 64)
            sar_data: SAR tensor (B, 8, 64, 64)

        Returns:
            Dictionary with TTA predictions and uncertainty measures
        """
        tta_transforms = get_tta_augmentations()
        batch_size = optical_data.size(0)

        # Collect predictions from all TTA transforms
        all_predictions = {"optical": [], "sar": [], "fusion": [], "final": []}

        with torch.no_grad():
            for transform in tta_transforms:
                # Apply transform to batch
                augmented_optical = []
                augmented_sar = []

                for i in range(batch_size):
                    # Convert to numpy for albumentations
                    opt_np = optical_data[i].permute(1, 2, 0).cpu().numpy()
                    sar_np = sar_data[i].permute(1, 2, 0).cpu().numpy()

                    # Apply transform
                    augmented = transform(image=opt_np, sar=sar_np)

                    augmented_optical.append(augmented["image"])
                    augmented_sar.append(augmented["sar"])

                # Stack back to tensors
                aug_optical = torch.stack(augmented_optical).to(self.device)
                aug_sar = torch.stack(augmented_sar).to(self.device)

                # Get predictions
                outputs = self.model(aug_optical, aug_sar)

                # Store probabilities
                all_predictions["optical"].append(torch.sigmoid(outputs["optical_logits"]))
                all_predictions["sar"].append(torch.sigmoid(outputs["sar_logits"]))
                all_predictions["fusion"].append(torch.sigmoid(outputs["fusion_logits"]))
                all_predictions["final"].append(torch.sigmoid(outputs["final_logits"]))

        # Aggregate TTA predictions
        tta_results = {}
        for branch, predictions in all_predictions.items():
            stacked_preds = torch.stack(predictions, dim=0)  # (num_tta, batch_size, 1)

            # Mean prediction
            mean_pred = stacked_preds.mean(dim=0)

            # Uncertainty (standard deviation)
            std_pred = stacked_preds.std(dim=0)

            tta_results[f"{branch}_mean"] = mean_pred
            tta_results[f"{branch}_std"] = std_pred
            tta_results[f"{branch}_confidence"] = 1.0 - std_pred  # Higher std = lower confidence

        return tta_results

    def predict_dataset(self, dataloader: DataLoader, use_tta: Optional[bool] = None) -> Dict[str, np.ndarray]:
        """
        Predict on entire dataset.

        Args:
            dataloader: DataLoader for prediction dataset
            use_tta: Override TTA setting for this prediction

        Returns:
            Dictionary with all predictions and metadata
        """
        use_tta = use_tta if use_tta is not None else self.use_tta

        results = {
            "ids": [],
            "optical_probs": [],
            "sar_probs": [],
            "fusion_probs": [],
            "final_probs": [],
        }

        if use_tta:
            results.update({"final_std": [], "final_confidence": [], "high_confidence_mask": []})

        logger.info(f"Starting prediction with TTA={use_tta}")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Extract data
                optical_data = batch["optical"].to(self.device)
                sar_data = batch["sar"].to(self.device)
                sample_ids = batch["id"]

                if use_tta:
                    # TTA prediction
                    predictions = self.predict_with_tta(optical_data, sar_data)

                    # Store TTA results
                    results["final_probs"].extend(predictions["final_mean"].cpu().numpy())
                    results["final_std"].extend(predictions["final_std"].cpu().numpy())
                    results["final_confidence"].extend(predictions["final_confidence"].cpu().numpy())

                    # High confidence mask
                    high_conf = predictions["final_confidence"] > self.tta_confidence_threshold
                    results["high_confidence_mask"].extend(high_conf.cpu().numpy())

                    # Also store branch predictions (use mean)
                    results["optical_probs"].extend(predictions["optical_mean"].cpu().numpy())
                    results["sar_probs"].extend(predictions["sar_mean"].cpu().numpy())
                    results["fusion_probs"].extend(predictions["fusion_mean"].cpu().numpy())

                else:
                    # Standard prediction
                    predictions = self.predict_batch(optical_data, sar_data)

                    # Store results
                    results["optical_probs"].extend(predictions["optical_probs"].cpu().numpy())
                    results["sar_probs"].extend(predictions["sar_probs"].cpu().numpy())
                    results["fusion_probs"].extend(predictions["fusion_probs"].cpu().numpy())
                    results["final_probs"].extend(predictions["final_probs"].cpu().numpy())

                # Store IDs
                results["ids"].extend(sample_ids)

        # Convert lists to numpy arrays
        for key, values in results.items():
            if key != "ids":
                results[key] = np.array(values)

        logger.info(f"Prediction completed. {len(results['ids'])} samples processed.")

        return results

    def create_submission(
        self, predictions: Dict[str, np.ndarray], threshold: float = 0.5, use_branch: str = "final"
    ) -> pd.DataFrame:
        """
        Create submission file from predictions.

        Args:
            predictions: Prediction results dictionary
            threshold: Classification threshold
            use_branch: Which branch to use ("optical", "sar", "fusion", "final")

        Returns:
            Submission DataFrame
        """
        prob_key = f"{use_branch}_probs"

        if prob_key not in predictions:
            raise ValueError(f"Prediction branch '{use_branch}' not found in results")

        # Apply threshold to get binary predictions
        binary_preds = (predictions[prob_key].squeeze() > threshold).astype(int)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"ID": predictions["ids"], "label": binary_preds})

        logger.info(f"Submission created using {use_branch} branch with threshold {threshold}")
        logger.info(f"Prediction distribution: {np.bincount(binary_preds)}")

        return submission_df

    def save_detailed_results(self, predictions: Dict[str, np.ndarray], output_path: Union[str, Path]) -> None:
        """
        Save detailed prediction results.

        Args:
            predictions: Prediction results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)

        # Create detailed results DataFrame
        detailed_df = pd.DataFrame(
            {
                "ID": predictions["ids"],
                "optical_prob": predictions["optical_probs"].squeeze(),
                "sar_prob": predictions["sar_probs"].squeeze(),
                "fusion_prob": predictions["fusion_probs"].squeeze(),
                "final_prob": predictions["final_probs"].squeeze(),
            }
        )

        # Add TTA results if available
        if "final_std" in predictions:
            detailed_df["final_std"] = predictions["final_std"].squeeze()
            detailed_df["final_confidence"] = predictions["final_confidence"].squeeze()
            detailed_df["high_confidence"] = predictions["high_confidence_mask"].squeeze()

        # Save to CSV
        detailed_df.to_csv(output_path, index=False)
        logger.info(f"Detailed results saved to {output_path}")


def create_test_dataset() -> MultiModalLandslideDataset:
    """Create test dataset for prediction."""
    test_csv = config.DATA_DIR / "Test.csv"
    test_data_dir = config.TEST_DATA_DIR

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

    # Load test DataFrame
    test_df = pd.read_csv(test_csv)

    # Create test dataset (no augmentations for test)
    test_dataset = MultiModalLandslideDataset(
        df=test_df,
        data_dir=test_data_dir,
        exclude_ids=[],  # No exclusions for test set
        augmentations=None,  # No augmentations for clean prediction
        mode="test",
    )

    return test_dataset


def run_prediction(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    batch_size: int = 32,
    use_tta: bool = True,
    threshold: float = 0.5,
    use_branch: str = "final",
) -> None:
    """
    Run complete prediction pipeline.

    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory to save results
        batch_size: Batch size for inference
        use_tta: Whether to use Test-Time Augmentation
        threshold: Classification threshold
        use_branch: Which branch to use for submission
    """
    init_inference()  # 只在推理时调用
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    seed_everything(config.RANDOM_SEED)

    # Create predictor
    predictor = TNFPredictor(model_path=model_path, use_tta=use_tta)

    # Create test dataset and dataloader
    test_dataset = create_test_dataset()
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Run prediction
    predictions = predictor.predict_dataset(test_loader, use_tta=use_tta)

    # Create submission
    submission_df = predictor.create_submission(predictions, threshold=threshold, use_branch=use_branch)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save submission file
    submission_path = output_dir / f"tnf_submission_{use_branch}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved: {submission_path}")

    # Save detailed results
    detailed_path = output_dir / f"tnf_detailed_results_{timestamp}.csv"
    predictor.save_detailed_results(predictions, detailed_path)

    # Save prediction metadata
    metadata = {
        "model_path": str(model_path),
        "use_tta": use_tta,
        "threshold": threshold,
        "use_branch": use_branch,
        "num_samples": len(predictions["ids"]),
        "timestamp": timestamp,
        "prediction_summary": {
            "positive_predictions": int(submission_df["label"].sum()),
            "negative_predictions": int((submission_df["label"] == 0).sum()),
            "positive_rate": float(submission_df["label"].mean()),
        },
    }

    if use_tta:
        metadata["tta_summary"] = {
            "mean_confidence": float(predictions["final_confidence"].mean()),
            "high_confidence_samples": int(predictions["high_confidence_mask"].sum()),
            "high_confidence_rate": float(predictions["high_confidence_mask"].mean()),
        }

    metadata_path = output_dir / f"tnf_prediction_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Prediction metadata saved: {metadata_path}")
    logger.info("🎉 Prediction pipeline completed successfully!")


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(description="Run TNF model prediction")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/submissions", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument(
        "--branch",
        type=str,
        default="final",
        choices=["optical", "sar", "fusion", "final"],
        help="Which branch to use for submission",
    )

    args = parser.parse_args()

    run_prediction(
        model_path=args.model,
        output_dir=args.output,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
        threshold=args.threshold,
        use_branch=args.branch,
    )


if __name__ == "__main__":
    main()
