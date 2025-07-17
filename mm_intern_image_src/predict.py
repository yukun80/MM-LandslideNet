"""
Inference Script for MM-InternImage-TNF

This module provides standalone inference functionality for the trained
multi-modal landslide detection model. It can process single .npy images
and return prediction probabilities.
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    # Try relative imports first (when imported as part of package)
    from .config import config
    from .models import create_model
    from .utils import load_checkpoint
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import config
    from models import create_model
    from utils import load_checkpoint


def load_normalization_stats(stats_path: Path) -> Dict:
    """
    Load normalization statistics from JSON file.

    Args:
        stats_path: Path to channel statistics JSON file

    Returns:
        Dictionary containing normalization statistics
    """
    with open(stats_path, "r") as f:
        stats_dict = json.load(f)

    stats = {
        "optical": {"mean": [], "std": []},
        "sar_combined": {"mean": [], "std": []},
    }

    # Extract optical statistics (channels 0-3 for R, G, B, NIR)
    optical_channels = stats_dict["channel_statistics_by_group"]["optical"]["channels"]
    for i in range(4):
        channel_key = f"channel_{i}"
        stats["optical"]["mean"].append(optical_channels[channel_key]["mean"])
        stats["optical"]["std"].append(optical_channels[channel_key]["std"])

    # Extract and combine all SAR statistics
    sar_desc = stats_dict["channel_statistics_by_group"]["sar_descending"]["channels"]
    sar_asc = stats_dict["channel_statistics_by_group"]["sar_ascending"]["channels"]
    sar_desc_diff = stats_dict["channel_statistics_by_group"]["sar_desc_diff"]["channels"]
    sar_asc_diff = stats_dict["channel_statistics_by_group"]["sar_asc_diff"]["channels"]

    # Order must match the concatenation in _split_modalities: sar, then sar_diff
    # sar = [4, 5, 8, 9], sar_diff = [6, 7, 10, 11]
    # Combined order: [4, 5, 8, 9, 6, 7, 10, 11]

    combined_sar_mean = [
        sar_desc["channel_4"]["mean"],
        sar_desc["channel_5"]["mean"],
        sar_asc["channel_8"]["mean"],
        sar_asc["channel_9"]["mean"],
        sar_desc_diff["channel_6"]["mean"],
        sar_desc_diff["channel_7"]["mean"],
        sar_asc_diff["channel_10"]["mean"],
        sar_asc_diff["channel_11"]["mean"],
    ]
    combined_sar_std = [
        sar_desc["channel_4"]["std"],
        sar_desc["channel_5"]["std"],
        sar_asc["channel_8"]["std"],
        sar_asc["channel_9"]["std"],
        sar_desc_diff["channel_6"]["std"],
        sar_desc_diff["channel_7"]["std"],
        sar_asc_diff["channel_10"]["std"],
        sar_asc_diff["channel_11"]["std"],
    ]

    stats["sar_combined"]["mean"].extend(combined_sar_mean)
    stats["sar_combined"]["std"].extend(combined_sar_std)

    # Convert to numpy arrays
    for modality in stats:
        stats[modality]["mean"] = np.array(stats[modality]["mean"], dtype=np.float32)
        stats[modality]["std"] = np.array(stats[modality]["std"], dtype=np.float32)

    return stats


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI (Normalized Difference Vegetation Index).

    Args:
        red: Red channel array
        nir: Near-infrared channel array

    Returns:
        NDVI array clipped to [-1, 1]
    """
    denominator = nir + red
    denominator = np.where(denominator == 0, 1e-8, denominator)
    ndvi = (nir - red) / denominator
    return np.clip(ndvi, -1, 1)


def split_and_process_modalities(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split 12-channel data into two modalities: 5-ch optical and 8-ch SAR.

    Args:
        data: Input array of shape (H, W, 12)

    Returns:
        Tuple of (optical, sar_combined) arrays
    """
    # Extract channels
    optical_raw = data[:, :, 0:4]  # R, G, B, NIR
    sar = data[:, :, [4, 5, 8, 9]]  # VV_desc, VH_desc, VV_asc, VH_asc
    sar_diff = data[:, :, [6, 7, 10, 11]]  # Diff channels

    # Combine all SAR channels
    sar_combined = np.dstack([sar, sar_diff])  # Shape: (H, W, 8)

    # Calculate NDVI and append to optical channels
    red = optical_raw[:, :, 0]
    nir = optical_raw[:, :, 3]
    ndvi = calculate_ndvi(red, nir)

    # Stack optical channels with NDVI
    optical = np.dstack([optical_raw, ndvi])  # Shape: (H, W, 5)

    return optical, sar_combined


def normalize_modality(data: np.ndarray, modality: str, stats: Dict) -> np.ndarray:
    """
    Apply Z-score normalization to a modality using pre-computed statistics.
    For optical data, normalizes the first 4 channels and clips NDVI.

    Args:
        data: Input data of shape (H, W, C)
        modality: Modality name ('optical', 'sar_combined')
        stats: Normalization statistics dictionary

    Returns:
        Normalized data
    """
    # For optical data, handle 4-ch normalization + NDVI clipping separately
    if modality == "optical" and data.shape[2] == 5:
        normalized_data = data.copy()

        # Normalize first 4 channels (R, G, B, NIR)
        mean = stats[modality]["mean"]
        std = stats[modality]["std"]
        std = np.where(std < 1e-6, 1.0, std)  # Prevent division by zero or small numbers

        normalized_data[:, :, :4] = (normalized_data[:, :, :4] - mean) / std

        # Clip the 5th channel (NDVI) to [-1, 1] range
        normalized_data[:, :, 4] = np.clip(normalized_data[:, :, 4], -1, 1)

        return normalized_data.astype(np.float32)

    # For other modalities, apply standard Z-score normalization
    mean = stats[modality]["mean"]
    std = stats[modality]["std"]
    std = np.where(std < 1e-6, 1.0, std)  # Prevent division by zero or small numbers

    normalized = (data - mean) / std
    return normalized.astype(np.float32)


def preprocess_single_image(image_path: Path, stats: Dict) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single .npy image for inference.

    Args:
        image_path: Path to .npy file
        stats: Normalization statistics dictionary

    Returns:
        Dictionary containing preprocessed tensors for each modality
    """
    # Load 12-channel data
    try:
        data = np.load(image_path)  # Shape: (H, W, 12)
        print(f"Loaded image: {image_path} with shape {data.shape}")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")

    # Validate input shape
    if data.shape != (64, 64, 12):
        raise ValueError(f"Expected shape (64, 64, 12), got {data.shape}")

    # Split into modalities
    optical, sar_combined = split_and_process_modalities(data)
    # print(f"Split into modalities - Optical: {optical.shape}, SAR Combined: {sar_combined.shape}")

    # Apply independent normalization
    optical_norm = normalize_modality(optical, "optical", stats)
    sar_norm = normalize_modality(sar_combined, "sar_combined", stats)

    # Convert to tensors and add batch dimension
    # Convert from HWC to CHW format
    optical_tensor = torch.from_numpy(optical_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 5, H, W]
    sar_tensor = torch.from_numpy(sar_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 8, H, W]

    return {"optical": optical_tensor, "sar": sar_tensor}


def predict(model: torch.nn.Module, image_tensors: Dict[str, torch.Tensor], device: str) -> Tuple[float, float]:
    """
    Run inference on preprocessed image tensors.

    Args:
        model: Trained model
        image_tensors: Dictionary containing preprocessed tensors
        device: Device to run inference on

    Returns:
        Tuple of (probability, logit) for landslide prediction
    """
    model.eval()

    # Move tensors to device
    for key in image_tensors:
        image_tensors[key] = image_tensors[key].to(device)

    with torch.no_grad():
        # Forward pass
        logits = model(image_tensors)

        # Convert to probability
        probability = torch.sigmoid(logits).item()
        logit = logits.item()

    return probability, logit


def load_trained_model(model_path: Path, device: str) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model architecture
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained_optical=False,  # Don't load pretrained weights when loading checkpoint
        device=device,
    )

    # Load checkpoint
    checkpoint_info = load_checkpoint(filepath=model_path, model=model, device=device)

    print(f"Model loaded from epoch {checkpoint_info['epoch']}")
    print(f"Best metric from training: {checkpoint_info['best_metric']:.4f}")

    return model


def main():
    """Main inference function with argument parsing."""
    parser = argparse.ArgumentParser(description="MM-InternImage-TNF Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pth file)")
    parser.add_argument(
        "--image_path", type=str, help="Path to single input .npy image file for individual prediction."
    )
    parser.add_argument(
        "--test_data_dir", type=str, help="Path to directory containing test .npy files for batch prediction."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to save the submission CSV file for batch prediction. E.g., outputs/submissions/submission.csv",
    )
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to channel statistics JSON file (default: use config path)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for inference (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    if not args.image_path and not args.test_data_dir:
        raise ValueError("Either --image_path or --test_data_dir must be provided.")
    if args.image_path and args.test_data_dir:
        raise ValueError("Cannot provide both --image_path and --test_data_dir. Choose one mode.")
    if args.test_data_dir and not args.output_csv:
        raise ValueError("When --test_data_dir is provided, --output_csv is required.")

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Set device
    device = args.device if args.device else config.DEVICE
    print(f"Using device: {device}")

    # Load normalization statistics
    stats_path = Path(args.stats_path) if args.stats_path else config.STATS_FILE_PATH
    print(f"Loading normalization statistics from: {stats_path}")
    stats = load_normalization_stats(stats_path)

    # Load trained model
    print(f"Loading model from: {model_path}")
    model = load_trained_model(model_path, device)

    if args.image_path:  # Single image prediction mode
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Preprocess image
        print(f"Preprocessing image: {image_path}")
        image_tensors = preprocess_single_image(image_path, stats)

        if args.verbose:
            for modality, tensor in image_tensors.items():
                print(f"{modality} tensor shape: {tensor.shape}")
                print(f"{modality} tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

        # Run inference
        print("Running inference...")
        probability, logit = predict(model, image_tensors, device)

        # Make prediction
        prediction_label = 1 if probability >= args.threshold else 0
        confidence = probability if probability >= 0.5 else (1 - probability)

        # Print results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"Image: {image_path.name}")
        print(f"Predicted Label: {prediction_label}")
        print(f"Probability: {probability:.4f}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Logit: {logit:.4f}")
        print(f"Threshold: {args.threshold}")
        print("=" * 50)

        # Return results as a dictionary for programmatic use
        return {
            "image_path": str(image_path),
            "prediction_label": prediction_label,
            "probability": probability,
            "confidence": confidence,
            "logit": logit,
            "threshold": args.threshold,
        }

    elif args.test_data_dir:  # Batch prediction mode
        test_data_dir = Path(args.test_data_dir)
        if not test_data_dir.is_dir():
            raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

        print(f"\nRunning batch inference on {test_data_dir}...")
        results_list = []
        image_files = sorted(list(test_data_dir.glob("*.npy")))

        # Import tqdm and pandas here to avoid unnecessary imports for single image mode
        from tqdm import tqdm
        import pandas as pd

        for image_path in tqdm(image_files, desc="Processing test images"):
            try:
                image_id = image_path.stem  # Get filename without extension
                image_tensors = preprocess_single_image(image_path, stats)
                probability, _ = predict(model, image_tensors, device)
                prediction_label = 1 if probability >= args.threshold else 0
                results_list.append({"ID": image_id, "Category": prediction_label})
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                # Append NaN or a default value for failed predictions, or skip
                results_list.append(
                    {"ID": image_path.stem, "Category": -1}
                )  # Use -1 to indicate error, or choose to skip

        # Save results to CSV
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv_path, index=False)

        print(f"\nBatch prediction completed. Results saved to {output_csv_path}")
        return {"output_csv_path": str(output_csv_path), "num_predictions": len(results_list)}


if __name__ == "__main__":
    results = main()
