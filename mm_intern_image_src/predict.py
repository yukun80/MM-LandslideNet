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

    # Extract and organize statistics similar to dataset.py
    stats = {"optical": {"mean": [], "std": []}, "sar": {"mean": [], "std": []}, "sar_diff": {"mean": [], "std": []}}

    # Extract optical statistics (channels 0-3)
    optical_channels = stats_dict["channel_statistics_by_group"]["optical"]["channels"]
    for i in range(4):  # R, G, B, NIR
        channel_key = f"channel_{i}"
        stats["optical"]["mean"].append(optical_channels[channel_key]["mean"])
        stats["optical"]["std"].append(optical_channels[channel_key]["std"])

    # NDVI stats (approximation)
    stats["optical"]["mean"].append(0.0)  # NDVI mean
    stats["optical"]["std"].append(1.0)  # NDVI std

    # Extract SAR statistics (channels 4-5, 8-9)
    sar_desc = stats_dict["channel_statistics_by_group"]["sar_descending"]["channels"]
    sar_asc = stats_dict["channel_statistics_by_group"]["sar_ascending"]["channels"]

    stats["sar"]["mean"].extend(
        [
            sar_desc["channel_4"]["mean"],
            sar_desc["channel_5"]["mean"],
            sar_asc["channel_8"]["mean"],
            sar_asc["channel_9"]["mean"],
        ]
    )
    stats["sar"]["std"].extend(
        [
            sar_desc["channel_4"]["std"],
            sar_desc["channel_5"]["std"],
            sar_asc["channel_8"]["std"],
            sar_asc["channel_9"]["std"],
        ]
    )

    # Extract SAR diff statistics (channels 6-7, 10-11)
    sar_desc_diff = stats_dict["channel_statistics_by_group"]["sar_desc_diff"]["channels"]
    sar_asc_diff = stats_dict["channel_statistics_by_group"]["sar_asc_diff"]["channels"]

    stats["sar_diff"]["mean"].extend(
        [
            sar_desc_diff["channel_6"]["mean"],
            sar_desc_diff["channel_7"]["mean"],
            sar_asc_diff["channel_10"]["mean"],
            sar_asc_diff["channel_11"]["mean"],
        ]
    )
    stats["sar_diff"]["std"].extend(
        [
            sar_desc_diff["channel_6"]["std"],
            sar_desc_diff["channel_7"]["std"],
            sar_asc_diff["channel_10"]["std"],
            sar_asc_diff["channel_11"]["std"],
        ]
    )

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


def split_and_process_modalities(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split 12-channel data into three modalities and calculate NDVI.

    Args:
        data: Input array of shape (H, W, 12)

    Returns:
        Tuple of (optical, sar, sar_diff) arrays
    """
    # Extract channels
    optical_raw = data[:, :, 0:4]  # R, G, B, NIR
    sar = data[:, :, [4, 5, 8, 9]]  # VV_desc, VH_desc, VV_asc, VH_asc
    sar_diff = data[:, :, [6, 7, 10, 11]]  # Diff channels

    # Calculate NDVI and append to optical channels
    red = optical_raw[:, :, 0]
    nir = optical_raw[:, :, 3]
    ndvi = calculate_ndvi(red, nir)

    # Stack optical channels with NDVI
    optical = np.dstack([optical_raw, ndvi])  # Shape: (H, W, 5)

    return optical, sar, sar_diff


def normalize_modality(data: np.ndarray, modality: str, stats: Dict) -> np.ndarray:
    """
    Apply Z-score normalization to a modality using pre-computed statistics.

    Args:
        data: Input data of shape (H, W, C)
        modality: Modality name ('optical', 'sar', 'sar_diff')
        stats: Normalization statistics dictionary

    Returns:
        Normalized data
    """
    mean = stats[modality]["mean"]
    std = stats[modality]["std"]

    # Ensure std is not zero
    std = np.where(std == 0, 1.0, std)

    # Apply normalization
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
    optical, sar, sar_diff = split_and_process_modalities(data)
    print(f"Split into modalities - Optical: {optical.shape}, SAR: {sar.shape}, SAR_diff: {sar_diff.shape}")

    # Apply independent normalization
    optical_norm = normalize_modality(optical, "optical", stats)
    sar_norm = normalize_modality(sar, "sar", stats)
    sar_diff_norm = normalize_modality(sar_diff, "sar_diff", stats)

    # Convert to tensors and add batch dimension
    # Convert from HWC to CHW format
    optical_tensor = torch.from_numpy(optical_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 5, H, W]
    sar_tensor = torch.from_numpy(sar_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
    sar_diff_tensor = torch.from_numpy(sar_diff_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]

    return {"optical": optical_tensor, "sar": sar_tensor, "sar_diff": sar_diff_tensor}


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
    parser.add_argument("--image_path", type=str, required=True, help="Path to input .npy image file")
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to channel statistics JSON file (default: use config path)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for inference (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

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
    prediction = "Landslide" if probability >= args.threshold else "No Landslide"
    confidence = probability if probability >= 0.5 else (1 - probability)

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Image: {image_path.name}")
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Logit: {logit:.4f}")
    print(f"Threshold: {args.threshold}")
    print("=" * 50)

    # Return results as a dictionary for programmatic use
    return {
        "image_path": str(image_path),
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence,
        "logit": logit,
        "threshold": args.threshold,
    }


if __name__ == "__main__":
    results = main()
