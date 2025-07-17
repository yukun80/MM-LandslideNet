import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
from typing import Dict, Tuple

# Use relative imports for consistency
from .config import config
from .models import create_optical_dominated_model
from .utils import load_checkpoint

def load_normalization_stats(stats_path: Path) -> Dict:
    with open(stats_path, "r") as f:
        full_stats = json.load(f)
    
    stats = {
        "optical": {"mean": np.array([s['mean'] for s in full_stats['channel_statistics_by_group']['optical']['channels'].values()]),
                     "std": np.array([s['std'] for s in full_stats['channel_statistics_by_group']['optical']['channels'].values()])},
        "sar": {"mean": np.array([full_stats['channel_statistics_by_group']['sar_descending']['channels']['channel_4']['mean'],
                                  full_stats['channel_statistics_by_group']['sar_descending']['channels']['channel_5']['mean'],
                                  full_stats['channel_statistics_by_group']['sar_ascending']['channels']['channel_8']['mean'],
                                  full_stats['channel_statistics_by_group']['sar_ascending']['channels']['channel_9']['mean']]),
                  "std": np.array([full_stats['channel_statistics_by_group']['sar_descending']['channels']['channel_4']['std'],
                                 full_stats['channel_statistics_by_group']['sar_descending']['channels']['channel_5']['std'],
                                 full_stats['channel_statistics_by_group']['sar_ascending']['channels']['channel_8']['std'],
                                 full_stats['channel_statistics_by_group']['sar_ascending']['channels']['channel_9']['std']])},
        "sar_change": {"mean": np.array([full_stats['channel_statistics_by_group']['sar_desc_diff']['channels']['channel_6']['mean'],
                                        full_stats['channel_statistics_by_group']['sar_desc_diff']['channels']['channel_7']['mean'],
                                        full_stats['channel_statistics_by_group']['sar_asc_diff']['channels']['channel_10']['mean'],
                                        full_stats['channel_statistics_by_group']['sar_asc_diff']['channels']['channel_11']['mean']]),
                       "std": np.array([full_stats['channel_statistics_by_group']['sar_desc_diff']['channels']['channel_6']['std'],
                                      full_stats['channel_statistics_by_group']['sar_desc_diff']['channels']['channel_7']['std'],
                                      full_stats['channel_statistics_by_group']['sar_asc_diff']['channels']['channel_10']['std'],
                                      full_stats['channel_statistics_by_group']['sar_asc_diff']['channels']['channel_11']['std']])}
    }
    return stats

def preprocess_single_image(image_path: Path, stats: Dict) -> Dict[str, torch.Tensor]:
    data = np.load(image_path).astype(np.float32)
    if data.shape != (64, 64, 12):
        raise ValueError(f"Expected shape (64, 64, 12), got {data.shape}")

    # --- Modality Splitting (matches dataset.py) ---
    optical_raw = data[:, :, 0:4]
    sar = data[:, :, [4, 5, 8, 9]]
    sar_change = data[:, :, [6, 7, 10, 11]]

    # --- NDVI Calculation ---
    red, nir = optical_raw[:, :, 0], optical_raw[:, :, 3]
    denominator = nir + red
    denominator = np.where(denominator == 0, 1e-8, denominator)
    ndvi = np.clip((nir - red) / denominator, -1, 1)
    optical = np.dstack([optical_raw, ndvi])

    # --- Normalization ---
    def normalize(d, s):
        std = np.where(s['std'] < 1e-6, 1.0, s['std'])
        return (d - s['mean']) / std

    optical_norm = normalize(optical, stats['optical'])
    sar_norm = normalize(sar, stats['sar'])
    sar_change_norm = normalize(sar_change, stats['sar_change'])

    # --- Tensor Conversion ---
    def to_tensor(d):
        return torch.from_numpy(d).permute(2, 0, 1).unsqueeze(0)

    return {
        "optical": to_tensor(optical_norm.astype(np.float32)),
        "sar": to_tensor(sar_norm.astype(np.float32)),
        "sar_change": to_tensor(sar_change_norm.astype(np.float32)),
    }

def predict(model: torch.nn.Module, image_tensors: Dict[str, torch.Tensor], device: str) -> Tuple[float, float]:
    model.eval()
    batch = {k: v.to(device) for k, v in image_tensors.items()}
    with torch.no_grad():
        logits = model(batch)
        probability = torch.sigmoid(logits).item()
    return probability, logits.item()

def load_trained_model(model_path: Path, device: str) -> torch.nn.Module:
    model = create_optical_dominated_model(num_classes=config.NUM_CLASSES, pretrained=False)
    model.to(device)
    load_checkpoint(filepath=model_path, model=model, device=device)
    print(f"Model loaded from: {model_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Inference for Optical-Dominated Cooperative Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a single .npy image file")
    parser.add_argument("--stats_path", type=str, default=config.STATS_FILE_PATH, help="Path to channel_stats.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    stats = load_normalization_stats(Path(args.stats_path))
    model = load_trained_model(Path(args.model_path), device)
    image_tensors = preprocess_single_image(Path(args.image_path), stats)
    
    probability, logit = predict(model, image_tensors, device)
    prediction = 1 if probability >= args.threshold else 0

    print("\n--- Prediction Results ---")
    print(f"Image: {args.image_path}")
    print(f"Predicted Label: {prediction} ({'Landslide' if prediction == 1 else 'No Landslide'})")
    print(f"Probability: {probability:.4f}")
    print(f"Logit: {logit:.4f}")
    print("------------------------")

if __name__ == "__main__":
    main()