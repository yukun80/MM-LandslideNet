"""
Multi-Modal Landslide Dataset Module

This module implements the data loading and preprocessing pipeline for the
MM-InternImage-TNF model. It handles 12-channel remote sensing data with
independent normalization for each modality.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    # Try relative import first (when imported as part of package)
    from .config import config
except ImportError:
    # Fall back to absolute import (when run directly)
    from config import config


class MultiModalLandslideDataset(Dataset):
    """
    Multi-modal dataset for landslide detection combining Sentinel-1 SAR and Sentinel-2 optical data.

    The dataset handles 12-channel input data and splits it into three modalities:
    - Optical: 5 channels (R, G, B, NIR, NDVI)
    - SAR: 4 channels (VV_desc, VH_desc, VV_asc, VH_asc)
    - SAR_diff: 4 channels (VV_desc_diff, VH_desc_diff, VV_asc_diff, VH_asc_diff)

    Each modality is normalized independently using pre-computed statistics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Union[str, Path],
        stats_dict: Optional[Dict] = None,
        exclude_ids: Optional[List[str]] = None,
        augmentations: Optional[A.Compose] = None,
        mode: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame containing ID and label columns
            data_dir: Path to directory containing .npy files
            stats_dict: Dictionary containing normalization statistics
            exclude_ids: List of IDs to exclude from dataset
            augmentations: Albumentations augmentation pipeline
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.augmentations = augmentations

        # Filter out excluded IDs
        if exclude_ids:
            df = df[~df["ID"].isin(exclude_ids)]

        self.df = df.reset_index(drop=True)

        # Load normalization statistics
        if stats_dict is None:
            stats_dict = self._load_stats()
        self.stats = self._extract_normalization_stats(stats_dict)

        print(f"Dataset initialized: {len(self.df)} samples in {mode} mode")

    def _load_stats(self) -> Dict:
        """Load channel statistics from JSON file"""
        with open(config.STATS_FILE_PATH, "r") as f:
            return json.load(f)

    def _extract_normalization_stats(self, stats_dict: Dict) -> Dict:
        """
        Extract and organize normalization statistics for each modality.
        NOTE: Optical stats are for 4 channels (R,G,B,NIR). NDVI is handled separately.

        Returns:
            Dictionary with mean and std for each modality
        """
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

    def _calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI (Normalized Difference Vegetation Index).

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            red: Red channel (channel 0)
            nir: Near-infrared channel (channel 3)

        Returns:
            NDVI array with values clipped to [-1, 1]
        """
        # Avoid division by zero
        denominator = nir + red
        denominator = np.where(denominator == 0, 1e-8, denominator)

        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1, 1)

    def _split_modalities(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        ndvi = self._calculate_ndvi(red, nir)

        # Stack optical channels with NDVI
        optical = np.dstack([optical_raw, ndvi])  # Shape: (H, W, 5)

        return optical, sar_combined

    def _normalize_modality(self, data: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply Z-score normalization to a modality using pre-computed statistics.
        For optical data, normalizes the first 4 channels and clips NDVI.

        Args:
            data: Input data of shape (H, W, C)
            modality: Modality name ('optical', 'sar', 'sar_diff')

        Returns:
            Normalized data
        """
        # For optical data, handle 4-ch normalization + NDVI clipping separately
        if modality == "optical" and data.shape[2] == 5:
            normalized_data = data.copy()

            # Normalize first 4 channels (R, G, B, NIR)
            mean = self.stats[modality]["mean"]
            std = self.stats[modality]["std"]
            std = np.where(std < 1e-6, 1.0, std)  # Prevent division by zero or small numbers

            normalized_data[:, :, :4] = (normalized_data[:, :, :4] - mean) / std

            # Clip the 5th channel (NDVI) to [-1, 1] range
            normalized_data[:, :, 4] = np.clip(normalized_data[:, :, 4], -1, 1)

            return normalized_data.astype(np.float32)

        # For other modalities, apply standard Z-score normalization
        mean = self.stats[modality]["mean"]
        std = self.stats[modality]["std"]
        std = np.where(std < 1e-6, 1.0, std)  # Prevent division by zero or small numbers

        normalized = (data - mean) / std
        return normalized.astype(np.float32)

    def _apply_augmentations(self, **data_dict) -> Dict:
        """Apply augmentations to all modalities consistently"""
        if self.augmentations is not None:
            # Apply same augmentation to all modalities
            augmented = self.augmentations(**data_dict)
            return augmented
        else:
            # Convert to tensors
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    result[key] = torch.from_numpy(value).permute(2, 0, 1)  # HWC -> CHW
                else:
                    result[key] = value
            return result

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing optical, sar tensors and label
        """
        # Get sample info
        sample = self.df.iloc[idx]
        sample_id = sample["ID"]
        label = sample["label"]

        # Load 12-channel data
        data_path = self.data_dir / f"{sample_id}.npy"
        try:
            data = np.load(data_path)  # Shape: (64, 64, 12)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Split into modalities
        optical, sar_combined = self._split_modalities(data)

        # Apply independent normalization
        optical_norm = self._normalize_modality(optical, "optical")
        sar_norm = self._normalize_modality(sar_combined, "sar_combined")

        # Add assertions to check for NaN/Inf values after normalization
        assert not np.isnan(optical_norm).any(), f"NaN found in optical data for sample {sample_id}"
        assert not np.isinf(optical_norm).any(), f"Inf found in optical data for sample {sample_id}"
        assert not np.isnan(sar_norm).any(), f"NaN found in SAR data for sample {sample_id}"
        assert not np.isinf(sar_norm).any(), f"Inf found in SAR data for sample {sample_id}"

        # Prepare data for augmentation
        data_dict = {"image": optical_norm, "sar": sar_norm}

        # Apply augmentations and convert to tensors
        augmented = self._apply_augmentations(**data_dict)

        return {
            "optical": augmented["image"],
            "sar": augmented["sar"],
            "label": torch.tensor(label, dtype=torch.float32),
            "id": sample_id,
        }


def get_augmentations(mode: str = "train") -> Optional[A.Compose]:
    """
    Get augmentation pipeline for training or validation.

    Args:
        mode: 'train' or 'val'

    Returns:
        Albumentations composition or None for validation
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=config.AUGMENTATION_CONFIG["horizontal_flip_prob"]),
                A.VerticalFlip(p=config.AUGMENTATION_CONFIG["vertical_flip_prob"]),
                A.Rotate(limit=config.AUGMENTATION_CONFIG["rotation_limit"], p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=config.AUGMENTATION_CONFIG["shift_limit"],
                    scale_limit=config.AUGMENTATION_CONFIG["scale_limit"],
                    rotate_limit=0,  # Already handled by Rotate
                    p=0.5,
                ),
                ToTensorV2(),
            ],
            additional_targets={"sar": "image"},
        )
    else:
        return A.Compose([ToTensorV2()], additional_targets={"sar": "image"})


def load_exclude_ids() -> List[str]:
    """Load list of excluded image IDs"""
    try:
        with open(config.EXCLUDE_FILE_PATH, "r") as f:
            exclude_data = json.load(f)
            return exclude_data.get("excluded_image_ids", [])
    except FileNotFoundError:
        print(f"Warning: Exclude file not found at {config.EXCLUDE_FILE_PATH}")
        return []


def create_datasets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, data_dir: Union[str, Path]
) -> Tuple[MultiModalLandslideDataset, MultiModalLandslideDataset]:
    """
    Create training and validation datasets.

    Args:
        train_df: Training DataFrame (already filtered)
        val_df: Validation DataFrame (already filtered)
        data_dir: Path to data directory

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create datasets (data already filtered, so no exclude_ids needed)
    train_dataset = MultiModalLandslideDataset(
        df=train_df, data_dir=data_dir, exclude_ids=[], augmentations=get_augmentations("train"), mode="train"
    )

    val_dataset = MultiModalLandslideDataset(
        df=val_df, data_dir=data_dir, exclude_ids=[], augmentations=get_augmentations("val"), mode="val"
    )

    return train_dataset, val_dataset
