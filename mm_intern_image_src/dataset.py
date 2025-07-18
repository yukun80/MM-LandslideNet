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
import cv2

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

    def _extract_normalization_stats(self, full_stats: Dict) -> Dict:
        """
        Extract statistics for three-branch normalization.

        Args:
            full_stats: Complete statistics dictionary from JSON file

        Returns:
            Dictionary with normalization statistics for each modality
        """
        try:
            # Extract statistics for optical modality (channels 0-3, plus computed NDVI)
            optical_stats = {
                "mean": np.array(
                    [
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_0"]["mean"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_1"]["mean"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_2"]["mean"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_3"]["mean"],
                        0.0,  # NDVI mean will be computed dynamically
                    ]
                ),
                "std": np.array(
                    [
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_0"]["std"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_1"]["std"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_2"]["std"],
                        full_stats["channel_statistics_by_group"]["optical"]["channels"]["channel_3"]["std"],
                        1.0,  # NDVI std will be computed dynamically
                    ]
                ),
            }

            # Extract statistics for SAR modality (channels 4-5, 8-9)
            sar_stats = {
                "mean": np.array(
                    [
                        full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_4"]["mean"],
                        full_stats["channel_statistics_by_group"]["sar_descending"]["channels"]["channel_5"]["mean"],
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
            }

            # Extract statistics for SAR change modality (channels 6-7, 10-11)
            sar_change_stats = {
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
            }

            return {"optical": optical_stats, "sar": sar_stats, "sar_change": sar_change_stats}

        except KeyError as e:
            print(f"Warning: Could not extract statistics for key {e}")
            # Return default statistics if extraction fails
            return {
                "optical": {"mean": np.zeros(5), "std": np.ones(5)},
                "sar": {"mean": np.zeros(4), "std": np.ones(4)},
                "sar_change": {"mean": np.zeros(4), "std": np.ones(4)},
            }

    def _split_modalities(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split 12-channel data into three modalities.

        Args:
            data: Input data of shape (64, 64, 12)

        Returns:
            Tuple of (optical, sar, sar_change) arrays
        """
        # Optical: channels 0-3 (R, G, B, NIR)
        optical = data[:, :, :4]

        # SAR: channels 4-5 (desc) and 8-9 (asc)
        sar = np.concatenate([data[:, :, 4:6], data[:, :, 8:10]], axis=-1)

        # SAR change: channels 6-7 (desc diff) and 10-11 (asc diff)
        sar_change = np.concatenate([data[:, :, 6:8], data[:, :, 10:12]], axis=-1)

        return optical, sar, sar_change

    def _compute_ndvi(self, optical_data: np.ndarray) -> np.ndarray:
        """
        Compute NDVI from optical data.

        Args:
            optical_data: Optical data with shape (64, 64, 4) - R, G, B, NIR

        Returns:
            NDVI array with shape (64, 64)
        """
        red = optical_data[:, :, 0]
        nir = optical_data[:, :, 3]

        # Compute NDVI with epsilon to avoid division by zero
        ndvi = (nir - red) / (nir + red + 1e-8)

        # Clip to valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)

        return ndvi

    def _normalize_modality(self, data: np.ndarray, modality: str) -> np.ndarray:
        """
        Normalize data for a specific modality.

        Args:
            data: Input data array
            modality: 'optical', 'sar', or 'sar_change'

        Returns:
            Normalized data array
        """
        stats = self.stats[modality]
        normalized = (data - stats["mean"]) / stats["std"]
        return normalized.astype(np.float32)

    def _apply_augmentations(self, **data_dict) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to the data.

        Args:
            **data_dict: Dictionary containing 'image', 'sar', 'sar_change' arrays

        Returns:
            Dictionary with augmented tensors
        """
        if self.augmentations is not None:
            augmented = self.augmentations(**data_dict)
            return {"image": augmented["image"], "sar": augmented["sar"], "sar_change": augmented["sar_change"]}
        else:
            # Convert to tensors without augmentations
            return {
                "image": torch.from_numpy(data_dict["image"].transpose(2, 0, 1)),
                "sar": torch.from_numpy(data_dict["sar"].transpose(2, 0, 1)),
                "sar_change": torch.from_numpy(data_dict["sar_change"].transpose(2, 0, 1)),
            }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing tensors for each modality and label
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

        # Split into three modalities
        optical, sar, sar_change = self._split_modalities(data)

        # Compute NDVI and add to optical data
        ndvi = self._compute_ndvi(optical)
        optical_with_ndvi = np.concatenate([optical, ndvi[:, :, np.newaxis]], axis=-1)

        # Apply independent normalization
        optical_norm = self._normalize_modality(optical_with_ndvi, "optical")
        sar_norm = self._normalize_modality(sar, "sar")
        sar_change_norm = self._normalize_modality(sar_change, "sar_change")

        # Prepare data for augmentation
        data_dict = {"image": optical_norm, "sar": sar_norm, "sar_change": sar_change_norm}

        # Apply augmentations and convert to tensors
        augmented = self._apply_augmentations(**data_dict)

        return {
            "optical": augmented["image"],
            "sar": augmented["sar"],
            "sar_change": augmented["sar_change"],
            "label": torch.tensor(label, dtype=torch.float32),
            "id": sample_id,
        }


def get_augmentations(mode: str = "train") -> Optional[A.Compose]:
    """
    Get augmentation pipeline for three-branch architecture.

    Args:
        mode: 'train', 'val', or 'test'

    Returns:
        Albumentations composition pipeline
    """
    if mode == "train":
        augmentations = [
            # 基础几何变换
            A.HorizontalFlip(p=config.AUGMENTATION_CONFIG["horizontal_flip_prob"]),
            A.VerticalFlip(p=config.AUGMENTATION_CONFIG["vertical_flip_prob"]),
            A.Rotate(limit=config.AUGMENTATION_CONFIG["rotation_limit"], p=0.6, border_mode=cv2.BORDER_REFLECT_101),
            # 修复：移除无效的mode参数
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.5),  # 10% scale variation  # 10% translation
            # 光学增强
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            # 使用GaussianBlur代替有问题的GaussNoise
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # 转换为tensor
            ToTensorV2(),
        ]

        return A.Compose(augmentations, additional_targets={"sar": "image", "sar_change": "image"})

    else:
        # 验证和测试只做基本转换
        return A.Compose([ToTensorV2()], additional_targets={"sar": "image", "sar_change": "image"})


def get_tta_augmentations() -> List[A.Compose]:
    """
    Get Test Time Augmentation (TTA) pipelines.

    Returns:
        List of augmentation pipelines for TTA
    """
    tta_transforms = [
        # Original
        A.Compose([ToTensorV2()], additional_targets={"sar": "image", "sar_change": "image"}),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()], additional_targets={"sar": "image", "sar_change": "image"}),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()], additional_targets={"sar": "image", "sar_change": "image"}),
        # Both flips
        A.Compose(
            [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), ToTensorV2()],
            additional_targets={"sar": "image", "sar_change": "image"},
        ),
        # Rotation 90°
        A.Compose(
            [A.Rotate(limit=(90, 90), p=1.0), ToTensorV2()], additional_targets={"sar": "image", "sar_change": "image"}
        ),
        # Rotation 180°
        A.Compose(
            [A.Rotate(limit=(180, 180), p=1.0), ToTensorV2()],
            additional_targets={"sar": "image", "sar_change": "image"},
        ),
        # Rotation 270°
        A.Compose(
            [A.Rotate(limit=(270, 270), p=1.0), ToTensorV2()],
            additional_targets={"sar": "image", "sar_change": "image"},
        ),
    ]

    return tta_transforms


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


# Add validation for augmentation parameters
def validate_augmentation_config():
    """Validate augmentation configuration parameters."""
    aug_config = config.AUGMENTATION_CONFIG

    # Check probability values
    for key in ["horizontal_flip_prob", "vertical_flip_prob"]:
        if not 0 <= aug_config[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1, got {aug_config[key]}")

    # Check limit values
    if aug_config["rotation_limit"] < 0:
        raise ValueError(f"rotation_limit must be non-negative, got {aug_config['rotation_limit']}")

    if not 0 <= aug_config["shift_limit"] <= 1:
        raise ValueError(f"shift_limit must be between 0 and 1, got {aug_config['shift_limit']}")

    if not 0 <= aug_config["scale_limit"] <= 1:
        raise ValueError(f"scale_limit must be between 0 and 1, got {aug_config['scale_limit']}")

    print("✅ Augmentation configuration validated successfully")


# Call validation when module is imported
try:
    validate_augmentation_config()
except Exception as e:
    print(f"⚠️ Augmentation config validation warning: {e}")
