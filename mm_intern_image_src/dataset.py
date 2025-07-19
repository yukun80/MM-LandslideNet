"""
Modified Multi-Modal Landslide Dataset Module for TNF Model

Key Changes:
1. Adapted data format for dual-branch TNF model
2. Optical: 5 channels (R,G,B,NIR,NDVI)
3. SAR: 8 channels (4 original + 4 difference channels combined)
4. Simplified data flow without separate sar_change branch
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
    from .config import config
except ImportError:
    from config import config


class MultiModalLandslideDataset(Dataset):
    """
    Multi-modal dataset for TNF landslide detection model.

    Data Flow:
    - Input: 12-channel data from .npy files
    - Output: optical (5ch) + sar (8ch) for dual-branch TNF model
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
        Initialize the TNF-compatible dataset.

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

        print(f"TNF Dataset initialized: {len(self.df)} samples in {mode} mode")

    def _load_stats(self) -> Dict:
        """Load channel statistics from JSON file and flatten them."""
        with open(config.STATS_FILE_PATH, "r") as f:
            stats = json.load(f)
        
        channel_means = []
        channel_stds = []
        
        # The stats are nested, so we need to extract them in the correct order
        channel_groups = stats.get("channel_statistics_by_group", {})
        
        # We need to ensure the channels are ordered correctly from 0 to 11
        all_channels = {}
        for group in channel_groups.values():
            for ch_key, ch_data in group.get("channels", {}).items():
                all_channels[ch_data["channel_index"]] = ch_data
                
        for i in range(12): # Assuming 12 channels total
            if i in all_channels:
                channel_means.append(all_channels[i]["mean"])
                channel_stds.append(all_channels[i]["std"])
            else:
                # Handle missing channels if necessary, e.g., by appending default values
                channel_means.append(0.0)
                channel_stds.append(1.0)

        return {"channel_means": channel_means, "channel_stds": channel_stds}

    def _extract_normalization_stats(self, full_stats: Dict) -> Dict:
        """
        Extract statistics for dual-branch normalization.

        Returns:
            Dict with normalization stats for optical and sar modalities
        """
        # The _load_stats function now returns the correct format, so we just need to slice it
        return {
            # Optical channels: R, G, B, NIR (0-3) + computed NDVI
            "optical": {
                "mean": np.array(full_stats["channel_means"][:4] + [0]), # Add 0 for NDVI mean
                "std": np.array(full_stats["channel_stds"][:4] + [1]),   # Add 1 for NDVI std
            },
            # SAR channels: VV_desc, VH_desc, VV_asc, VH_asc (4-7) + diff channels (8-11)
            "sar": {
                "mean": np.array(full_stats["channel_means"][4:]),  # All SAR channels
                "std": np.array(full_stats["channel_stds"][4:]),
            },
        }

    def _compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Compute NDVI from NIR and Red channels"""
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)
        return np.clip(ndvi, -1, 1)

    def _normalize_modality(self, data: np.ndarray, modality: str) -> np.ndarray:
        """Normalize data using modality-specific statistics"""
        stats = self.stats[modality]
        return (data - stats["mean"]) / (stats["std"] + 1e-8)

    def _prepare_optical_data(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare 5-channel optical data: R, G, B, NIR, NDVI

        Args:
            data: Full 12-channel data array (H, W, 12)

        Returns:
            optical_data: (H, W, 5) array
        """
        # Extract optical channels (R, G, B, NIR)
        optical = data[:, :, :4]  # Channels 0-3

        # Compute NDVI
        red = data[:, :, 0]
        nir = data[:, :, 3]
        ndvi = self._compute_ndvi(nir, red)

        # Stack all optical channels
        optical_with_ndvi = np.concatenate([optical, ndvi[..., np.newaxis]], axis=2)  # Add NDVI as 5th channel

        # Normalize optical data
        optical_normalized = self._normalize_modality(optical_with_ndvi, "optical")

        return optical_normalized.astype(np.float32)

    def _prepare_sar_data(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare 8-channel SAR data: 4 original + 4 difference channels

        Args:
            data: Full 12-channel data array (H, W, 12)

        Returns:
            sar_data: (H, W, 8) array
        """
        # Extract SAR channels (original: 4-7, difference: 8-11)
        sar_original = data[:, :, 4:8]  # VV_desc, VH_desc, VV_asc, VH_asc
        sar_diff = data[:, :, 8:12]  # Difference channels

        # Combine all SAR channels
        sar_combined = np.concatenate([sar_original, sar_diff], axis=2)

        # Normalize SAR data
        sar_normalized = self._normalize_modality(sar_combined, "sar")

        return sar_normalized.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for TNF model.

        Returns:
            Dict containing:
                - optical: (5, 64, 64) tensor
                - sar: (8, 64, 64) tensor
                - label: scalar tensor (for training)
                - id: sample ID string
        """
        # Get sample info
        row = self.df.iloc[idx]
        sample_id = row["ID"]

        # Load data
        data_path = self.data_dir / f"{sample_id}.npy"
        try:
            data = np.load(data_path)  # Shape: (64, 64, 12)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            # Return dummy data to prevent training crash
            data = np.zeros((64, 64, 12), dtype=np.float32)

        # Prepare modality-specific data
        optical_data = self._prepare_optical_data(data)  # (64, 64, 5)
        sar_data = self._prepare_sar_data(data)  # (64, 64, 8)

        # Apply augmentations if provided
        if self.augmentations is not None:
            # Albumentations expects HWC format
            augmented = self.augmentations(image=optical_data, sar=sar_data)
            optical_data = augmented["image"]  # Now CHW tensor
            sar_data = augmented["sar"]  # Now CHW tensor
        else:
            # Convert to CHW tensors manually
            optical_data = torch.from_numpy(optical_data).permute(2, 0, 1)
            sar_data = torch.from_numpy(sar_data).permute(2, 0, 1)

        # Prepare return dictionary
        sample = {"optical": optical_data, "sar": sar_data, "id": sample_id}  # (5, 64, 64)  # (8, 64, 64)

        # Add label for training/validation
        if self.mode != "test" and "label" in row:
            sample["label"] = torch.tensor(row["label"], dtype=torch.float32)

        return sample


def get_augmentations(mode: str) -> A.Compose:
    """
    Get augmentation pipeline for TNF model.

    Args:
        mode: 'train' or 'val'

    Returns:
        Albumentations composition with dual-modality support
    """
    aug_config = config.AUGMENTATION_CONFIG

    if mode == "train":
        transforms = [
            A.HorizontalFlip(p=aug_config["horizontal_flip_prob"]),
            A.VerticalFlip(p=aug_config["vertical_flip_prob"]),
            A.Rotate(limit=aug_config["rotation_limit"], p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.ShiftScaleRotate(
                shift_limit=aug_config["shift_limit"],
                scale_limit=aug_config["scale_limit"],
                rotate_limit=0,  # Already handled by A.Rotate
                p=0.3,
                border_mode=cv2.BORDER_REFLECT,
            ),
        ]

        # Add advanced augmentations if enabled
        if aug_config.get("apply_advanced", False):
            transforms.extend(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=aug_config["brightness_limit"],
                        contrast_limit=aug_config["contrast_limit"],
                        p=0.3,
                    ),
                    A.GaussianBlur(blur_limit=aug_config["blur_limit"], p=0.2),
                    A.GaussNoise(var_limit=aug_config["noise_var_limit"], per_channel=True, p=0.2),
                ]
            )

    else:  # validation
        transforms = []

    # Add tensor conversion (always last)
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets={"sar": "image"})  # Tell albumentations that 'sar' is also an image


def get_tta_augmentations() -> List[A.Compose]:
    """
    Get Test-Time Augmentation transforms for TNF model.

    Returns:
        List of augmentation pipelines for TTA
    """
    tta_transforms = [
        # Original
        A.Compose([ToTensorV2()], additional_targets={"sar": "image"}),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
        # Both flips
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
        # Rotation 90°
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
        # Rotation 180°
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
        # Rotation 270°
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), ToTensorV2()], additional_targets={"sar": "image"}),
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
    Create training and validation datasets for TNF model.

    Args:
        train_df: Training DataFrame (already filtered)
        val_df: Validation DataFrame (already filtered)
        data_dir: Path to data directory

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = MultiModalLandslideDataset(
        df=train_df, data_dir=data_dir, exclude_ids=[], augmentations=get_augmentations("train"), mode="train"
    )

    val_dataset = MultiModalLandslideDataset(
        df=val_df, data_dir=data_dir, exclude_ids=[], augmentations=get_augmentations("val"), mode="val"
    )

    return train_dataset, val_dataset


# Validation function for augmentation config
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

    print("✅ Augmentation configuration validated for TNF model")


# Call validation when module is imported
try:
    validate_augmentation_config()
except Exception as e:
    print(f"⚠️ Augmentation config validation warning: {e}")
