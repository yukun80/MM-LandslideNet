import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, Any, Union

# Import optical baseline config and utils
from optical_src.config import OpticalBaselineConfig
from optical_src.utils import setup_logging

# Setup logger for this module
logger = logging.getLogger("optical_baseline.dataset")


class LandslideDataset(Dataset):
    """
    Dataset class for landslide detection using optical-only data.
    Processes 5-channel input: R, G, B, NIR, NDVI
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        data_dir: Union[str, Path],
        exclude_ids: Optional[set] = None,
        channel_stats: Optional[Dict[str, Any]] = None,
        transforms: Optional[A.Compose] = None,
        is_test: bool = False,
    ):
        """
        Args:
            csv_file: Path to Train.csv or Test.csv
            data_dir: Path to train_data or test_data directory
            exclude_ids: Set of image IDs to exclude (low quality images)
            channel_stats: Dictionary with optical channel statistics for normalization
            transforms: Albumentations transforms
            is_test: Whether this is test dataset (no labels)
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.is_test = is_test
        self.channel_stats = channel_stats

        # Load CSV data
        self.df = pd.read_csv(csv_file)  # type: ignore
        logger.info(f"Loaded {len(self.df)} samples from {csv_file}")

        # Filter out excluded images if provided
        if exclude_ids is not None:
            original_len = len(self.df)
            exclude_ids_list = list(exclude_ids)  # Convert set to list for pandas
            self.df = self.df[~self.df["ID"].isin(exclude_ids_list)]  # type: ignore
            filtered_len = len(self.df)
            logger.info(
                f"Filtered dataset: {original_len} -> {filtered_len} samples "
                f"({original_len - filtered_len} excluded)"
            )

        self.image_ids = self.df["ID"].values  # type: ignore
        if not is_test:
            self.labels = self.df["label"].values  # type: ignore
            logger.info(
                f"Class distribution: " f"Class 0: {sum(self.labels == 0)}, " f"Class 1: {sum(self.labels == 1)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        # Load image data
        image_id = self.image_ids[idx]
        image_path = self.data_dir / f"{image_id}.npy"

        try:
            # Load 12-channel data
            data = np.load(image_path).astype(np.float32)  # Shape: (64, 64, 12)

            # Extract optical channels (0-3: R, G, B, NIR)
            optical_data = data[:, :, :4]  # Shape: (64, 64, 4)

            # Calculate NDVI: (NIR - Red) / (NIR + Red + epsilon)
            nir = optical_data[:, :, 3]  # NIR channel
            red = optical_data[:, :, 0]  # Red channel
            epsilon = 1e-8
            ndvi = (nir - red) / (nir + red + epsilon)

            # Combine into 5-channel array: R, G, B, NIR, NDVI
            optical_5ch = np.concatenate(
                [optical_data, ndvi[:, :, np.newaxis]], axis=2  # R, G, B, NIR (first 4 channels)  # NDVI as 5th channel
            )  # Shape: (64, 64, 5)

            # Apply channel-wise normalization if statistics provided
            if self.channel_stats is not None:
                optical_5ch = self._normalize_channels(optical_5ch)

            # Apply augmentations
            if self.transforms is not None:
                # Albumentations expects HWC format
                augmented = self.transforms(image=optical_5ch)
                optical_5ch = augmented["image"]
            else:
                # Convert to tensor and change from HWC to CHW
                optical_5ch = torch.from_numpy(optical_5ch).permute(2, 0, 1)

            if self.is_test:
                return optical_5ch, image_id
            else:
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                return optical_5ch, label

        except Exception as e:
            logger.error(f"Error loading image {image_id}: {str(e)}")
            # Return zero tensor as fallback
            optical_5ch = torch.zeros(5, 64, 64, dtype=torch.float32)
            if self.is_test:
                return optical_5ch, image_id
            else:
                label = torch.tensor(0.0, dtype=torch.float32)
                return optical_5ch, label

    def _normalize_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Apply channel-wise normalization using precomputed statistics.
        Args:
            data: numpy array of shape (H, W, 5) - R, G, B, NIR, NDVI
        Returns:
            Normalized data of same shape
        """
        normalized_data = data.copy()

        # Handle case where channel_stats might be None
        if self.channel_stats is None:
            return normalized_data

        # Get optical channel statistics from channel_stats
        optical_stats = self.channel_stats.get("channel_statistics_by_group", {}).get("optical", {})

        # Normalize first 4 channels (R, G, B, NIR) using precomputed stats
        for i in range(4):
            channel_key = f"channel_{i}"
            if channel_key in optical_stats.get("channels", {}):
                stats = optical_stats["channels"][channel_key]
                mean = stats["mean"]
                std = stats["std"]
                normalized_data[:, :, i] = (normalized_data[:, :, i] - mean) / (std + 1e-8)

        # For NDVI (channel 4), normalize to [-1, 1] range (it should already be close)
        # NDVI is typically in range [-1, 1], so we can use simple clipping
        normalized_data[:, :, 4] = np.clip(normalized_data[:, :, 4], -1, 1)

        return normalized_data

    @staticmethod
    def get_transforms(is_training: bool = True, config: Optional[OpticalBaselineConfig] = None) -> A.Compose:
        """
        Get albumentations transforms for training or validation.
        """
        if config is None:
            config = OpticalBaselineConfig()

        if is_training:
            return A.Compose(
                [
                    A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                    A.VerticalFlip(p=config.VERTICAL_FLIP_PROB),
                    A.RandomRotate90(p=config.ROTATION_PROB),
                    ToTensorV2(),  # Converts to tensor and changes HWC to CHW
                ]
            )
        else:
            return A.Compose([ToTensorV2()])

    @staticmethod
    def get_train_val_dataloaders(
        config: Optional[OpticalBaselineConfig] = None,
    ) -> Tuple[DataLoader, DataLoader, "LandslideDataset", "LandslideDataset"]:
        """
        Create training and validation dataloaders with proper configuration.

        Returns:
            tuple: (train_loader, val_loader, train_dataset, val_dataset)
        """
        if config is None:
            config = OpticalBaselineConfig()

        # Load exclude_ids
        exclude_ids_path = config.PROJECT_ROOT / "dataset" / "data_check" / "exclude_ids.json"
        with open(exclude_ids_path, "r") as f:
            exclude_data = json.load(f)
            exclude_ids = set(exclude_data["excluded_image_ids"])

        logger.info(f"Loaded {len(exclude_ids)} excluded image IDs")

        # Load channel statistics
        channel_stats_path = config.PROJECT_ROOT / "dataset" / "data_check" / "channel_stats.json"
        with open(channel_stats_path, "r") as f:
            channel_stats = json.load(f)

        logger.info("Loaded channel statistics for normalization")

        # Load training data
        df = pd.read_csv(config.TRAIN_CSV)

        # Filter out excluded samples
        original_len = len(df)
        exclude_ids_list = list(exclude_ids)  # Convert set to list for pandas
        df = df[~df["ID"].isin(exclude_ids_list)]
        logger.info(f"Filtered training data: {original_len} -> {len(df)} samples")

        # Split into train/val
        if config.STRATIFY:
            train_df, val_df = train_test_split(
                df, test_size=config.VAL_SPLIT, stratify=df["label"], random_state=config.RANDOM_STATE
            )
        else:
            train_df, val_df = train_test_split(df, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE)

        logger.info(f"Train split: {len(train_df)} samples")
        logger.info(f"Val split: {len(val_df)} samples")

        # Create temporary CSV files for datasets
        train_csv_path = config.OUTPUT_ROOT / "train_split.csv"
        val_csv_path = config.OUTPUT_ROOT / "val_split.csv"

        # Ensure output directory exists
        config.create_dirs()

        train_df.to_csv(train_csv_path, index=False)  # type: ignore
        val_df.to_csv(val_csv_path, index=False)  # type: ignore

        # Create datasets
        train_transforms = LandslideDataset.get_transforms(is_training=True, config=config)
        val_transforms = LandslideDataset.get_transforms(is_training=False, config=config)

        train_dataset = LandslideDataset(
            csv_file=train_csv_path,
            data_dir=config.TRAIN_DATA_DIR,
            exclude_ids=None,  # Already filtered in the CSV
            channel_stats=channel_stats,
            transforms=train_transforms,
            is_test=False,
        )

        val_dataset = LandslideDataset(
            csv_file=val_csv_path,
            data_dir=config.TRAIN_DATA_DIR,
            exclude_ids=None,  # Already filtered in the CSV
            channel_stats=channel_stats,
            transforms=val_transforms,
            is_test=False,
        )

        # Create weighted sampler for training to handle class imbalance
        train_labels = train_dataset.labels
        class_counts = np.bincount(train_labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels.astype(int)]

        # Convert to list for WeightedRandomSampler
        sample_weights_list = sample_weights.tolist()

        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights_list, num_samples=len(sample_weights_list), replacement=True
        )

        logger.info(f"Class counts in training: {class_counts}")
        logger.info(f"Class weights: {class_weights}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=weighted_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
        )

        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, " f"Val: {len(val_loader)} batches")

        return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    config = OpticalBaselineConfig()
    train_loader, val_loader, train_dataset, val_dataset = LandslideDataset.get_train_val_dataloaders(config)

    # Test loading a batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {labels.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Labels: {labels[:10]}")
        if batch_idx >= 2:  # Just test a few batches
            break
