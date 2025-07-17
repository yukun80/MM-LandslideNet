"""
Global Configuration for MM-InternImage-TNF Multi-Modal Landslide Detection

This module centralizes all configuration parameters for the project,
following the principle of avoiding hardcoded values for better maintainability
and experimental flexibility.
"""

import os
import torch
from pathlib import Path


class Config:
    """Global configuration class for MM-InternImage-TNF project"""

    # ============= Project Paths =============
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "dataset"
    TRAIN_CSV_PATH = DATA_DIR / "Train.csv"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"
    TEST_DATA_DIR = DATA_DIR / "test_data"

    # Statistics and filtering files
    STATS_FILE_PATH = DATA_DIR / "data_check" / "channel_stats.json"
    EXCLUDE_FILE_PATH = DATA_DIR / "data_check" / "exclude_ids.json"

    # Output directories
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_ROOT / "mm_intern_image_tnf"
    LOG_DIR = PROJECT_ROOT / "logs" / "mm_intern_image_tnf"

    # Ensure output directories exist
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ============= Hardware Configuration =============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading
    PIN_MEMORY = True if torch.cuda.is_available() else False

    # ============= Model Configuration =============
    MODEL_NAME = "AsymmetricFusionModel"
    NUM_CLASSES = 1  # Binary classification (sigmoid output)

    # InternImage-T configuration (matching the compiled version)
    INTERNIMAGE_CONFIG = {
        "core_op": "DCNv3",
        "channels": 64,
        "depths": [4, 4, 18, 4],
        "groups": [4, 8, 16, 32],
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
        "offset_scale": 1.0,
        "layer_scale": None,
        "post_norm": False,
        "with_cp": False,  # Checkpoint during training
    }

    # Input channel configuration
    OPTICAL_CHANNELS = 5  # R, G, B, NIR, NDVI
    SAR_COMBINED_CHANNELS = 8  # VV_desc, VH_desc, VV_asc, VH_asc, VV_desc_diff, VH_desc_diff, VV_asc_diff, VH_asc_diff

    # Lightweight SAR CNN configuration
    SAR_CNN_CONFIG = {
        "channels": [8, 16, 32, 64],  # Output channels for each stage
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
    }

    # Cross Fusion Block configuration
    FUSION_BLOCK_CONFIG = {
        "hidden_dim": 256,  # Hidden dimension in fusion block
        "num_heads": 8,  # Multi-head attention heads
        "dropout": 0.1,  # Dropout rate in fusion block
    }

    # ============= Training Configuration =============
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP_VAL = 1.0  # Gradient clipping for stability

    # Optimizer configuration
    OPTIMIZER_CONFIG = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "betas": (0.9, 0.999), "eps": 1e-8}

    # Learning rate scheduler configuration
    SCHEDULER_CONFIG = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}

    # Loss function weights
    LOSS_CONFIG = {"focal_alpha": 0.25, "focal_gamma": 2.0, "dice_smooth": 1.0, "focal_weight": 1.0, "dice_weight": 1.0}

    # ============= Data Augmentation =============
    AUGMENTATION_CONFIG = {
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotation_limit": 15,
        "shift_limit": 0.1,
        "scale_limit": 0.1,
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
    }

    # ============= Early Stopping & Monitoring =============
    EARLY_STOPPING_PATIENCE = 15
    MONITOR_METRIC = "val_f1_score"  # Metric to monitor for best model
    SAVE_TOP_K = 3  # Save top 3 best models

    # ============= Logging Configuration =============
    LOG_EVERY_N_STEPS = 10
    VALIDATE_EVERY_N_EPOCHS = 1

    # ============= Random Seed =============
    RANDOM_SEED = 42

    # ============= Class Imbalance Handling =============
    # Will be calculated dynamically from the dataset
    CLASS_WEIGHTS = None  # To be set during training initialization
    USE_WEIGHTED_SAMPLER = True

    @classmethod
    def get_model_save_path(cls, epoch: int, metric_value: float) -> Path:
        """Generate standardized model save path"""
        filename = f"{cls.MODEL_NAME}_epoch_{epoch:03d}_{cls.MONITOR_METRIC}_{metric_value:.4f}.pth"
        return cls.CHECKPOINT_DIR / filename

    @classmethod
    def get_best_model_path(cls) -> Path:
        """Get path for the best model"""
        return cls.CHECKPOINT_DIR / f"{cls.MODEL_NAME}_best.pth"

    @classmethod
    def get_latest_model_path(cls) -> Path:
        """Get path for the latest model"""
        return cls.CHECKPOINT_DIR / f"{cls.MODEL_NAME}_latest.pth"

    @classmethod
    def print_config(cls):
        """Print current configuration for debugging"""
        print(f"=== {cls.MODEL_NAME} Configuration ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Checkpoint Directory: {cls.CHECKPOINT_DIR}")
        print(f"InternImage Config: {cls.INTERNIMAGE_CONFIG}")
        print("=" * 50)


# Global configuration instance
config = Config()
