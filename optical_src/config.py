"""
Configuration for MM-LandslideNet Optical Baseline

This module contains the OpticalBaselineConfig class that inherits from the base Config
and provides specific parameters for the optical baseline model.
"""

import sys
from pathlib import Path

# Add parent directory to path to import base config
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config


class OpticalBaselineConfig(Config):
    """
    Configuration class for optical baseline model that inherits from base Config.

    This class overrides and adds parameters specific to the optical baseline implementation.
    """

    def __init__(self):
        """Initialize optical baseline configuration."""
        super().__init__()

        # Model specific parameters
        self.MODEL_NAME = "swin_tiny_patch4_window7_224"
        self.NUM_CLASSES = 1  # Binary classification
        self.DROPOUT_RATE = 0.2
        self.PRETRAINED = True

        # Training specific parameters
        self.NUM_EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4

        # Scheduler parameters
        self.SCHEDULER_T_MAX = 50
        self.SCHEDULER_ETA_MIN = 1e-6

        # Early stopping parameters
        self.EARLY_STOPPING = True
        self.PATIENCE = 10
        self.MIN_DELTA = 1e-4

        # Data parameters
        self.INPUT_CHANNELS = 5  # R, G, B, NIR, NDVI
        self.IMAGE_SIZE = 64  # Original image size
        self.TARGET_SIZE = 224  # Resized for Swin Transformer

        # Data augmentation parameters
        self.HORIZONTAL_FLIP_PROB = 0.5
        self.VERTICAL_FLIP_PROB = 0.5
        self.ROTATION_PROB = 0.5

        # Loss function parameters
        self.USE_CLASS_WEIGHTING = True
        self.FOCAL_LOSS_ALPHA = 0.25
        self.FOCAL_LOSS_GAMMA = 2.0

        # Optical-specific paths
        self.OPTICAL_LOG_DIR = self.LOG_DIR / "optical_baseline"
        self.OPTICAL_CHECKPOINT_DIR = self.CHECKPOINT_DIR / "optical_baseline"
        self.OPTICAL_OUTPUT_DIR = self.OUTPUT_ROOT / "optical_baseline"

        # Model variant options
        self.MODEL_VARIANTS = {
            "swin_tiny": {"model_name": "swin_tiny_patch4_window7_224", "feature_dim": 768, "pretrained": True},
            "swin_small": {"model_name": "swin_small_patch4_window7_224", "feature_dim": 768, "pretrained": True},
            "swin_base": {"model_name": "swin_base_patch4_window7_224", "feature_dim": 1024, "pretrained": True},
        }

        # Training monitoring parameters
        self.LOG_FREQUENCY = 50  # Log every N batches
        self.SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
        self.VALIDATION_FREQUENCY = 1  # Validate every N epochs

        # Reproducibility
        self.SEED = 42
        self.DETERMINISTIC = True

        # Hardware settings
        self.NUM_WORKERS = 4
        self.PIN_MEMORY = True
        self.MIXED_PRECISION = True  # Use automatic mixed precision

        # Evaluation metrics
        self.METRICS = ["accuracy", "precision", "recall", "f1_score", "auc"]

        # Phase 2 specific data paths
        self.PHASE_2_OUTPUTS = self.OUTPUT_ROOT / "phase_2_optical"
        self.OPTICAL_RESULTS = self.PHASE_2_OUTPUTS / "results"
        self.OPTICAL_PREDICTIONS = self.PHASE_2_OUTPUTS / "predictions"

        # Ensure optical-specific directories exist
        self._create_optical_dirs()

    def _create_optical_dirs(self) -> None:
        """Create optical baseline specific directories."""
        dirs_to_create = [
            self.OPTICAL_LOG_DIR,
            self.OPTICAL_CHECKPOINT_DIR,
            self.OPTICAL_OUTPUT_DIR,
            self.PHASE_2_OUTPUTS,
            self.OPTICAL_RESULTS,
            self.OPTICAL_PREDICTIONS,
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_model_config(self, variant: str = "swin_tiny") -> dict:
        """
        Get model configuration for a specific variant.

        Args:
            variant: Model variant name

        Returns:
            Dictionary with model configuration
        """
        if variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {variant}. " f"Available variants: {list(self.MODEL_VARIANTS.keys())}"
            )

        config = self.MODEL_VARIANTS[variant].copy()
        config.update(
            {"num_classes": self.NUM_CLASSES, "dropout_rate": self.DROPOUT_RATE, "input_channels": self.INPUT_CHANNELS}
        )

        return config

    def get_training_config(self) -> dict:
        """
        Get training configuration dictionary.

        Returns:
            Dictionary with training parameters
        """
        return {
            "num_epochs": self.NUM_EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "weight_decay": self.WEIGHT_DECAY,
            "scheduler_t_max": self.SCHEDULER_T_MAX,
            "scheduler_eta_min": self.SCHEDULER_ETA_MIN,
            "early_stopping": self.EARLY_STOPPING,
            "patience": self.PATIENCE,
            "min_delta": self.MIN_DELTA,
            "use_class_weighting": self.USE_CLASS_WEIGHTING,
            "mixed_precision": self.MIXED_PRECISION,
            "seed": self.SEED,
        }

    def get_data_config(self) -> dict:
        """
        Get data configuration dictionary.

        Returns:
            Dictionary with data parameters
        """
        return {
            "input_channels": self.INPUT_CHANNELS,
            "image_size": self.IMAGE_SIZE,
            "target_size": self.TARGET_SIZE,
            "batch_size": self.BATCH_SIZE,
            "num_workers": self.NUM_WORKERS,
            "pin_memory": self.PIN_MEMORY,
            "horizontal_flip_prob": self.HORIZONTAL_FLIP_PROB,
            "vertical_flip_prob": self.VERTICAL_FLIP_PROB,
            "rotation_prob": self.ROTATION_PROB,
        }

    def __str__(self) -> str:
        """String representation of the config."""
        return f"OpticalBaselineConfig(model={self.MODEL_NAME}, epochs={self.NUM_EPOCHS}, batch_size={self.BATCH_SIZE})"

    def __repr__(self) -> str:
        """Detailed representation of the config."""
        return self.__str__()
