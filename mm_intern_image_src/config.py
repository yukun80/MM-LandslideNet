"""
Modified Configuration for MM-TNF Model

Key Changes:
1. Updated model configuration for dual-branch TNF architecture
2. Simplified data configuration (optical: 5ch, sar: 8ch)
3. Added TNF-specific training parameters
4. Updated loss configuration for multi-branch training
"""

import torch
from pathlib import Path
from datetime import datetime


class Config:
    """Global configuration class for MM-TNF model."""

    # ============= Project Paths =============
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "dataset"
    TRAIN_CSV_PATH = DATA_DIR / "Train.csv"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"
    TEST_DATA_DIR = DATA_DIR / "test_data"
    STATS_FILE_PATH = DATA_DIR / "data_check" / "channel_stats.json"
    EXCLUDE_FILE_PATH = DATA_DIR / "data_check" / "exclude_ids.json"
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"

    # --- These paths are now set dynamically per run ---
    CHECKPOINT_DIR = None
    LOG_DIR = None
    RUN_ID = None

    # ============= Hardware Configuration =============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    MIXED_PRECISION = True  # Enable for faster training on modern GPUs

    # ============= Model Configuration =============
    MODEL_NAME = "MM-TNF"  # Updated model name
    NUM_CLASSES = 1

    # TNF Model specific configuration
    MODEL_CONFIG = {
        "pretrained": True,
        "optical_channels": 5,     # R, G, B, NIR, NDVI
        "sar_channels": 8,         # 4 original + 4 difference SAR channels
        "optical_feature_dim": 512,  # InternImage-T output dimension
        "sar_feature_dim": 512,      # EfficientNet-B0 aligned dimension
        "fusion_dim": 512,           # TNF fusion dimension
        "dropout_rate": 0.1,
        "branch_weights": (0.3, 0.2, 0.5),  # (optical, sar, fusion) loss weights
    }

    # ============= Training Configuration =============
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP_VAL = 1.0
    RANDOM_SEED = 42

    # Optimizer configuration
    OPTIMIZER_CONFIG = {
        "lr": LEARNING_RATE, 
        "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    # Scheduler configuration
    SCHEDULER_CONFIG = {
        "T_max": NUM_EPOCHS, 
        "eta_min": 1e-6,
    }

    # Multi-branch loss configuration for TNF model
    LOSS_CONFIG = {
        "focal_alpha": 0.25,        # Focal loss alpha (class balance)
        "focal_gamma": 2.0,         # Focal loss gamma (focusing parameter)
        "dice_smooth": 1.0,         # Dice loss smoothing
        "focal_weight": 0.7,        # Weight of focal loss in combination
        "dice_weight": 0.3,         # Weight of dice loss in combination
        "pos_weight": None,         # Will be calculated automatically from data
    }

    # ============= Data Augmentation Configuration =============
    AUGMENTATION_CONFIG = {
        # Basic augmentations
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotation_limit": 15,       # degrees
        "shift_limit": 0.1,         # fraction of image size
        "scale_limit": 0.1,         # fraction of scale change
        
        # Advanced augmentations (for optical data primarily)
        "brightness_limit": 0.1,    # brightness variation
        "contrast_limit": 0.1,      # contrast variation
        "gamma_limit": (80, 120),   # gamma correction range
        "blur_limit": 3,            # maximum blur kernel size
        "noise_var_limit": (10, 50), # noise variance range
        
        # Cutout augmentations
        "cutout_max_holes": 2,      # maximum cutout holes
        "cutout_max_size": 8,       # maximum cutout size (pixels)
        
        # Mix augmentations (experimental)
        "mixup_alpha": 0.2,         # mixup alpha parameter
        "cutmix_alpha": 1.0,        # cutmix alpha parameter
        
        # Enable/disable advanced augmentations
        "apply_advanced": False,    # whether to apply advanced augmentations
    }

    # ============= Data Configuration =============
    DATA_CONFIG = {
        "image_size": 64,           # input image size (64x64)
        "channels": {
            "optical": 5,           # R, G, B, NIR, NDVI
            "sar": 8,              # 4 original + 4 difference SAR channels
            "total": 12,           # Total channels in raw data
        },
        "normalization_method": "per_modality",  # independent normalization per modality
        "clip_outliers": True,      # clip extreme values
        "outlier_percentile": 99.5, # percentile for clipping
        "ndvi_range": (-1, 1),      # NDVI value range
    }

    # ============= Class Imbalance Handling =============
    IMBALANCE_CONFIG = {
        "use_weighted_sampler": True,    # Use weighted random sampler
        "use_class_weights": True,       # Use class weights in loss
        "pos_weight_multiplier": 4.0,    # Multiplier for positive class weight
        "sampling_strategy": "balanced", # 'balanced' or 'custom'
    }

    # Expose at top level for backward compatibility
    USE_WEIGHTED_SAMPLER = IMBALANCE_CONFIG["use_weighted_sampler"]

    # ============= Training Monitoring =============
    MONITOR_METRIC = "f1"           # Primary metric to monitor
    EARLY_STOPPING_PATIENCE = 15   # Epochs to wait before early stopping
    CHECKPOINT_SAVE_FREQ = 5        # Save checkpoint every N epochs
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    SAVE_TRAINING_PLOTS = True
    PLOT_UPDATE_FREQ = 10          # Update plots every N epochs

    # ============= Prediction Configuration =============
    PREDICTION_CONFIG = {
        "default_threshold": 0.5,
        "use_tta": True,            # Use Test-Time Augmentation by default
        "tta_confidence_threshold": 0.9,  # Threshold for high-confidence TTA predictions
        "ensemble_weights": {       # Weights for ensemble prediction
            "optical": 0.3,
            "sar": 0.2, 
            "fusion": 0.5
        },
        "output_detailed_results": True,  # Save detailed branch predictions
    }

    # ============= Validation and Quality Control =============
    QUALITY_CONFIG = {
        "enable_sanity_checks": True,
        "validate_data_loading": True,
        "check_gradient_flow": True,
        "monitor_learning_curves": True,
        "detect_overfitting": True,
    }

    def setup_run_paths(self, run_id: str = None):
        """Setup paths for a specific training run."""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"tnf_run_{timestamp}"
        
        self.RUN_ID = run_id
        self.CHECKPOINT_DIR = self.OUTPUT_ROOT / "checkpoints" / run_id
        self.LOG_DIR = self.OUTPUT_ROOT / "logs" / run_id
        
        # Create directories
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Run paths setup for: {run_id}")
        print(f"   Checkpoints: {self.CHECKPOINT_DIR}")
        print(f"   Logs: {self.LOG_DIR}")

    def validate_config(self):
        """Validate configuration parameters."""
        # Basic validation
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.NUM_EPOCHS > 0, "Number of epochs must be positive"
        assert 0 < self.VALIDATION_SPLIT < 1, "Validation split must be between 0 and 1"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"

        # Model configuration validation
        model_config = self.MODEL_CONFIG
        assert model_config["optical_channels"] == 5, "Optical channels must be 5 for TNF model"
        assert model_config["sar_channels"] == 8, "SAR channels must be 8 for TNF model"
        assert len(model_config["branch_weights"]) == 3, "Branch weights must have 3 elements"
        assert abs(sum(model_config["branch_weights"]) - 1.0) < 1e-6, "Branch weights must sum to 1.0"

        # Data configuration validation
        data_config = self.DATA_CONFIG
        assert data_config["image_size"] == 64, "Image size must be 64 for current model"
        expected_total = data_config["channels"]["optical"] + data_config["channels"]["sar"]
        assert expected_total <= data_config["channels"]["total"], "Channel counts inconsistent"

        # Path validation
        assert self.DATA_DIR.exists(), f"Data directory does not exist: {self.DATA_DIR}"
        assert self.TRAIN_CSV_PATH.exists(), f"Train CSV does not exist: {self.TRAIN_CSV_PATH}"
        assert self.TRAIN_DATA_DIR.exists(), f"Train data directory does not exist: {self.TRAIN_DATA_DIR}"

        # Augmentation validation
        aug_config = self.AUGMENTATION_CONFIG
        assert 0 <= aug_config["horizontal_flip_prob"] <= 1, "Flip probabilities must be [0,1]"
        assert 0 <= aug_config["vertical_flip_prob"] <= 1, "Flip probabilities must be [0,1]"
        assert aug_config["rotation_limit"] >= 0, "Rotation limit must be non-negative"

    def print_config(self):
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print(f"MM-TNF Model Configuration")
        print(f"{'='*60}")
        
        # Model info
        print(f"Model: {self.MODEL_NAME}")
        print(f"Architecture: Dual-branch TNF ({self.MODEL_CONFIG['optical_channels']}ch optical + {self.MODEL_CONFIG['sar_channels']}ch SAR)")
        print(f"Feature dimensions: {self.MODEL_CONFIG['optical_feature_dim']} (opt) / {self.MODEL_CONFIG['sar_feature_dim']} (sar)")
        print(f"Branch weights: {self.MODEL_CONFIG['branch_weights']}")
        
        # Training info
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {self.BATCH_SIZE}")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Learning rate: {self.LEARNING_RATE}")
        print(f"  Device: {self.DEVICE}")
        print(f"  Mixed precision: {self.MIXED_PRECISION}")
        print(f"  Weighted sampler: {self.USE_WEIGHTED_SAMPLER}")
        
        # Data info
        print(f"\nData Configuration:")
        print(f"  Image size: {self.DATA_CONFIG['image_size']}x{self.DATA_CONFIG['image_size']}")
        print(f"  Optical channels: {self.DATA_CONFIG['channels']['optical']}")
        print(f"  SAR channels: {self.DATA_CONFIG['channels']['sar']}")
        print(f"  Validation split: {self.VALIDATION_SPLIT}")
        
        # Loss info
        print(f"\nLoss Configuration:")
        print(f"  Focal (α={self.LOSS_CONFIG['focal_alpha']}, γ={self.LOSS_CONFIG['focal_gamma']})")
        print(f"  Combination weights: Focal {self.LOSS_CONFIG['focal_weight']}, Dice {self.LOSS_CONFIG['dice_weight']}")
        
        print(f"{'='*60}\n")

    def get_model_config(self) -> dict:
        """Get model configuration dictionary."""
        return self.MODEL_CONFIG.copy()

    def update_model_config(self, updates: dict):
        """Update model configuration."""
        self.MODEL_CONFIG.update(updates)
        print(f"✅ Model configuration updated: {list(updates.keys())}")

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        config_dict = {}
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                try:
                    # Handle Path objects
                    if isinstance(value, Path):
                        config_dict[key] = str(value)
                    # Handle other serializable objects
                    elif isinstance(value, (int, float, str, bool, list, dict, tuple)):
                        config_dict[key] = value
                    else:
                        config_dict[key] = str(value)
                except:
                    config_dict[key] = str(value)
        return config_dict


# Global configuration instance
config = Config()

# Validate configuration on import
try:
    config.validate_config()
    print("✅ TNF Configuration validated successfully")
except AssertionError as e:
    print(f"⚠️ Configuration validation warning: {e}")
except Exception as e:
    print(f"⚠️ Configuration setup warning: {e}")

# Auto-setup paths if not already done
if config.RUN_ID is None:
    try:
        config.setup_run_paths()
    except Exception as e:
        print(f"⚠️ Could not auto-setup paths: {e}")