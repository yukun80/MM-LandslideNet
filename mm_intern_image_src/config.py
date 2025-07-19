import torch
from pathlib import Path
from datetime import datetime


class Config:
    """Global configuration class for the project."""

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

    # ============= Model Configuration =============
    MODEL_NAME = "OpticalDominatedCooperativeModel"
    NUM_CLASSES = 1

    # ============= Training Configuration =============
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP_VAL = 1.0
    OPTIMIZER_CONFIG = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    SCHEDULER_CONFIG = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}
    LOSS_CONFIG = {"focal_alpha": 0.25, "focal_gamma": 2.0, "dice_smooth": 1.0, "focal_weight": 0.7, "dice_weight": 0.3}

    # ============= Data Augmentation Configuration =============
    AUGMENTATION_CONFIG = {
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotation_limit": 15,  # degrees
        "shift_limit": 0.1,  # fraction of image size
        "scale_limit": 0.1,  # fraction of scale change
        "brightness_limit": 0.1,  # brightness variation
        "contrast_limit": 0.1,  # contrast variation
        "gamma_limit": (80, 120),  # gamma correction range
        "blur_limit": 3,  # maximum blur kernel size
        "noise_var_limit": (10, 50),  # noise variance range
        "cutout_max_holes": 2,  # maximum cutout holes
        "cutout_max_size": 8,  # maximum cutout size (pixels)
        "mixup_alpha": 0.2,  # mixup alpha parameter
        "cutmix_alpha": 1.0,  # cutmix alpha parameter
        "apply_advanced": False,  # whether to apply advanced augmentations
    }

    # ============= Data Preprocessing Configuration =============
    DATA_CONFIG = {
        "image_size": 64,  # input image size (64x64)
        "channels": {
            "optical": 5,  # R, G, B, NIR, NDVI
            "sar": 4,  # VV_desc, VH_desc, VV_asc, VH_asc
            "sar_change": 4,  # diff channels
        },
        "normalization_method": "per_modality",  # independent normalization
        "clip_outliers": True,  # clip extreme values
        "outlier_percentile": 99.5,  # percentile for outlier clipping
    }

    # ============= Checkpoint & Logging Configuration =============
    EARLY_STOPPING_PATIENCE = 20
    MONITOR_METRIC = "val_f1_score"
    SAVE_EVERY_N_EPOCHS = 10
    SAVE_TOP_K = 3  # keep top 3 models

    # ============= Logging Configuration =============
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    TENSORBOARD_LOG = True  # enable tensorboard logging
    WANDB_LOG = False  # enable weights & biases logging (optional)

    # ============= Random Seed =============
    RANDOM_SEED = 42
    USE_WEIGHTED_SAMPLER = True

    # ============= Advanced Training Settings =============
    MIXED_PRECISION = True  # use automatic mixed precision
    COMPILE_MODEL = False  # use torch.compile (PyTorch 2.0+)
    GRADIENT_ACCUMULATION_STEPS = 1  # for effective larger batch size

    # ============= Model-specific Configuration =============
    MODEL_CONFIG = {
        "pretrained": True,
        "dropout_rate": 0.5,
        "use_checkpoint": False,  # gradient checkpointing
        "fusion_dropout": 0.1,  # dropout in fusion blocks
        "classifier_dropout": 0.5,  # dropout in classifier
    }

    def setup_run_paths(self):
        """Creates a unique directory for the current training run."""
        if self.RUN_ID is None:
            self.RUN_ID = f"{self.MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.CHECKPOINT_DIR = self.OUTPUT_ROOT / "checkpoints" / self.RUN_ID
        self.LOG_DIR = self.OUTPUT_ROOT / "logs" / self.RUN_ID

        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.OUTPUT_ROOT / "submissions").mkdir(exist_ok=True)
        (self.OUTPUT_ROOT / "visualizations").mkdir(exist_ok=True)
        (self.OUTPUT_ROOT / "reports").mkdir(exist_ok=True)

    def get_best_model_path(self) -> Path:
        return self.CHECKPOINT_DIR / "best_model.pth"

    def get_latest_model_path(self) -> Path:
        return self.CHECKPOINT_DIR / "latest_model.pth"

    def get_epoch_model_path(self, epoch: int) -> Path:
        return self.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"

    def get_tensorboard_dir(self) -> Path:
        return self.LOG_DIR / "tensorboard"

    def validate_config(self):
        """Validate configuration parameters."""
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.NUM_EPOCHS > 0, "Number of epochs must be positive"
        assert 0 < self.VALIDATION_SPLIT < 1, "Validation split must be between 0 and 1"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"

        # Validate paths exist
        assert self.DATA_DIR.exists(), f"Data directory does not exist: {self.DATA_DIR}"
        assert self.TRAIN_CSV_PATH.exists(), f"Train CSV does not exist: {self.TRAIN_CSV_PATH}"
        assert self.TRAIN_DATA_DIR.exists(), f"Train data directory does not exist: {self.TRAIN_DATA_DIR}"

    def print_config(self):
        """Print configuration summary."""
        print(f"--- {self.MODEL_NAME} Configuration ---")
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {getattr(self, key)}")
        print("-------------------------------------")

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        config_dict = {}
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                config_dict[key] = getattr(self, key)
        return config_dict


# Global configuration instance
config = Config()

# Validate configuration on import
try:
    config.validate_config()
except AssertionError as e:
    print(f"⚠️ Configuration validation warning: {e}")
except Exception as e:
    print(f"⚠️ Configuration setup warning: {e}")
