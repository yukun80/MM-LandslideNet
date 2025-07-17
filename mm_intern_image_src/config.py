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
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP_VAL = 1.0
    OPTIMIZER_CONFIG = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    SCHEDULER_CONFIG = {"T_max": NUM_EPOCHS, "eta_min": 1e-6}
    LOSS_CONFIG = {"focal_alpha": 0.25, "focal_gamma": 2.0, "dice_smooth": 1.0, "focal_weight": 0.7, "dice_weight": 0.3}

    # ============= Checkpoint & Logging Configuration =============
    EARLY_STOPPING_PATIENCE = 20
    MONITOR_METRIC = "val_f1_score"
    SAVE_EVERY_N_EPOCHS = 10  # New: Save a checkpoint every 10 epochs

    # ============= Random Seed =============
    RANDOM_SEED = 42
    USE_WEIGHTED_SAMPLER = True

    def setup_run_paths(self):
        """Creates a unique directory for the current training run."""
        if self.RUN_ID is None:
            self.RUN_ID = f"{self.MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.CHECKPOINT_DIR = self.OUTPUT_ROOT / "checkpoints" / self.RUN_ID
        self.LOG_DIR = self.OUTPUT_ROOT / "logs" / self.RUN_ID

        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def get_best_model_path(self) -> Path:
        return self.CHECKPOINT_DIR / "best_model.pth"

    def get_latest_model_path(self) -> Path:
        return self.CHECKPOINT_DIR / "latest_model.pth"

    def get_epoch_model_path(self, epoch: int) -> Path:
        return self.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"

    def print_config(self):
        print(f"--- {self.MODEL_NAME} Configuration ---")
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {getattr(self, key)}")
        print("-------------------------------------")


# Global configuration instance
config = Config()
