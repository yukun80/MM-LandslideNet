"""
Fixed Configuration for MM-TNF Model

Key Fixes:
1. 修正通道数量验证逻辑，区分原始数据通道数和模型输入通道数
2. 添加更清晰的配置文档和验证说明
3. 修复DATA_CONFIG中的通道数量逻辑错误
4. 保持与现有代码的完全兼容性
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
    OUTPUT_LOGS = PROJECT_ROOT / "logs"

    # --- These paths are now set dynamically per run ---
    CHECKPOINT_DIR = None
    LOG_DIR = None
    RUN_ID = None

    # ============= Hardware Configuration =============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12
    PIN_MEMORY = True if torch.cuda.is_available() else False
    MIXED_PRECISION = False  # Enable for faster training on modern GPUs

    # ============= Model Configuration =============
    MODEL_NAME = "MM-TNF"  # Updated model name
    NUM_CLASSES = 1

    # TNF Model specific configuration
    MODEL_CONFIG = {
        "pretrained": True,
        "optical_channels": 5,  # R, G, B, NIR, NDVI (4 raw + 1 computed)
        "sar_channels": 8,  # 4 original + 4 difference SAR channels
        "optical_feature_dim": 512,  # InternImage-T output dimension
        "sar_feature_dim": 512,  # EfficientNet-B0 aligned dimension
        "fusion_dim": 512,  # TNF fusion dimension
        "dropout_rate": 0.15,
        "branch_weights": (0.4, 0.3, 0.3),  # Changed from (0.3, 0.2, 0.5)
    }

    # ============= Training Configuration =============
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP_VAL = 0.5  # Changed from 1.0，更严格的梯度裁剪
    RANDOM_SEED = 3407

    # Optimizer configuration
    OPTIMIZER_CONFIG = {
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    # Scheduler configuration
    SCHEDULER_CONFIG = {"T_max": NUM_EPOCHS, "eta_min": 1e-7}

    # Multi-branch loss configuration for TNF model
    LOSS_CONFIG = {
        "focal_alpha": 0.25,  # Focal loss alpha (class balance)
        "focal_gamma": 2.0,  # Focal loss gamma (focusing parameter)
        "dice_smooth": 1e-6,  # Dice loss smoothing
        "focal_weight": 0.7,  # Weight of focal loss in combination
        "dice_weight": 0.3,  # Weight of dice loss in combination
        "pos_weight": None,  # Will be calculated automatically from data
    }

    # ============= Data Augmentation Configuration =============
    AUGMENTATION_CONFIG = {
        # Basic augmentations
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotation_limit": 15,  # degrees
        "shift_limit": 0.1,  # fraction of image size
        "scale_limit": 0.1,  # fraction of scale change
        # Advanced augmentations (for optical data primarily)
        "brightness_limit": 0.1,  # brightness variation
        "contrast_limit": 0.1,  # contrast variation
        "gamma_limit": (80, 120),  # gamma correction range
        "blur_limit": 3,  # maximum blur kernel size
        "noise_var_limit": (10, 50),  # noise variance range
        # Cutout augmentations
        "cutout_max_holes": 2,  # maximum cutout holes
        "cutout_max_size": 8,  # maximum cutout size (pixels)
        # Mix augmentations (experimental)
        "mixup_alpha": 0.2,  # mixup alpha parameter
        "cutmix_alpha": 1.0,  # cutmix alpha parameter
        # Enable/disable advanced augmentations
        "apply_advanced": False,  # whether to apply advanced augmentations
    }

    # ============= Data Configuration =============
    # FIXED: 修正通道数量逻辑，区分原始数据和模型输入
    DATA_CONFIG = {
        "image_size": 64,  # input image size (64x64)
        "raw_data_channels": {
            # 原始.npy文件中的通道数（来自数据集）
            "optical_raw": 4,  # R, G, B, NIR (来自Sentinel-2)
            "sar_raw": 8,  # 8个SAR通道（来自Sentinel-1）
            "total_raw": 12,  # 原始数据文件总通道数
        },
        "model_input_channels": {
            # 模型实际使用的通道数（包括计算得出的特征）
            "optical_input": 5,  # R, G, B, NIR + NDVI(计算得出)
            "sar_input": 8,  # 直接使用8个SAR通道
            "total_input": 13,  # 模型输入总通道数 (5 + 8)
        },
        # 为了向后兼容，保留简化的channels字段
        "channels": {
            "optical": 4,  # 原始光学通道数（用于数据加载验证）
            "sar": 8,  # SAR通道数
            "total": 12,  # 原始数据总通道数
        },
        "normalization_method": "per_channel",
        "data_split_strategy": "stratified",
        "validation_strategy": "holdout",
    }

    # ============= Early Stopping Configuration =============
    EARLY_STOPPING_CONFIG = {
        "patience": 20,
        "min_delta": 1e-4,
        "monitor": "f1_score",
        "mode": "max",
        "restore_best_weights": True,
    }
    # 检查点配置
    CHECKPOINT_CONFIG = {
        "save_interval": 10,  # 每10个epoch保存一次
        "monitor": "f1_score",  # 监控F1分数
    }

    # ============= Logging Configuration =============
    LOGGING_CONFIG = {
        "log_level": "INFO",
        "log_interval": 10,  # Log every N batches during training
        "save_model_every_n_epochs": 10,
        "log_model_architecture": True,
        "log_grad_norm": True,
        "log_learning_rate": True,
    }

    # ============= Weighted Sampling Configuration =============
    USE_WEIGHTED_SAMPLER = True
    SAMPLER_CONFIG = {
        "strategy": "inverse_frequency",  # inverse_frequency, balanced, custom
        "smooth_factor": 0.1,  # Add smoothing to weights
    }

    # ============= Methods =============
    def setup_run_paths(self, run_id: str = None):
        """Setup directory paths for a training run."""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"tnf_run_{timestamp}"
            
        self.RUN_ID = run_id
        self.CHECKPOINT_DIR = self.OUTPUT_ROOT / "checkpoints" / run_id
        self.LOG_DIR = self.OUTPUT_LOGS / run_id

        # Create directories
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        print(f"✅ Run paths setup for: {run_id}")
        print(f"   Checkpoints: {self.CHECKPOINT_DIR}")
        print(f"   Logs: {self.LOG_DIR}")

    def validate_config(self):
        """Validate configuration parameters."""
        print("🔍 Validating TNF configuration...")

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

        # Data configuration validation - FIXED LOGIC
        data_config = self.DATA_CONFIG
        assert data_config["image_size"] == 64, "Image size must be 64 for current model"

        # 验证原始数据通道数
        raw_channels = data_config["raw_data_channels"]
        assert raw_channels["total_raw"] == 12, "Raw data must have 12 channels"
        assert (
            raw_channels["optical_raw"] + raw_channels["sar_raw"] == raw_channels["total_raw"]
        ), "Raw channel counts must sum correctly"

        # 验证模型输入通道数
        model_channels = data_config["model_input_channels"]
        assert (
            model_channels["optical_input"] == model_config["optical_channels"]
        ), "Model optical channels must match config"
        assert model_channels["sar_input"] == model_config["sar_channels"], "Model SAR channels must match config"
        assert (
            model_channels["total_input"] == model_channels["optical_input"] + model_channels["sar_input"]
        ), "Model input channel counts must sum correctly"

        # 验证向后兼容的channels字段
        legacy_channels = data_config["channels"]
        assert (
            legacy_channels["total"] == raw_channels["total_raw"]
        ), "Legacy total channels must match raw data channels"

        # Path validation (only if paths exist)
        if self.DATA_DIR.exists():
            if not self.TRAIN_CSV_PATH.exists():
                print(f"⚠️ Warning: Train CSV not found: {self.TRAIN_CSV_PATH}")
            if not self.TRAIN_DATA_DIR.exists():
                print(f"⚠️ Warning: Train data directory not found: {self.TRAIN_DATA_DIR}")

        # Augmentation validation
        aug_config = self.AUGMENTATION_CONFIG
        assert 0 <= aug_config["horizontal_flip_prob"] <= 1, "Flip probabilities must be [0,1]"
        assert 0 <= aug_config["vertical_flip_prob"] <= 1, "Flip probabilities must be [0,1]"
        assert aug_config["rotation_limit"] >= 0, "Rotation limit must be non-negative"

        print("✅ Configuration validation successful!")

    def print_config(self):
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print(f"MM-TNF Model Configuration")
        print(f"{'='*60}")

        # Model info
        print(f"Model: {self.MODEL_NAME}")
        print(
            f"Architecture: Dual-branch TNF ({self.MODEL_CONFIG['optical_channels']}ch optical + {self.MODEL_CONFIG['sar_channels']}ch SAR)"
        )
        print(
            f"Feature dimensions: {self.MODEL_CONFIG['optical_feature_dim']} (opt) / {self.MODEL_CONFIG['sar_feature_dim']} (sar)"
        )
        print(f"Branch weights: {self.MODEL_CONFIG['branch_weights']}")

        # Training info
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {self.BATCH_SIZE}")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Learning rate: {self.LEARNING_RATE}")
        print(f"  Device: {self.DEVICE}")
        print(f"  Mixed precision: {self.MIXED_PRECISION}")
        print(f"  Weighted sampler: {self.USE_WEIGHTED_SAMPLER}")

        # Data info - IMPROVED DISPLAY
        print(f"\nData Configuration:")
        print(f"  Image size: {self.DATA_CONFIG['image_size']}x{self.DATA_CONFIG['image_size']}")
        print(
            f"  Raw data channels: {self.DATA_CONFIG['raw_data_channels']['total_raw']} "
            f"({self.DATA_CONFIG['raw_data_channels']['optical_raw']} optical + "
            f"{self.DATA_CONFIG['raw_data_channels']['sar_raw']} SAR)"
        )
        print(
            f"  Model input channels: {self.DATA_CONFIG['model_input_channels']['total_input']} "
            f"({self.DATA_CONFIG['model_input_channels']['optical_input']} optical + "
            f"{self.DATA_CONFIG['model_input_channels']['sar_input']} SAR)"
        )
        print(f"  Validation split: {self.VALIDATION_SPLIT}")

        # Loss info
        print(f"\nLoss Configuration:")
        print(f"  Focal (α={self.LOSS_CONFIG['focal_alpha']}, γ={self.LOSS_CONFIG['focal_gamma']})")
        print(
            f"  Combination weights: Focal {self.LOSS_CONFIG['focal_weight']}, Dice {self.LOSS_CONFIG['dice_weight']}"
        )

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
    print(f"❌ Configuration validation failed: {e}")
    print("Please check the configuration and fix the errors before proceeding.")
    raise
except Exception as e:
    print(f"⚠️ Configuration setup warning: {e}")


def init_training():
    """初始化训练模式"""
    if config.RUN_ID is None:
        config.setup_run_paths()
        print("🎯 Training mode initialized")


def init_inference():
    """初始化推理模式"""
    # 确保输出目录存在，但不创建训练目录
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("🎯 Inference mode initialized")


print("✅ MM-TNF configuration loaded (manual initialization required)")
