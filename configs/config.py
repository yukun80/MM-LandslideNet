import os
from pathlib import Path


class Config:
    """
    多模态滑坡检测项目配置文件
    """

    # ============= 项目路径配置 =============
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATASET_ROOT = PROJECT_ROOT / "dataset"

    # 数据路径
    TRAIN_CSV = DATASET_ROOT / "Train.csv"
    TEST_CSV = DATASET_ROOT / "Test.csv"
    TRAIN_DATA_DIR = DATASET_ROOT / "train_data"
    TEST_DATA_DIR = DATASET_ROOT / "test_data"
    SAMPLE_SUBMISSION = DATASET_ROOT / "SampleSubmission.csv"

    # 输出路径
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    SUBMISSION_DIR = OUTPUT_ROOT / "submissions"
    LOG_DIR = PROJECT_ROOT / "logs"

    # ============= 数据配置 =============
    # 图像尺寸和通道数
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 12

    # 多模态通道描述
    CHANNEL_DESCRIPTIONS = [
        "Red (Sentinel-2)",  # 0
        "Green (Sentinel-2)",  # 1
        "Blue (Sentinel-2)",  # 2
        "Near Infrared (Sentinel-2)",  # 3
        "Descending VV (Sentinel-1)",  # 4
        "Descending VH (Sentinel-1)",  # 5
        "Descending Diff VV",  # 6
        "Descending Diff VH",  # 7
        "Ascending VV (Sentinel-1)",  # 8
        "Ascending VH (Sentinel-1)",  # 9
        "Ascending Diff VV",  # 10
        "Ascending Diff VH",  # 11
    ]

    # 数据预处理参数
    NORMALIZE_GLOBAL = True  # 是否全局归一化
    NORMALIZE_CHANNEL_WISE = True  # 是否按通道归一化

    # ============= 模型配置 =============
    # 基础模型参数
    MODEL_NAME = "MultiModalCNN"
    NUM_CLASSES = 1  # 二元分类

    # CNN架构参数
    CNN_CHANNELS = [32, 64, 128, 256]  # 卷积层通道数
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.25

    # 分类头参数
    FC_HIDDEN_DIMS = [128, 64]
    FC_DROPOUT = 0.5

    # ============= 训练配置 =============
    # 基础训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 损失函数参数
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.5
    FOCAL_GAMMA = 2.0

    # 类别权重 (用于处理不平衡数据)
    CLASS_WEIGHTS = None  # 会在训练时动态计算

    # ============= 验证配置 =============
    VAL_SPLIT = 0.2
    STRATIFY = True  # 分层采样
    RANDOM_STATE = 42

    # 早停参数
    EARLY_STOPPING = True
    PATIENCE = 15
    MIN_DELTA = 1e-4

    # ============= 数据增强配置 =============
    USE_AUGMENTATION = True
    AUG_ROTATION = 40
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = True
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2

    # ============= 优化器配置 =============
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    SCHEDULER_T_MAX = 50

    # ============= 评估配置 =============
    # 主要评估指标
    PRIMARY_METRIC = "f1_score"

    # 监控指标
    MONITOR_METRICS = ["accuracy", "precision", "recall", "f1_score", "auc"]

    # ============= 设备配置 =============
    USE_CUDA = True
    DEVICE = "cuda" if USE_CUDA and os.path.exists("/usr/local/cuda") else "cpu"
    NUM_WORKERS = 4

    # ============= 日志配置 =============
    LOG_LEVEL = "INFO"
    SAVE_FREQUENCY = 5  # 每5个epoch保存一次

    # ============= 推理配置 =============
    TTA_ENABLE = True  # Test Time Augmentation
    TTA_STEPS = 4

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        dirs = [cls.OUTPUT_ROOT, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR, cls.LOG_DIR]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_save_path(cls, model_name, epoch=None):
        """获取模型保存路径"""
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pth"
        else:
            filename = f"{model_name}_best.pth"
        return cls.CHECKPOINT_DIR / filename

    @classmethod
    def get_submission_path(cls, model_name):
        """获取提交文件路径"""
        filename = f"submission_{model_name}.csv"
        return cls.SUBMISSION_DIR / filename
