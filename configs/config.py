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

    # 基础模型参数
    MODEL_NAME = "MultiModalCNN"
    NUM_CLASSES = 1  # 二元分类

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
