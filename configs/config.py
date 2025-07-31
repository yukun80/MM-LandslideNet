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
    OUTPUT_ROOT = PROJECT_ROOT / "dataset"

    # ============= 数据规格配置 =============
    # 🔧 修复：添加缺失的数据维度属性
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 12

    # 🔧 修复：添加完整的通道描述信息
    CHANNEL_DESCRIPTIONS = [
        "Red (Sentinel-2)",  # Channel 0: 红光
        "Green (Sentinel-2)",  # Channel 1: 绿光
        "Blue (Sentinel-2)",  # Channel 2: 蓝光
        "Near Infrared (Sentinel-2)",  # Channel 3: 近红外
        "Descending VV (Sentinel-1)",  # Channel 4: 降轨VV极化
        "Descending VH (Sentinel-1)",  # Channel 5: 降轨VH极化
        "Descending Diff VV",  # Channel 6: 降轨VV差值
        "Descending Diff VH",  # Channel 7: 降轨VH差值
        "Ascending VV (Sentinel-1)",  # Channel 8: 升轨VV极化
        "Ascending VH (Sentinel-1)",  # Channel 9: 升轨VH极化
        "Ascending Diff VV",  # Channel 10: 升轨VV差值
        "Ascending Diff VH",  # Channel 11: 升轨VH差值
    ]

    # 基础模型参数
    MODEL_NAME = "MultiModalCNN"
    NUM_CLASSES = 1  # 二元分类

    # ============= 通道组配置 =============
    # 🔧 新增：通道分组信息，便于不同模态的处理
    CHANNEL_GROUPS = {
        "optical": {
            "name": "Sentinel-2 Optical",
            "description": "Red, Green, Blue, Near-Infrared",
            "channels": [0, 1, 2, 3],
        },
        "sar_descending": {"name": "SAR Descending", "description": "VV, VH descending pass", "channels": [4, 5]},
        "sar_desc_diff": {
            "name": "SAR Descending Diff",
            "description": "Differential VV, VH descending",
            "channels": [6, 7],
        },
        "sar_ascending": {"name": "SAR Ascending", "description": "VV, VH ascending pass", "channels": [8, 9]},
        "sar_asc_diff": {
            "name": "SAR Ascending Diff",
            "description": "Differential VV, VH ascending",
            "channels": [10, 11],
        },
    }

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

    @classmethod
    def get_channel_info(cls, channel_idx):
        """
        获取指定通道的详细信息

        Args:
            channel_idx: 通道索引 (0-11)

        Returns:
            dict: 包含通道名称、描述、所属组等信息
        """
        if not 0 <= channel_idx < cls.IMG_CHANNELS:
            raise ValueError(f"Invalid channel index: {channel_idx}, must be 0-{cls.IMG_CHANNELS-1}")

        # 找到通道所属的组
        channel_group = None
        for group_name, group_info in cls.CHANNEL_GROUPS.items():
            if channel_idx in group_info["channels"]:
                channel_group = group_name
                break

        return {
            "channel_index": channel_idx,
            "name": cls.CHANNEL_DESCRIPTIONS[channel_idx],
            "group": channel_group,
            "group_info": cls.CHANNEL_GROUPS.get(channel_group, {}) if channel_group else None,
        }

    @classmethod
    def validate_config(cls):
        """验证配置文件的完整性"""
        # 验证通道描述数量与通道数匹配
        if len(cls.CHANNEL_DESCRIPTIONS) != cls.IMG_CHANNELS:
            raise ValueError(
                f"Channel descriptions count ({len(cls.CHANNEL_DESCRIPTIONS)}) "
                f"doesn't match IMG_CHANNELS ({cls.IMG_CHANNELS})"
            )

        # 验证通道组配置
        all_channels = set()
        for group_name, group_info in cls.CHANNEL_GROUPS.items():
            group_channels = set(group_info["channels"])

            # 检查重复通道
            if all_channels & group_channels:
                raise ValueError(f"Duplicate channels found in group {group_name}")

            all_channels.update(group_channels)

            # 检查通道索引范围
            if not all(0 <= ch < cls.IMG_CHANNELS for ch in group_channels):
                raise ValueError(f"Invalid channel indices in group {group_name}")

        # 验证所有通道都被分组
        expected_channels = set(range(cls.IMG_CHANNELS))
        if all_channels != expected_channels:
            missing = expected_channels - all_channels
            extra = all_channels - expected_channels
            if missing:
                raise ValueError(f"Missing channels in groups: {missing}")
            if extra:
                raise ValueError(f"Extra channels in groups: {extra}")

        print("✅ Configuration validation passed!")
        return True
