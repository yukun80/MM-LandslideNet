# =============================================================================
# lightning_landslide/src/data/multimodal_datamodule.py - 多模态数据模块
# =============================================================================

"""
多模态数据模块 - Lightning DataModule实现

这个模块是您项目数据处理的核心，它继承了PyTorch Lightning的
DataModule类，提供了标准化的数据加载和处理流程。

设计理念：
1. 配置驱动：所有参数都通过配置文件控制
2. 灵活性：支持不同的数据使用模式（仅光学、多模态等）
3. 可重现性：固定随机种子，确保数据分割一致
4. 高效性：支持多进程数据加载和内存优化
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Any, List, Callable
import logging
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
import numpy as np

from .multimodal_dataset import MultiModalDataset, create_train_dataset, create_test_dataset
from .base import BaseDataModule

logger = logging.getLogger(__name__)


class MultiModalDataModule(pl.LightningDataModule):
    """
    多模态遥感数据的Lightning数据模块

    这个类是您项目的数据处理中心。它负责：
    1. 管理多通道遥感数据的加载
    2. 处理训练/验证/测试数据集的划分
    3. 配置数据增强和预处理策略
    4. 提供标准化的数据加载器接口

    与latent-diffusion的数据处理方式对比：
    - 同样支持配置驱动的参数设置
    - 提供灵活的数据增强策略
    - 支持多种数据使用模式
    """

    def __init__(
        self,
        # 数据路径配置
        train_data_dir: str,
        test_data_dir: str,
        train_csv: str,
        test_csv: str,
        exclude_ids_file: Optional[str] = None,
        # 🔧 新增：跨目录映射配置
        cross_directory_mapping: Optional[str] = None,
        # 通道配置
        channel_config: Dict[str, Any] = None,
        active_mode: str = "optical_only",
        # 数据加载配置
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        # 数据分割配置
        val_split: float = 0.2,
        stratify: bool = True,
        use_weighted_sampling: bool = False,
        # 数据预处理配置
        preprocessing: Optional[Dict] = None,
        augmentation: Optional[Dict] = None,
        # 其他配置
        seed: int = 3407,
        **kwargs,
    ):
        """
        初始化多模态数据模块

        Args:
            train_data_dir: 训练数据目录
            test_data_dir: 测试数据目录
            train_csv: 训练数据标签文件
            test_csv: 测试数据标签文件
            exclude_ids_file: 需要排除的样本ID文件
            cross_directory_mapping: 跨目录数据路径映射文件 (JSON格式)
            channel_config: 通道配置字典
            active_mode: 当前使用的数据模式
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            pin_memory: 是否将数据固定在内存中
            shuffle_train: 是否打乱训练数据
            val_split: 验证集划分比例
            stratify: 是否进行分层划分
            use_weighted_sampling: 是否使用加权采样
            preprocessing: 预处理配置
            augmentation: 数据增强配置
            seed: 随机种子
        """
        super().__init__()

        # 保存所有参数（用于Lightning的超参数记录）
        self.save_hyperparameters()

        # 数据路径
        self.train_data_dir = Path(train_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.exclude_ids_file = exclude_ids_file

        # 🔧 新增：跨目录映射支持
        self.cross_directory_mapping = cross_directory_mapping

        # 通道和模式配置
        self.channel_config = channel_config or self._get_default_channel_config()
        self.active_mode = active_mode

        # 数据加载配置
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

        # 数据分割配置
        self.val_split = val_split
        self.stratify = stratify
        self.use_weighted_sampling = use_weighted_sampling
        self.seed = seed

        # 预处理配置
        self.preprocessing = preprocessing or {}
        self.augmentation = augmentation or {}

        # 随机种子
        self.seed = seed

        # 数据集对象（在setup中初始化）
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 数据统计信息
        self._data_stats = {}

        # 🔧 加载跨目录映射文件
        self._cross_directory_mapping_dict = self._load_cross_directory_mapping()

        logger.info("🔢MultiModalDataModule initialized" + "-" * 100)
        logger.info(f"Active mode: {self.active_mode}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")
        logger.info(f"Validation split: {self.val_split}")
        if self.cross_directory_mapping:
            logger.info(f"🎩 Cross-directory mapping: {len(self._cross_directory_mapping_dict)} samples")
        logger.info("-" * 100)

    def _load_cross_directory_mapping(self) -> Dict[str, str]:
        """加载跨目录数据路径映射"""
        if not self.cross_directory_mapping:
            return {}

        mapping_file = Path(self.cross_directory_mapping)
        if not mapping_file.exists():
            logger.warning(f"Cross-directory mapping file not found: {mapping_file}")
            return {}

        try:
            import json

            with open(mapping_file, "r") as f:
                mapping = json.load(f)

            logger.info(f"📁 Loaded cross-directory mapping: {len(mapping)} samples")
            return mapping
        except Exception as e:
            logger.error(f"Failed to load cross-directory mapping: {e}")
            return {}

    def _create_transforms(self, stage: str) -> Optional[Callable]:
        """
        根据阶段创建数据变换

        这个方法实现了类似latent-diffusion的数据增强策略，
        根据训练/测试阶段应用不同的变换。

        Args:
            stage: 数据阶段 ('train', 'val', 'test')

        Returns:
            数据变换函数
        """
        if stage not in self.augmentation:
            return None

        # 这里可以根据配置创建具体的数据变换
        # 例如：随机翻转、旋转、噪声添加等
        # 具体实现可以参考albumentations或torchvision

        transforms = []
        stage_config = self.augmentation.get(stage, {})

        # 示例：几何变换
        if stage_config.get("geometric", {}).get("random_flip", False):
            # transforms.append(RandomHorizontalFlip())
            pass

        # 示例：光谱增强
        if stage_config.get("spectral", {}):
            # transforms.append(SpectralNoise())
            pass

        return None  # 暂时返回None，您可以根据需要实现具体变换

    def prepare_data(self) -> None:
        """
        数据准备阶段（全局执行一次），检查数据是否都存在

        这个方法遵循Lightning的设计模式，只在主进程中执行一次。
        主要用于：
        1. 验证数据文件存在性
        2. 执行一次性的数据预处理
        3. 创建必要的目录结构
        """
        logger.info("🔍 Preparing data...")

        # 验证数据目录存在
        if not self.train_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.train_data_dir}")
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")

        # 验证CSV文件存在
        if not self.train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {self.train_csv}")
        if not self.test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {self.test_csv}")

        logger.info("Data preparation completed")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集（每个进程执行）

        这是数据处理的核心方法。它根据不同的stage创建相应的数据集。
        """
        logger.info(f"setup:Setting up datasets for stage: {stage}")

        if stage == "fit" or stage is None:
            # 创建完整的训练数据集
            full_dataset = create_train_dataset(
                data_dir=str(self.train_data_dir),
                csv_file=str(self.train_csv),
                exclude_ids_file=self.exclude_ids_file,
                transform=self._create_transforms("train"),
                channel_config=self.channel_config,
                usage_mode=self.active_mode,
                # 🔧 传递跨目录映射
                cross_directory_mapping=self._cross_directory_mapping_dict,
            )

            # 分割训练集和验证集
            if self.val_split > 0:
                total_size = len(full_dataset)
                val_size = int(total_size * self.val_split)
                train_size = total_size - val_size

                if self.stratify and full_dataset.has_labels:
                    labels = full_dataset.data_index["label"].tolist()

                    # 分层采样：保持类别比例
                    train_indices, val_indices = train_test_split(
                        range(len(full_dataset)), test_size=self.val_split, stratify=labels, random_state=self.seed
                    )

                    self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                    self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

                    logger.info(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                else:
                    self.train_dataset = full_dataset
                    self.val_dataset = None
                    logger.info(f"Using full dataset for training: {len(self.train_dataset)} samples")

        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = create_test_dataset(
                data_dir=str(self.test_data_dir),
                csv_file=str(self.test_csv),
                transform=self._create_transforms("test"),
                channel_config=self.channel_config,
                usage_mode=self.active_mode,
            )
            logger.info(f"Test dataset: {len(self.test_dataset)} samples")

        logger.info("-" * 100)

    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # 训练时丢弃最后一个不完整的批次
            persistent_workers=self.num_workers > 0,  # 保持工作进程活跃
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """创建验证数据加载器"""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """创建测试数据加载器"""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        """创建预测数据加载器"""
        return self.test_dataloader()

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据模块信息"""
        return {
            "data_stats": self._data_stats,
            "channel_config": self.channel_config,
            "active_mode": self.active_mode,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
