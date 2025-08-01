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
import json

from .multimodal_dataset import MultiModalDataset, create_train_dataset, create_test_dataset

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
        # 跨目录映射配置
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
            train_csv: 训练数据标签文件（清洁数据）
            test_csv: 测试数据标签文件
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

        # 跨目录映射支持
        self.cross_directory_mapping = cross_directory_mapping

        # 通道和模式配置
        self.channel_config = channel_config
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

        logger.info("🔢MultiModalDataModule initialized" + "=" * 100)
        logger.info(f"Active mode: {self.active_mode}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")
        logger.info(f"Validation split: {self.val_split}")
        if self.cross_directory_mapping:
            logger.info(f"🔗 Cross-directory mapping: {len(self._cross_directory_mapping_dict)} samples")

        logger.info("=" * 80)

    def _load_cross_directory_mapping(self) -> Dict[str, str]:
        """
        加载跨目录映射配置（简化版）

        简化说明：
        - 移除了复杂的错误处理和日志
        - 专注于核心功能

        Returns:
            Dict[str, str]: 样本ID到完整路径的映射
        """
        if not self.cross_directory_mapping:
            return {}

        mapping_file = Path(self.cross_directory_mapping)
        if not mapping_file.exists():
            logger.warning(f"⚠️ Cross-directory mapping file not found: {mapping_file}")
            return {}

        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)

            logger.info(f"📂 Loaded cross-directory mapping: {len(mapping_data)} entries")
            return mapping_data

        except Exception as e:
            logger.error(f"❌ Error loading cross-directory mapping: {e}")
            return {}

    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集（每个进程执行）

        这个方法在分布式训练的每个进程中都会被调用。
        简化后的逻辑更加清晰直接。

        Args:
            stage: 当前阶段 ('fit', 'validate', 'test', 'predict')
        """
        logger.info(f"🔧 Setting up data for stage: {stage}")

        if stage == "fit" or stage is None:
            # 创建完整的训练数据集
            full_dataset = create_train_dataset(
                data_dir=str(self.train_data_dir),
                csv_file=str(self.train_csv),
                transform=self._create_transforms("train"),  # ✅ 使用原始方法
                channel_config=self.channel_config,
                usage_mode=self.active_mode,
                cross_directory_mapping=self._cross_directory_mapping_dict,
            )

            # 数据分割
            self.train_dataset, self.val_dataset = self._split_dataset(
                full_dataset,
                self._create_transforms("train"),  # ✅ 训练变换
                self._create_transforms("val"),  # ✅ 验证变换
            )

            logger.info(f"✅ Train dataset: {len(self.train_dataset)} samples")
            logger.info(f"✅ Val dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = create_test_dataset(
                data_dir=str(self.test_data_dir),
                csv_file=str(self.test_csv),
                transform=self._create_transforms("test"),  # ✅ 使用原始方法
                channel_config=self.channel_config,
                usage_mode=self.active_mode,
            )

            logger.info(f"✅ Test dataset: {len(self.test_dataset)} samples")

        if stage == "predict":
            # 预测阶段使用测试数据集
            if self.test_dataset is None:
                self.test_dataset = create_test_dataset(
                    data_dir=str(self.test_data_dir),
                    csv_file=str(self.test_csv),
                    transform=self._create_transforms("test"),  # ✅ 使用原始方法
                    channel_config=self.channel_config,
                    usage_mode=self.active_mode,
                )

    def _create_transforms(self, stage: str) -> Optional[Callable]:
        """
        根据阶段创建数据变换（保持原始实现）

        Args:
            stage: 数据阶段 ('train', 'val', 'test')

        Returns:
            数据变换函数，如果没有配置则返回None
        """
        if stage not in self.augmentation:
            return None

        # 🔧 这里保持原始的简单实现
        # 如果需要复杂变换，可以后续扩展
        stage_config = self.augmentation.get(stage, {})

        # 目前返回None，让数据集类处理原始数据
        # 这是最简单、最稳定的方案
        return None

    def _split_dataset(self, full_dataset, train_transform, val_transform):
        """
        分割数据集为训练集和验证集

        Args:
            full_dataset: 完整数据集
            train_transform: 训练数据变换
            val_transform: 验证数据变换

        Returns:
            Tuple: (训练数据集, 验证数据集)
        """
        # 获取数据索引
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size

        # 分层划分（如果启用）
        if self.stratify and full_dataset.has_labels:
            # 基于标签进行分层划分
            labels = [full_dataset.data_index.iloc[i]["label"] for i in range(total_size)]
            train_indices, val_indices = train_test_split(
                range(total_size), test_size=self.val_split, stratify=labels, random_state=self.seed
            )
        else:
            # 随机划分
            torch.manual_seed(self.seed)
            train_indices, val_indices = random_split(range(total_size), [train_size, val_size])
            train_indices = train_indices.indices
            val_indices = val_indices.indices

        # 创建训练和验证数据集
        train_dataset = DatasetSubset(full_dataset, train_indices, train_transform)
        val_dataset = DatasetSubset(full_dataset, val_indices, val_transform)

        return train_dataset, val_dataset

    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """创建测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """创建预测数据加载器"""
        return self.test_dataloader()

    def _get_test_transforms(self):
        """获取测试数据变换"""
        from .transforms import get_test_transforms

        return get_test_transforms(self.augmentation.get("test", {}))


class DatasetSubset:
    """
    数据集子集包装器

    用于将完整数据集分割为训练/验证子集，
    并为每个子集应用不同的数据变换。
    """

    def __init__(self, dataset, indices, transform=None):
        """
        初始化数据集子集

        Args:
            dataset: 原始数据集
            indices: 子集索引列表
            transform: 数据变换函数
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取原始数据
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label
