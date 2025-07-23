"""
Lightning数据模块实现

这个模块将我们的MultiModalDataset和数据变换整合到PyTorch Lightning的
数据处理框架中。它处理数据的完整生命周期：

1. 数据准备：验证文件存在性，加载配置
2. 数据设置：创建训练/验证数据集，应用变换
3. 数据加载：创建高效的DataLoader

设计亮点：
- 自动的训练/验证分割
- 类别平衡处理
- 灵活的数据增强配置
- 完善的错误处理和统计信息

教学要点：
Lightning数据模块的设计遵循"关注点分离"原则。数据处理的逻辑
与训练逻辑完全分离，这样我们可以独立地优化和测试数据管道。
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from omegaconf import DictConfig
import numpy as np

from .base import BaseDataModule
from .multimodal_dataset import MultiModalDataset, create_train_dataset, create_test_dataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms, compute_channel_statistics

logger = logging.getLogger(__name__)


class MultiModalDataModule(BaseDataModule):
    """
    光学数据的Lightning数据模块

    这个类管理着整个数据处理流程。它的设计哲学是"智能默认，灵活配置"：
    - 提供合理的默认配置，让用户能快速开始
    - 支持细粒度的自定义配置，满足高级用户需求
    - 自动处理常见的数据问题（类别不平衡、数据分割等）
    """

    def __init__(self, cfg: DictConfig):
        """
        初始化光学数据模块

        Args:
            cfg: 完整的配置对象，包含所有数据相关配置
        """
        super().__init__(cfg)

        # 数据路径配置
        self.train_data_dir = Path(cfg.data.train_data_dir)
        self.test_data_dir = Path(cfg.data.get("test_data_dir", cfg.data.train_data_dir))
        self.train_csv = Path(cfg.data.train_csv)
        self.test_csv = Path(cfg.data.get("test_csv", "dataset/Test.csv"))
        self.exclude_ids_file = Path(cfg.data.exclude_ids_file) if cfg.data.get("exclude_ids_file") else None

        # 数据分割配置
        self.val_split = cfg.data.get("val_split", 0.2)
        self.stratify = cfg.data.get("stratify", True)  # 是否按标签分层采样

        # 采样策略配置
        self.use_weighted_sampling = cfg.data.get("use_weighted_sampling", False)
        self.class_weights = None

        # 数据增强配置
        self.augmentation_config = cfg.data.get("transforms", {})

        # 数据变换对象（在setup中初始化）
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        # 数据统计信息
        self._data_stats = {}

        logger.info(f"MultiModalDataModule initialized")
        logger.info(f"Train data: {self.train_data_dir}")
        logger.info(f"Validation split: {self.val_split}")
        logger.info(f"Weighted sampling: {self.use_weighted_sampling}")

    def prepare_data(self) -> None:
        """
        数据准备阶段（全局执行一次）

        这个方法验证所有必需的数据文件是否存在，
        为后续的数据处理做好准备。
        """
        logger.info("Preparing data...")

        # 验证必需文件存在
        required_paths = [str(self.train_data_dir), str(self.train_csv)]

        # 检查排除列表文件
        if self.exclude_ids_file:
            required_paths.append(str(self.exclude_ids_file))

        try:
            self.validate_data_paths(required_paths)
        except FileNotFoundError as e:
            logger.error(f"Data preparation failed: {e}")
            raise

        # 验证数据目录结构
        self._validate_data_structure()

        logger.info("✓ Data preparation completed successfully")

    def _validate_data_structure(self) -> None:
        """验证数据目录结构"""
        # 检查训练数据目录是否包含.npy文件
        npy_files = list(self.train_data_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {self.train_data_dir}")

        logger.info(f"Found {len(npy_files)} data files in training directory")

        # 检查CSV文件格式
        try:
            import pandas as pd

            df = pd.read_csv(self.train_csv)
            if "ID" not in df.columns:
                raise ValueError("Train CSV must contain 'ID' column")

            logger.info(f"Train CSV contains {len(df)} samples")

        except Exception as e:
            raise RuntimeError(f"Invalid train CSV format: {e}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集（每个进程执行）

        这是数据模块的核心方法，负责：
        1. 创建数据变换管道
        2. 创建数据集对象
        3. 进行训练/验证分割
        4. 设置采样策略

        Args:
            stage: 训练阶段 ('fit', 'test', 'predict' 或 None)
        """
        logger.info(f"Setting up data for stage: {stage}")

        # 创建数据变换
        self._setup_transforms()

        if stage == "fit" or stage is None:
            self._setup_train_val_datasets()

        if stage == "test" or stage is None:
            self._setup_test_dataset()

        # 计算和记录数据统计信息
        self._compute_data_statistics()

        logger.info("✓ Data setup completed")

    def _setup_transforms(self) -> None:
        """设置数据变换管道"""
        try:
            self.train_transform = get_train_transforms(self.cfg)
            self.val_transform = get_val_transforms(self.cfg)
            self.test_transform = get_test_transforms(self.cfg)

            logger.info("✓ Data transforms created successfully")

        except Exception as e:
            logger.warning(f"Failed to create configured transforms: {e}")
            logger.info("Using default transforms")

            # 使用默认变换作为fallback
            from .transforms import create_default_transforms

            self.train_transform, self.val_transform = create_default_transforms()
            self.test_transform = self.val_transform

    def _setup_train_val_datasets(self) -> None:
        """设置训练和验证数据集"""

        # 从配置中提取通道相关参数
        channel_config = self.cfg.data.get("channel_config", None)
        usage_mode = self.cfg.data.get("usage_mode", "optical_only")

        # 使用便捷函数创建数据集，保持原有的接口风格
        full_dataset = create_train_dataset(
            data_dir=str(self.train_data_dir),
            csv_file=str(self.train_csv),
            exclude_ids_file=str(self.exclude_ids_file) if self.exclude_ids_file else None,
            transform=None,  # 变换将在分割后单独应用
            channel_config=channel_config,  # 新增的参数
            usage_mode=usage_mode,  # 新增的参数
        )

        logger.info(f"Created full dataset with {len(full_dataset)} samples")

        # 进行训练/验证分割
        self._split_train_val_datasets(full_dataset)

        # # 设置加权采样（如果需要）
        # if self.use_weighted_sampling:
        #     self._setup_weighted_sampling()

    def _split_train_val_datasets(self, full_dataset: MultiModalDataset) -> None:
        """
        分割训练和验证数据集

        支持两种分割方式：
        1. 随机分割：简单随机采样
        2. 分层分割：保持各类别比例

        Args:
            full_dataset: 完整的数据集
        """
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        logger.info(f"Splitting dataset: {train_size} train, {val_size} val")

        if self.stratify and full_dataset.has_labels:
            # 分层分割，保持类别平衡
            self._stratified_split(full_dataset, train_size, val_size)
        else:
            # 简单随机分割
            self._random_split(full_dataset, train_size, val_size)

    def _stratified_split(self, full_dataset: MultiModalDataset, train_size: int, val_size: int) -> None:
        """
        分层分割数据集

        确保训练集和验证集中各类别的比例与原数据集相同。
        这对于类别不平衡的数据集特别重要。
        """
        try:
            from sklearn.model_selection import train_test_split

            # 获取所有样本的标签
            labels = full_dataset.data_index["label"].values
            indices = np.arange(len(full_dataset))

            # 分层分割
            train_indices, val_indices = train_test_split(
                indices, test_size=self.val_split, stratify=labels, random_state=self.cfg.training.get("seed", 42)
            )

            logger.info("✓ Stratified split completed")

        except ImportError:
            logger.warning("sklearn not available, falling back to random split")
            self._random_split(full_dataset, train_size, val_size)
            return
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}, falling back to random split")
            self._random_split(full_dataset, train_size, val_size)
            return

        # 创建训练和验证数据集
        self._create_split_datasets(full_dataset, train_indices, val_indices)

    def _random_split(self, full_dataset: MultiModalDataset, train_size: int, val_size: int) -> None:
        """简单随机分割"""
        generator = torch.Generator()
        generator.manual_seed(self.cfg.training.get("seed", 42))

        train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

        train_indices = train_indices.indices
        val_indices = val_indices.indices

        logger.info("✓ Random split completed")

        # 创建数据集
        self._create_split_datasets(full_dataset, train_indices, val_indices)

    def _create_split_datasets(
        self, full_dataset: MultiModalDataset, train_indices: np.ndarray, val_indices: np.ndarray
    ) -> None:
        """
        根据索引创建分割后的数据集

        这个方法创建两个独立的数据集对象，分别应用训练和验证的变换。
        """
        # 创建训练数据集
        self.train_dataset = MultiModalDataset(
            data_dir=full_dataset.data_dir,
            csv_file=full_dataset.csv_file,
            exclude_ids_file=None,  # 已经在full_dataset中处理过了
            transform=self.train_transform,
            channels=full_dataset.channels,
            compute_ndvi=full_dataset.compute_ndvi,
        )

        # 只保留训练索引对应的样本
        self.train_dataset.data_index = full_dataset.data_index.iloc[train_indices].reset_index(drop=True)

        # 创建验证数据集
        self.val_dataset = MultiModalDataset(
            data_dir=full_dataset.data_dir,
            csv_file=full_dataset.csv_file,
            exclude_ids_file=None,
            transform=self.val_transform,
            channels=full_dataset.channels,
            compute_ndvi=full_dataset.compute_ndvi,
        )

        # 只保留验证索引对应的样本
        self.val_dataset.data_index = full_dataset.data_index.iloc[val_indices].reset_index(drop=True)

        logger.info(f"Created train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Created val dataset: {len(self.val_dataset)} samples")

        # 记录类别分布
        if full_dataset.has_labels:
            train_dist = self.train_dataset.get_class_distribution()
            val_dist = self.val_dataset.get_class_distribution()
            logger.info(f"Train class distribution: {train_dist}")
            logger.info(f"Val class distribution: {val_dist}")

    def _setup_test_dataset(self) -> None:
        """设置测试数据集"""
        if not self.test_csv.exists():
            logger.info("Test CSV not found, skipping test dataset setup")
            return

        try:
            # 使用相同的配置参数
            channel_config = self.cfg.data.get("channel_config", None)
            usage_mode = self.cfg.data.get("usage_mode", "optical_only")

            self.test_dataset = create_test_dataset(
                data_dir=str(self.test_data_dir),
                csv_file=str(self.test_csv),
                transform=self.test_transform,
                channel_config=channel_config,
                usage_mode=usage_mode,
            )

            logger.info(f"Created test dataset: {len(self.test_dataset)} samples")

        except Exception as e:
            logger.warning(f"Failed to create test dataset: {e}")
            self.test_dataset = None

    def _setup_weighted_sampling(self) -> None:
        """
        设置加权采样以处理类别不平衡

        计算每个类别的权重，让模型在训练时更关注少数类别。
        这是处理类别不平衡问题的有效方法之一。
        """
        if not self.train_dataset or not self.train_dataset.has_labels:
            logger.warning("Cannot setup weighted sampling: no labels available")
            return

        # 计算类别权重
        class_counts = self.train_dataset.get_class_distribution()
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)

        # 计算平衡权重：权重 = 总样本数 / (类别数 * 该类别样本数)
        class_weights = {}
        for class_id, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[class_id] = weight

        self.class_weights = class_weights
        logger.info(f"Computed class weights for sampling: {class_weights}")

        # 创建样本权重列表
        sample_weights = []
        for idx in range(len(self.train_dataset)):
            _, label = self.train_dataset[idx]
            weight = class_weights[label.item()]
            sample_weights.append(weight)

        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        logger.info("✓ Weighted sampling setup completed")

    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        # 设置采样器
        sampler = None
        shuffle = self.shuffle_train

        if self.use_weighted_sampling and hasattr(self, "sample_weights"):
            sampler = WeightedRandomSampler(
                weights=self.sample_weights, num_samples=len(self.sample_weights), replacement=True
            )
            shuffle = False  # 使用采样器时不能shuffle
            logger.info("Using weighted random sampler for training")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # 丢弃最后一个不完整的批次，保证批次大小一致
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # 验证时不丢弃数据
            persistent_workers=True if self.num_workers > 0 else False,
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
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def _compute_data_statistics(self) -> None:
        """计算数据统计信息"""
        self._data_stats = {
            "data_module": self.__class__.__name__,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "val_split": self.val_split,
            "stratified": self.stratify,
            "weighted_sampling": self.use_weighted_sampling,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

        # 添加类别分布信息
        if self.train_dataset and self.train_dataset.has_labels:
            self._data_stats["train_class_dist"] = self.train_dataset.get_class_distribution()

        if self.val_dataset and self.val_dataset.has_labels:
            self._data_stats["val_class_dist"] = self.val_dataset.get_class_distribution()

        # 添加类别权重信息
        if self.class_weights:
            self._data_stats["class_weights"] = self.class_weights

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据模块的详细信息"""
        base_info = super().get_data_info()
        base_info.update(self._data_stats)
        return base_info

    def compute_normalization_stats(self, max_samples: int = 1000) -> Dict[str, List[float]]:
        """
        计算标准化统计信息

        这个方法计算训练数据的均值和标准差，用于设置标准化参数。
        在第一次运行时调用这个方法，然后将结果保存到配置文件中。

        Args:
            max_samples: 用于计算统计信息的最大样本数

        Returns:
            包含均值和标准差的字典
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        logger.info("Computing normalization statistics...")

        # 创建一个没有标准化的数据集用于统计计算
        temp_dataset = MultiModalDataset(
            data_dir=self.train_dataset.data_dir,
            csv_file=self.train_dataset.csv_file,
            transform=None,  # 不应用任何变换
            channels=self.train_dataset.channels,
            compute_ndvi=self.train_dataset.compute_ndvi,
        )
        temp_dataset.data_index = self.train_dataset.data_index

        stats = compute_channel_statistics(temp_dataset, max_samples)

        logger.info("Normalization statistics computed:")
        logger.info(f"Means: {stats['means']}")
        logger.info(f"Stds: {stats['stds']}")

        return stats

    def get_sample_for_debug(
        self, split: str = "train", index: int = 0, apply_transform: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        获取调试样本及其详细信息

        Args:
            split: 数据集分割 ('train', 'val', 'test')
            index: 样本索引
            apply_transform: 是否应用数据变换

        Returns:
            (data, label, info) 元组
        """
        # 选择数据集
        if split == "train" and self.train_dataset:
            dataset = self.train_dataset
        elif split == "val" and self.val_dataset:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset:
            dataset = self.test_dataset
        else:
            raise ValueError(f"Dataset for split '{split}' not available")

        # 获取原始数据（不应用变换）
        original_transform = dataset.transform
        if not apply_transform:
            dataset.transform = None

        try:
            data, label = dataset[index]
            info = dataset.get_sample_info(index)
        finally:
            # 恢复原始变换
            dataset.transform = original_transform

        return data, label, info
