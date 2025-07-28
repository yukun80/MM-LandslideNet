# =============================================================================
# lightning_landslide/src/data/kfold_datamodule.py - K折交叉验证数据模块
# =============================================================================

"""
K折交叉验证数据模块 - 专为Kaggle竞赛设计

这个模块实现了竞赛级的K折交叉验证策略，包括：
1. 分层K折分割确保每折类别分布一致
2. OOF(Out-of-Fold)预测管理
3. 测试集预测一致性处理
4. 种子管理确保可重现性
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Optional, Dict, Any, List, Callable, Tuple
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json
from copy import deepcopy

from .multimodal_dataset import create_train_dataset, create_test_dataset
from .base import BaseDataModule

logger = logging.getLogger(__name__)


class KFoldDataModule(pl.LightningDataModule):
    """
    K折交叉验证数据模块

    设计理念：
    1. 竞赛导向：专门为Kaggle竞赛设计的K折验证
    2. 性能优先：确保每折的数据分布和模型性能
    3. 可重现性：严格的种子管理和状态保存
    4. 集成友好：为模型集成和堆叠提供支持
    """

    def __init__(
        self,
        # 基础数据配置
        train_data_dir: str,
        test_data_dir: str,
        train_csv: str,
        test_csv: str,
        exclude_ids_file: Optional[str] = None,
        # K折配置
        n_splits: int = 5,
        current_fold: int = 0,
        stratified: bool = True,
        # 通道配置
        channel_config: Dict[str, Any] = None,
        active_mode: str = "optical_only",
        # 数据加载配置
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        # 预处理配置
        preprocessing: Optional[Dict] = None,
        augmentation: Optional[Dict] = None,
        # K折管理配置
        save_fold_info: bool = True,
        fold_info_dir: str = "outputs/kfold_info",
        # 其他配置
        seed: int = 3407,
        **kwargs,
    ):
        """
        初始化K折数据模块

        Args:
            n_splits: K折数量
            current_fold: 当前训练的折数(0-based)
            stratified: 是否使用分层采样
            save_fold_info: 是否保存折信息
            fold_info_dir: 折信息保存目录
        """
        super().__init__()

        # 保存超参数
        self.save_hyperparameters()

        # 基础配置
        self.train_data_dir = Path(train_data_dir)
        self.test_data_dir = Path(test_data_dir)
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.exclude_ids_file = exclude_ids_file

        # K折配置
        self.n_splits = n_splits
        self.current_fold = current_fold
        self.stratified = stratified

        # 数据配置
        self.channel_config = channel_config
        self.active_mode = active_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

        # 预处理配置
        self.preprocessing = preprocessing or {}
        self.augmentation = augmentation or {}

        # K折管理
        self.save_fold_info = save_fold_info
        self.fold_info_dir = Path(fold_info_dir)
        self.fold_info_dir.mkdir(parents=True, exist_ok=True)

        # 种子管理
        self.seed = seed

        # 数据集和折信息
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.fold_indices = None

        logger.info(f"🎯 KFoldDataModule initialized for {n_splits}-fold CV")
        logger.info(f"Current fold: {current_fold}/{n_splits-1}")
        logger.info(f"Stratified: {stratified}")

    def prepare_data(self) -> None:
        """准备数据和K折分割"""
        # 验证数据文件存在
        assert self.train_data_dir.exists(), f"Training data directory not found: {self.train_data_dir}"
        assert self.test_data_dir.exists(), f"Test data directory not found: {self.test_data_dir}"
        assert self.train_csv.exists(), f"Training CSV not found: {self.train_csv}"
        assert self.test_csv.exists(), f"Test CSV not found: {self.test_csv}"

        logger.info("✓ Data files validation passed")

    def setup(self, stage: Optional[str] = None) -> None:
        """设置K折数据集"""
        if stage == "fit" or stage is None:
            # 创建完整的训练数据集
            self.full_dataset = create_train_dataset(
                data_dir=str(self.train_data_dir),
                csv_file=str(self.train_csv),
                exclude_ids_file=self.exclude_ids_file,
                transform=None,  # 基础变换，后续在DataLoader中处理
                channel_config=self.channel_config,
                usage_mode=self.active_mode,
            )

            # 生成或加载K折分割
            self.fold_indices = self._generate_kfold_splits()

            # 获取当前折的数据
            train_indices, val_indices = self.fold_indices[self.current_fold]

            # 创建当前折的数据集
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)

            logger.info(f"Fold {self.current_fold}: Train={len(train_indices)}, Val={len(val_indices)}")

            # 保存当前折信息
            if self.save_fold_info:
                self._save_current_fold_info(train_indices, val_indices)

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

    def _generate_kfold_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成K折分割索引

        Returns:
            List of (train_indices, val_indices) for each fold
        """
        fold_file = self.fold_info_dir / f"{self.n_splits}fold_splits.json"

        # 尝试加载已存在的分割
        if fold_file.exists():
            logger.info(f"Loading existing K-fold splits from {fold_file}")
            with open(fold_file, "r") as f:
                fold_data = json.load(f)
                return [(np.array(fold["train"]), np.array(fold["val"])) for fold in fold_data["folds"]]

        # 生成新的分割
        logger.info(f"Generating new {self.n_splits}-fold splits...")

        # 获取标签用于分层采样
        labels = []
        for i in range(len(self.full_dataset)):
            _, label = self.full_dataset[i]
            labels.append(label)
        labels = np.array(labels)

        # 创建K折分割器
        if self.stratified:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(kfold.split(np.arange(len(labels)), labels))
        else:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(kfold.split(np.arange(len(labels))))

        # 保存分割信息
        if self.save_fold_info:
            fold_data = {
                "n_splits": self.n_splits,
                "stratified": self.stratified,
                "seed": self.seed,
                "total_samples": len(labels),
                "class_distribution": {"negative": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1))},
                "folds": [],
            }

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_info = {
                    "fold": fold_idx,
                    "train": train_idx.tolist(),
                    "val": val_idx.tolist(),
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "train_pos_ratio": float(np.mean(labels[train_idx])),
                    "val_pos_ratio": float(np.mean(labels[val_idx])),
                }
                fold_data["folds"].append(fold_info)

            with open(fold_file, "w") as f:
                json.dump(fold_data, f, indent=2)

            logger.info(f"K-fold splits saved to {fold_file}")

        return splits

    def _save_current_fold_info(self, train_indices: np.ndarray, val_indices: np.ndarray) -> None:
        """保存当前折的详细信息"""
        fold_info = {
            "fold": self.current_fold,
            "n_splits": self.n_splits,
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "seed": self.seed,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        current_fold_file = self.fold_info_dir / f"fold_{self.current_fold}_info.json"
        with open(current_fold_file, "w") as f:
            json.dump(fold_info, f, indent=2)

    def _create_transforms(self, stage: str) -> Optional[Callable]:
        """创建数据变换（占位符，需要根据实际需求实现）"""
        return None

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
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """创建测试数据加载器"""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test') first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """创建预测数据加载器"""
        return self.test_dataloader()

    def get_fold_info(self) -> Dict[str, Any]:
        """获取当前折信息"""
        return {
            "current_fold": self.current_fold,
            "n_splits": self.n_splits,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
        }

    def get_all_folds_info(self) -> List[Dict[str, Any]]:
        """获取所有折的信息"""
        fold_file = self.fold_info_dir / f"{self.n_splits}fold_splits.json"
        if fold_file.exists():
            with open(fold_file, "r") as f:
                return json.load(f)["folds"]
        return []

    @property
    def is_last_fold(self) -> bool:
        """检查是否为最后一折"""
        return self.current_fold == self.n_splits - 1

    def summary(self) -> None:
        """打印K折数据模块摘要"""
        print("\n" + "=" * 60)
        print(f"K-Fold Cross Validation Data Module")
        print("=" * 60)
        print(f"Number of folds: {self.n_splits}")
        print(f"Current fold: {self.current_fold}")
        print(f"Stratified: {self.stratified}")
        print(f"Seed: {self.seed}")

        if self.fold_indices:
            train_indices, val_indices = self.fold_indices[self.current_fold]
            print(f"Current fold - Train: {len(train_indices)}, Val: {len(val_indices)}")

        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.num_workers}")
        print("=" * 60 + "\n")


# 便利函数：创建K折数据模块
def create_kfold_datamodule(config: Dict[str, Any], current_fold: int) -> KFoldDataModule:
    """
    从配置创建K折数据模块的便利函数

    Args:
        config: 数据模块配置
        current_fold: 当前折索引

    Returns:
        配置好的K折数据模块
    """
    config = deepcopy(config)
    config["current_fold"] = current_fold

    return KFoldDataModule(**config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 示例配置
    config = {
        "train_data_dir": "dataset/train_data",
        "test_data_dir": "dataset/test_data",
        "train_csv": "dataset/Train.csv",
        "test_csv": "dataset/Test.csv",
        "n_splits": 5,
        "current_fold": 0,
        "batch_size": 16,
        "num_workers": 4,
    }

    # 创建K折数据模块
    dm = KFoldDataModule(**config)
    dm.prepare_data()
    dm.setup("fit")
    dm.summary()

    print("✓ KFoldDataModule test completed!")
