# =============================================================================
# lightning_landslide/src/data/kfold_extension.py - K折数据模块扩展
# =============================================================================

"""
K折交叉验证数据模块扩展

这个模块的设计哲学是"扩展而不是重写"：
- 基于现有的MultiModalDataModule
- 添加k折分割功能
- 保持接口的简洁性
- 确保与现有训练流程的兼容性

设计理念：
就像给汽车安装GPS导航系统，我们不重新制造汽车，
而是在现有功能基础上添加新的导航能力。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import logging

from .multimodal_datamodule import MultiModalDataModule

logger = logging.getLogger(__name__)


class KFoldDataModuleWrapper:
    """
    K折数据模块包装器

    这个类采用"装饰器模式"，它包装现有的MultiModalDataModule，
    为其添加k折分割功能。就像给手机贴膜一样，
    不改变手机本身，但增加了新的保护功能。

    核心思想：
    1. 组合而非继承：包装现有的数据模块
    2. 单一职责：只负责k折分割逻辑
    3. 接口兼容：对外提供标准的数据模块接口
    """

    def __init__(
        self,
        base_datamodule_config: Dict[str, Any],
        n_splits: int = 5,
        stratified: bool = True,
        seed: int = 3407,
        output_dir: str = "outputs/kfold_info",
    ):
        """
        初始化K折数据包装器

        Args:
            base_datamodule_config: 基础数据模块配置
            n_splits: 折数
            stratified: 是否分层抽样
            seed: 随机种子
            output_dir: 输出目录
        """
        self.base_config = base_datamodule_config
        self.n_splits = n_splits
        self.stratified = stratified
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 存储折分割信息
        self.fold_splits = None
        self.current_fold = None

        logger.info(f"🎯 KFoldDataModuleWrapper initialized for {n_splits}-fold CV")

    def prepare_fold_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        准备k折分割

        这个方法的核心思想很简单：
        1. 创建一个临时的数据模块来获取完整数据集
        2. 获取所有样本的标签
        3. 使用sklearn进行k折分割
        4. 保存分割信息供后续使用

        就像切蛋糕一样，我们先看看蛋糕有多大（获取数据集），
        然后决定怎么切（k折分割），最后记录切法（保存分割）。
        """
        logger.info("📊 Preparing K-fold splits...")

        # 创建临时数据模块获取完整数据集
        temp_datamodule = self._create_base_datamodule()
        temp_datamodule.prepare_data()
        temp_datamodule.setup("fit")

        # 获取训练数据集
        train_dataset = temp_datamodule.train_dataset
        total_samples = len(train_dataset)

        logger.info(f"📈 Total training samples: {total_samples}")

        # 提取标签用于分层分割
        labels = []
        logger.info("🔍 Extracting labels for stratified splitting...")

        for i in range(total_samples):
            # 这里我们假设数据集返回 (image, label) 的格式
            try:
                _, label = train_dataset[i]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
            except Exception as e:
                logger.error(f"Error extracting label for sample {i}: {e}")
                # 如果提取标签失败，我们假设为负样本
                labels.append(0)

        labels = np.array(labels)

        # 打印数据分布信息
        pos_count = np.sum(labels == 1)
        neg_count = np.sum(labels == 0)
        logger.info(f"📊 Label distribution - Positive: {pos_count}, Negative: {neg_count}")
        logger.info(f"📊 Positive ratio: {pos_count/total_samples:.3f}")

        # 执行k折分割
        if self.stratified and len(np.unique(labels)) > 1:
            logger.info("🎯 Using stratified K-fold split")
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(kfold.split(np.arange(total_samples), labels))
        else:
            logger.info("🎯 Using regular K-fold split")
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(kfold.split(np.arange(total_samples)))

        # 验证分割质量
        self._validate_splits(splits, labels)

        # 保存分割信息
        self._save_fold_splits(splits, labels)

        self.fold_splits = splits
        logger.info(f"✅ K-fold splits prepared successfully!")

        return splits

    def get_fold_datamodule(self, fold_idx: int) -> MultiModalDataModule:
        """
        获取指定折的数据模块

        这是整个设计的核心方法。它的思路很简单：
        1. 获取当前折的训练和验证索引
        2. 创建一个标准的数据模块
        3. 设置数据后，用索引创建子集
        4. 返回配置好的数据模块

        就像从图书馆的完整书籍中，根据不同的借阅卡
        为不同的读者准备不同的书单。

        Args:
            fold_idx: 折索引 (0-based)

        Returns:
            配置好的数据模块
        """
        if self.fold_splits is None:
            raise ValueError("Must call prepare_fold_splits() first")

        if fold_idx >= self.n_splits:
            raise ValueError(f"fold_idx {fold_idx} >= n_splits {self.n_splits}")

        logger.info(f"🔄 Preparing data module for fold {fold_idx + 1}/{self.n_splits}")

        # 获取当前折的索引
        train_indices, val_indices = self.fold_splits[fold_idx]

        logger.info(f"📊 Fold {fold_idx}: Train={len(train_indices)}, Val={len(val_indices)}")

        # 创建基础数据模块
        datamodule = self._create_base_datamodule()
        datamodule.prepare_data()
        datamodule.setup("fit")

        # 使用索引创建子集
        # 这里是关键：我们不重新加载数据，而是创建现有数据集的子集
        original_train_dataset = datamodule.train_dataset
        original_val_dataset = datamodule.val_dataset  # 这个在k折中不会使用

        # 创建当前折的训练和验证子集
        fold_train_dataset = Subset(original_train_dataset, train_indices)
        fold_val_dataset = Subset(original_train_dataset, val_indices)

        # 替换数据模块中的数据集
        datamodule.train_dataset = fold_train_dataset
        datamodule.val_dataset = fold_val_dataset

        # 记录当前折信息
        self.current_fold = fold_idx

        logger.info(f"✅ Fold {fold_idx} data module ready!")

        return datamodule

    def _create_base_datamodule(self) -> MultiModalDataModule:
        """创建基础数据模块"""
        from ..utils.instantiate import instantiate_from_config

        # 这里我们使用配置来创建数据模块
        # 这确保了与标准训练流程的完全一致性
        config = {"target": "lightning_landslide.src.data.MultiModalDataModule", "params": self.base_config}

        return instantiate_from_config(config)

    def _validate_splits(self, splits: List[Tuple[np.ndarray, np.ndarray]], labels: np.ndarray):
        """验证分割质量"""
        logger.info("🔍 Validating split quality...")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]

            train_pos_ratio = np.mean(train_labels) if len(train_labels) > 0 else 0
            val_pos_ratio = np.mean(val_labels) if len(val_labels) > 0 else 0

            logger.info(
                f"Fold {fold_idx}: Train pos ratio: {train_pos_ratio:.3f}, " f"Val pos ratio: {val_pos_ratio:.3f}"
            )

            # 警告：如果分割质量很差
            if abs(train_pos_ratio - val_pos_ratio) > 0.1:
                logger.warning(f"⚠️  Fold {fold_idx} has imbalanced splits! " f"Consider using stratified=True")

    def _save_fold_splits(self, splits: List[Tuple[np.ndarray, np.ndarray]], labels: np.ndarray):
        """保存分割信息"""
        splits_info = {
            "n_splits": self.n_splits,
            "total_samples": len(labels),
            "stratified": self.stratified,
            "seed": self.seed,
            "positive_samples": int(np.sum(labels == 1)),
            "negative_samples": int(np.sum(labels == 0)),
            "folds": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_info = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_pos_ratio": float(np.mean(labels[train_idx])),
                "val_pos_ratio": float(np.mean(labels[val_idx])),
                # 为了节省空间，我们不保存完整的索引，只保存统计信息
                # 如果需要完整索引，可以取消下面两行的注释
                # "train_indices": train_idx.tolist(),
                # "val_indices": val_idx.tolist()
            }
            splits_info["folds"].append(fold_info)

        # 保存分割信息
        splits_file = self.output_dir / f"kfold_{self.n_splits}splits_seed{self.seed}.json"
        with open(splits_file, "w") as f:
            json.dump(splits_info, f, indent=2)

        logger.info(f"📁 Fold splits info saved to: {splits_file}")


def create_kfold_wrapper(
    base_datamodule_config: Dict[str, Any],
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 3407,
    output_dir: str = "outputs/kfold_info",
) -> KFoldDataModuleWrapper:
    """
    创建K折数据包装器的便利函数

    这个函数就像一个工厂，根据配置生产出合适的k折数据包装器。
    使用这个函数可以让代码更加简洁和易读。
    """
    return KFoldDataModuleWrapper(
        base_datamodule_config=base_datamodule_config,
        n_splits=n_splits,
        stratified=stratified,
        seed=seed,
        output_dir=output_dir,
    )
