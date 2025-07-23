import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Tuple, Optional, Callable, List, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    光学遥感数据集类

    这个类专门处理您项目中的多通道遥感数据。它的核心职责是：

    1. 数据加载：从.npy文件加载多通道数据
    2. 通道处理：提取并组合光学通道
    3. NDVI计算：实时计算归一化植被指数
    4. 质量控制：过滤低质量样本
    5. 标签处理：正确处理二分类标签

    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_file: Union[str, Path],
        exclude_ids_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # channels: List[int] = [0, 1, 2, 3],  # R, G, B, NIR的索引
        compute_ndvi: bool = True,
        cache_data: bool = True,
        validate_data: bool = True,
        channel_config: Optional[Dict] = None,
        usage_mode: str = "optical_only",
    ):
        """
        初始化光学数据集

        这个初始化方法设计得非常灵活，既支持您当前的数据格式，
        也为未来的扩展留下了空间。

        Args:
            data_dir: 数据文件目录路径
            csv_file: 标签文件路径
            exclude_ids_file: 需要排除的样本ID文件（JSON格式）
            transform: 数据变换函数
            target_transform: 标签变换函数
            channels: 光学通道在原始数据中的索引位置
            compute_ndvi: 是否计算NDVI通道
            cache_data: 是否缓存数据到内存（小数据集时有用）
            validate_data: 是否验证数据完整性
            channel_config: ；输入数据通道配置
        """
        # 路径处理
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file)

        # 验证路径存在性
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        # 数据处理配置
        self.transform = transform
        self.target_transform = target_transform
        # self.channels = channels
        self.compute_ndvi = compute_ndvi
        self.cache_data = cache_data

        self.channel_config = channel_config
        self.usage_mode = usage_mode
        self.active_channels = self._parse_active_channels()

        # self.channel_groups = self.channel_config["channel_groups"]
        # self.total_channels = self.channel_config["total_channels"]

        # 计算最终通道数
        self.num_channels = len(self.active_channels)

        # 数据缓存（如果启用）
        self.data_cache = {} if cache_data else None

        # 加载样本索引和标签
        self.data_index = self._load_data_index()

        # 加载排除列表
        self.exclude_ids = self._load_exclude_ids(exclude_ids_file)

        # 过滤数据
        self._filter_data()

        # 数据验证
        if validate_data:
            self._validate_data_samples()

        # 数据统计
        self._compute_data_stats()

        logger.info(f"MultiModalDataset initialized with {len(self)} samples")
        logger.info(f"Channels: {self.channels}, NDVI: {self.compute_ndvi}")
        logger.info(f"Final channel count: {self.num_channels}")

    def _load_data_index(self) -> pd.DataFrame:
        """
        加载数据索引文件

        这个方法读取CSV文件，建立样本ID到标签的映射。
        我们进行了一些数据质量检查，确保数据格式正确。

        Returns:
            包含ID和标签的DataFrame
        """
        logger.info(f"Loading data index from {self.csv_file}")

        try:
            df = pd.read_csv(self.csv_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")

        # 验证必需的列存在
        required_columns = ["ID"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # 检查是否有标签列（训练集有，测试集可能没有）
        self.has_labels = "label" in df.columns

        if self.has_labels:
            # 验证标签值
            unique_labels = df["label"].unique()
            logger.info(f"Found labels: {sorted(unique_labels)}")

            # 检查是否为二分类问题
            if not set(unique_labels).issubset({0, 1}):
                logger.warning(f"Unexpected label values: {unique_labels}")
        else:
            logger.info("No label column found - assuming test dataset")

        logger.info(f"Loaded {len(df)} samples from CSV")
        return df.reset_index(drop=True)

    def _load_exclude_ids(self, exclude_ids_file: Optional[Union[str, Path]]) -> set:
        """
        加载需要排除的样本ID列表

        在数据质量分析阶段，您可能已经识别出了一些低质量的样本。
        这个方法加载这些样本的ID，确保它们不会用于训练。

        Args:
            exclude_ids_file: 排除列表文件路径

        Returns:
            需要排除的样本ID集合
        """
        if exclude_ids_file is None:
            return set()

        exclude_path = Path(exclude_ids_file)
        if not exclude_path.exists():
            logger.warning(f"Exclude IDs file not found: {exclude_path}")
            return set()

        try:
            with open(exclude_path, "r") as f:
                exclude_ids = json.load(f)

            if isinstance(exclude_ids, list):
                exclude_set = set(exclude_ids)
            elif isinstance(exclude_ids, dict):
                # 支持更复杂的排除规则
                exclude_set = set(exclude_ids.get("exclude", []))
            else:
                raise ValueError("Exclude IDs should be a list or dict")

            logger.info(f"Loaded {len(exclude_set)} samples to exclude")
            return exclude_set

        except Exception as e:
            logger.error(f"Failed to load exclude IDs: {e}")
            return set()

    def _filter_data(self) -> None:
        """
        过滤数据，移除排除列表中的样本

        这个步骤确保我们只处理高质量的数据样本。
        """
        if not self.exclude_ids:
            return

        initial_count = len(self.data_index)

        # 过滤排除样本
        mask = ~self.data_index["ID"].isin(self.exclude_ids)
        # .reset_index(drop=True)：重置行索引
        self.data_index = self.data_index[mask].reset_index(drop=True)

        filtered_count = initial_count - len(self.data_index)
        logger.info(f"Filtered out {filtered_count} low-quality samples")
        logger.info(f"Remaining samples: {len(self.data_index)}")

    def _parse_active_channels(self) -> Dict[str, List[int]]:
        """解析当前使用模式下的活跃通道"""
        mode_config = self.channel_config["usage_modes"][self.usage_mode]
        active_groups = mode_config["groups"]

        channels_map = {}

        for group_name in active_groups:
            if group_name == "derived":
                # 处理计算得出的通道（如NDVI）
                channels_map[group_name] = ["ndvi"]  # 特殊标记
            else:
                group_channels = self.channel_config["channel_groups"][group_name]
                channels_map[group_name] = group_channels

        return channels_map

    def _load_sample_data(self, sample_id: str) -> torch.Tensor:
        """
        加载并处理单个样本多模态的数据

        这个方法实现了您原有数据加载逻辑的核心部分：
        1. 加载多通道.npy文件
        2. 提取指定的光学通道
        3. 计算NDVI通道
        4. 组合成最终的多通道数据

        Args:
            sample_id: 样本ID

        Returns:
            处理后的数据张量，形状为 (channels, height, width)
        """

        # 构造数据文件路径
        data_path = self.data_dir / f"{sample_id}.npy"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # 加载原始数据
        raw_data = np.load(data_path)  # 形状通常是 (12, 64, 64)

        # 根据配置选择通道
        selected_channels = []

        for group_name, channels in self.active_channels.items():
            if group_name == "derived" and "ndvi" in channels:
                # 计算NDVI
                optical_channels = self.channel_config["channel_groups"]["optical"]
                red_idx, nir_idx = optical_channels[0], optical_channels[3]
                ndvi = self._compute_ndvi(raw_data[red_idx], raw_data[nir_idx])
                selected_channels.append(ndvi)
            else:
                # 选择原始通道
                for ch_idx in channels:
                    selected_channels.append(raw_data[ch_idx])

        # 堆叠所有通道
        final_data = np.stack(selected_channels, axis=0)  # (channels, height, width)

        # 转换为PyTorch张量
        data_tensor = torch.from_numpy(final_data).float()

        return data_tensor

    def _validate_data_samples(self, max_check: int = 100) -> None:
        """
        验证数据样本的完整性

        这个方法随机检查一些数据文件，确保：
        1. 文件存在且可读
        2. 数据形状正确
        3. 数据值在合理范围内

        Args:
            max_check: 最大检查样本数
        """
        logger.info("Validating data samples...")

        # 随机选择一些样本进行检查
        check_indices = np.random.choice(len(self.data_index), size=min(max_check, len(self.data_index)), replace=False)

        missing_files = []
        invalid_files = []

        for idx in check_indices:
            sample_id = self.data_index.iloc[idx]["ID"]
            data_path = self.data_dir / f"{sample_id}.npy"

            if not data_path.exists():
                missing_files.append(sample_id)
                continue

            try:
                data = np.load(data_path)

                # 检查数据形状
                if len(data.shape) != 3:
                    invalid_files.append(f"{sample_id}: wrong shape {data.shape}")
                    continue

                # 检查通道数
                if data.shape[0] < max(self.channels) + 1:
                    invalid_files.append(f"{sample_id}: insufficient channels {data.shape[0]}")
                    continue

                # 检查数据值范围（基本合理性检查）
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    invalid_files.append(f"{sample_id}: contains NaN or Inf values")
                    continue

            except Exception as e:
                invalid_files.append(f"{sample_id}: load error - {str(e)}")

        # 报告验证结果
        if missing_files:
            logger.error(f"Missing data files: {missing_files[:10]}...")  # 只显示前10个
        if invalid_files:
            logger.error(f"Invalid data files: {invalid_files[:10]}...")

        if missing_files or invalid_files:
            raise RuntimeError(f"Data validation failed: {len(missing_files)} missing, {len(invalid_files)} invalid")

        logger.info(f"✓ Data validation passed ({len(check_indices)} samples checked)")

    def _compute_data_stats(self) -> None:
        """
        计算数据统计信息

        这些统计信息对于数据分析和模型调试很有用。
        """
        self.stats = {
            "total_samples": len(self.data_index),
            "num_channels": self.num_channels,
            "channels": self.channels,
            "compute_ndvi": self.compute_ndvi,
        }

        if self.has_labels:
            label_counts = self.data_index["label"].value_counts().to_dict()
            self.stats.update(
                {
                    "label_distribution": label_counts,
                    "class_balance": min(label_counts.values()) / max(label_counts.values()) if label_counts else 0,
                }
            )

        logger.info(f"Dataset statistics: {self.stats}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本

        这是数据集类的核心方法。它的职责是：
        1. 加载原始数据
        2. 提取和处理光学通道
        3. 计算NDVI（如果需要）
        4. 应用数据变换
        5. 返回标准格式的数据

        Args:
            idx: 样本索引

        Returns:
            (data, label) 元组，其中：
            - data: 形状为 (channels, height, width) 的张量
            - label: 标签张量
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self)}")

        # 获取样本信息
        """
        iloc 是 pandas 中用于 按位置（position）索引 行的方法。
        iloc[idx] 返回的是第 idx 行的数据（按数字位置，不是按 ID）。
        结果是一个 pandas.Series，包含该行的所有列内容。
        """
        row = self.data_index.iloc[idx]
        sample_id = row["ID"]

        try:
            # 从缓存或文件加载数据
            if self.data_cache is not None and sample_id in self.data_cache:
                data = self.data_cache[sample_id]
            else:
                data = self._load_sample_data(sample_id)
                if self.data_cache is not None:
                    self.data_cache[sample_id] = data

            # 处理标签
            if self.has_labels:
                label = torch.tensor(row["label"], dtype=torch.long)
            else:
                label = torch.tensor(-1, dtype=torch.long)  # 测试集的占位符标签

            # 应用变换
            if self.transform is not None:
                data = self.transform(data)

            if self.target_transform is not None:
                label = self.target_transform(label)

            return data, label

        except Exception as e:
            logger.error(f"Error loading sample {sample_id}: {e}")
            # 返回一个全零张量作为fallback，避免训练中断
            fallback_data = torch.zeros(self.num_channels, 64, 64)
            fallback_label = torch.tensor(0, dtype=torch.long)
            return fallback_data, fallback_label

    def _compute_ndvi(self, optical_channels: List[np.ndarray]) -> np.ndarray:
        """
        计算归一化植被指数 (NDVI)

        NDVI是遥感中最重要的植被指数之一，计算公式为：
        NDVI = (NIR - Red) / (NIR + Red)

        在滑坡检测中，NDVI特别有用，因为：
        1. 滑坡区域通常植被覆盖较少
        2. NDVI能够突出植被与裸土的差异
        3. 时间序列NDVI变化能指示地表扰动

        Args:
            optical_channels: 光学通道列表，顺序为 [R, G, B, NIR]

        Returns:
            NDVI数组，形状与输入通道相同
        """
        if len(optical_channels) < 4:
            raise ValueError("Need at least 4 channels to compute NDVI (R, G, B, NIR)")

        # 提取红光和近红外通道
        red = optical_channels[0].astype(np.float32)  # Red通道
        nir = optical_channels[3].astype(np.float32)  # NIR通道

        # 计算NDVI，添加小常数避免除零
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)

        # 将NDVI限制在合理范围内 [-1, 1]
        ndvi = np.clip(ndvi, -1.0, 1.0)

        return ndvi

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        获取样本的详细信息（调试用）

        Args:
            idx: 样本索引

        Returns:
            包含样本详细信息的字典
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        row = self.data_index.iloc[idx]
        sample_id = row["ID"]

        # 加载数据以获取统计信息
        try:
            data, label = self[idx]

            info = {
                "index": idx,
                "sample_id": sample_id,
                "data_shape": tuple(data.shape),
                "label": label.item() if self.has_labels else None,
                "data_min": data.min().item(),
                "data_max": data.max().item(),
                "data_mean": data.mean().item(),
                "data_std": data.std().item(),
            }

            # 各通道统计
            channel_stats = {}
            channel_names = [f"ch_{i}" for i in self.channels]
            if self.compute_ndvi:
                channel_names.append("ndvi")

            for i, name in enumerate(channel_names):
                channel_data = data[i]
                channel_stats[name] = {
                    "min": channel_data.min().item(),
                    "max": channel_data.max().item(),
                    "mean": channel_data.mean().item(),
                    "std": channel_data.std().item(),
                }

            info["channel_stats"] = channel_stats

        except Exception as e:
            info = {"index": idx, "sample_id": sample_id, "error": str(e)}

        return info

    def get_class_distribution(self) -> Dict[int, int]:
        """
        获取类别分布

        Returns:
            类别分布字典 {class_id: count}
        """
        if not self.has_labels:
            return {}

        return self.data_index["label"].value_counts().to_dict()

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取完整的数据统计信息

        Returns:
            数据统计字典
        """
        return self.stats.copy()


# 便捷函数：创建不同配置的数据集
def create_train_dataset(
    data_dir: str,
    csv_file: str,
    exclude_ids_file: str = None,
    transform: Callable = None,
    channel_config: Optional[Dict] = None,
    usage_mode: str = "optical_only",
) -> MultiModalDataset:

    # 验证usage_mode的有效性
    valid_modes = channel_config.get("usage_modes", {}).keys()
    if usage_mode not in valid_modes:
        raise ValueError(f"Invalid usage_mode: {usage_mode}. Valid options: {list(valid_modes)}")

    """创建训练数据集"""
    return MultiModalDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        exclude_ids_file=exclude_ids_file,
        transform=transform,
        channel_config=channel_config,
        usage_mode=usage_mode,
        compute_ndvi=True,
        validate_data=True,
        cache_data=True,
    )


def create_test_dataset(
    data_dir: str,
    csv_file: str,
    transform: Callable = None,
    channel_config: Optional[Dict] = None,
    usage_mode: str = "optical_only",
) -> MultiModalDataset:
    """创建测试数据集"""
    return MultiModalDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        exclude_ids_file=None,
        transform=transform,
        channel_config=channel_config,
        usage_mode=usage_mode,
        compute_ndvi=True,
        validate_data=False,  # 测试集可能没有所有样本
        cache_data=True,
    )
