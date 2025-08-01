import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Tuple, Optional, Callable, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    简化的多模态遥感数据集类

    核心职责（简化后）：
    1. 数据加载：从.npy文件加载多通道数据
    2. 通道处理：提取并组合光学通道
    3. NDVI计算：实时计算归一化植被指数
    4. 标签处理：正确处理二分类标签
    5. 数据变换：应用预处理和增强

    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_file: Union[str, Path],
        transform: Optional[Callable] = None,
        compute_ndvi: bool = True,
        cache_data: bool = True,
        channel_config: Optional[Dict] = None,
        usage_mode: str = "optical_only",
        # 🔧 跨目录映射支持（保留高级功能）
        cross_directory_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        初始化光学数据集

        Args:
            data_dir: 数据文件目录路径
            csv_file: 标签文件路径（已清洁的CSV文件）
            transform: 数据变换函数
            compute_ndvi: 是否计算NDVI通道
            cache_data: 是否缓存数据到内存（小数据集时有用）
            channel_config: 输入数据通道配置
            usage_mode: 使用模式
            cross_directory_mapping: 跨目录数据路径映射字典 {sample_id: full_path}
        """
        logger.info("MultiModalDataset_init" + "-" * 100)
        # 路径处理
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file)

        # 数据处理配置
        self.transform = transform
        self.compute_ndvi = compute_ndvi
        self.cache_data = cache_data

        # 通道配置
        self.channel_config = channel_config
        self.usage_mode = usage_mode
        self.active_channels = self._parse_active_channels()

        # 🔧 跨目录映射支持（保留高级功能）
        self.cross_directory_mapping = cross_directory_mapping or {}

        # 计算最终通道数
        self.num_channels = len(self.active_channels)

        # 数据缓存（如果启用）
        self.data_cache = {} if cache_data else None

        # 🎯 简化的数据加载流程
        self.data_index = self._load_data_index()

        # 检查是否有标签列
        self.has_labels = "label" in self.data_index.columns

        # 日志信息
        logger.info(f"📊 Loaded {len(self.data_index)} samples from cleaned CSV")
        logger.info(f"🔢 Active channels: {self.active_channels}")
        logger.info(f"🔢 Final channel count: {self.num_channels}")
        logger.info(f"📋 Has labels: {self.has_labels}")

        # 🔧 跨目录映射信息
        if self.cross_directory_mapping:
            logger.info(f"📁 Cross-directory mapping: {len(self.cross_directory_mapping)} samples")

        logger.info("🔢✅ MultiModalDataset initialization completed!")

    def _parse_active_channels(self) -> List[int]:
        """解析当前使用模式下的活跃通道"""
        mode_config = self.channel_config["usage_modes"][self.usage_mode]
        active_groups = mode_config["groups"]

        active_channels = []

        for group_name in active_groups:
            group_channels = self.channel_config["channel_groups"][group_name]
            active_channels.extend(group_channels)

        return active_channels

    def _load_data_index(self) -> pd.DataFrame:
        """
        加载数据索引文件

        这个方法读取CSV文件，建立样本ID到标签的映射。
        Returns:
            包含ID和标签的DataFrame
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        df = pd.read_csv(self.csv_file)

        # 基本数据验证
        if "ID" not in df.columns:
            raise ValueError("CSV file must contain 'ID' column")

        # 检查标签列（训练集有，测试集可能没有）
        has_labels = "label" in df.columns

        logger.info(f"🔢 Loaded {len(df)} samples from CSV")
        return df.reset_index(drop=True)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本

        这是数据集类的核心方法。它的职责是：
        1. 加载原始数据
        2. 提取和处理指定通道
        3. 计算NDVI（如果需要）
        4. 应用数据变换
        5. 返回tensor格式的数据和标签

        Args:
            idx: 样本索引

        Returns:
            (data, label) 元组，其中：
            - data: 形状为 (channels, height, width) 的张量
            - label: 标签张量
        """

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

            return data, label

        except Exception as e:
            logger.error(f"Error loading sample {sample_id}: {e}")
            raise e

    def _load_sample_data(self, sample_id: str) -> torch.Tensor:
        """
        加载并处理单个样本多模态的数据 - 支持跨目录访问

        这个方法实现了您原有数据加载逻辑的核心部分：
        1. 检查跨目录映射，优先使用映射路径
        2. 加载多通道.npy文件
        3. 提取指定的光学通道
        4. 计算NDVI通道
        5. 组合成最终的多通道数据

        Args:
            sample_id: 样本ID

        Returns:
            处理后的数据张量，形状为 (channels, height, width)
        """

        # 🔧 核心修改：优先使用跨目录映射路径
        if sample_id in self.cross_directory_mapping:
            data_path = Path(self.cross_directory_mapping[sample_id])
            logger.debug(f"🔗 Using cross-directory path for {sample_id}: {data_path}")
        else:
            # 默认路径：在数据目录中查找
            data_path = self.data_dir / f"{sample_id}.npy"
            logger.debug(f"📁 Using default path for {sample_id}: {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # 加载原始数据
        raw_data = np.load(data_path)  # 形状通常是 (12, 64, 64)

        # 确保数据形状为 (channels, height, width)
        if raw_data.shape[-1] == 12:  # (64, 64, 12) → (12, 64, 64)
            raw_data = np.transpose(raw_data, (2, 0, 1))

        # 根据配置选择通道
        selected_channels = []

        for channel in self.active_channels:
            if channel == "ndvi" and self.compute_ndvi:
                # 计算NDVI
                optical_channels = self.channel_config["channel_groups"]["optical"]
                red_idx, nir_idx = optical_channels[0], optical_channels[3]
                ndvi = self._compute_ndvi(raw_data[red_idx], raw_data[nir_idx])
                selected_channels.append(ndvi)
            else:
                selected_channels.append(raw_data[channel])

        # 堆叠所有通道
        final_data = np.stack(selected_channels, axis=0)  # (channels, height, width)

        # 转换为PyTorch张量
        data_tensor = torch.from_numpy(final_data).float()

        return data_tensor

    def _compute_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        计算归一化植被指数 (NDVI)

        NDVI是遥感中最重要的植被指数之一，计算公式为：
        NDVI = (NIR - Red) / (NIR + Red)

        NDVI在滑坡检测中的重要性：
        1. 滑坡区域通常植被覆盖较少
        2. NDVI能够突出植被与裸土的差异
        3. 时间序列NDVI变化能指示地表扰动

        Args:
            red: 红光通道
            nir: 近红外通道

        Returns:
            NDVI数组，形状与输入通道相同
        """
        # 转换数据类型
        red = red.astype(np.float32)
        nir = nir.astype(np.float32)

        # 计算NDVI，添加小常数避免除零
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)

        # 将NDVI限制在合理范围内 [-1, 1]
        ndvi = np.clip(ndvi, -1.0, 1.0)

        return ndvi


# 便捷函数：创建不同配置的数据集
def create_train_dataset(
    data_dir: str,
    csv_file: str,
    transform: Callable = None,
    channel_config: Optional[Dict] = None,
    usage_mode: str = "full_multimodal",
    cross_directory_mapping: Optional[Dict[str, str]] = None,
) -> MultiModalDataset:
    """
    创建训练数据集（简化版）

    简化说明：
    - 移除了 exclude_ids_file 参数
    - 假设 csv_file 已经是清洁的数据

    Args:
        data_dir: 数据目录
        csv_file: 清洁的CSV文件路径
        transform: 数据变换函数
        channel_config: 通道配置
        usage_mode: 使用模式
        cross_directory_mapping: 跨目录映射

    Returns:
        MultiModalDataset: 训练数据集实例
    """
    # 验证usage_mode的有效性
    valid_modes = channel_config.get("usage_modes", {}).keys()
    if usage_mode not in valid_modes:
        raise ValueError(f"Invalid usage_mode: {usage_mode}. Valid options: {list(valid_modes)}")

    """创建训练数据集"""
    return MultiModalDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=transform,
        compute_ndvi=True,
        cache_data=True,
        channel_config=channel_config,
        usage_mode=usage_mode,
        cross_directory_mapping=cross_directory_mapping,
    )


def create_test_dataset(
    data_dir: str,
    csv_file: str,
    transform: Callable = None,
    channel_config: Optional[Dict] = None,
    usage_mode: str = "optical_only",
) -> MultiModalDataset:
    """
    创建测试数据集（简化版）

    Args:
        data_dir: 数据目录
        csv_file: CSV文件路径
        transform: 数据变换函数
        channel_config: 通道配置
        usage_mode: 使用模式

    Returns:
        MultiModalDataset: 测试数据集实例
    """
    return MultiModalDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=transform,
        compute_ndvi=True,
        cache_data=True,
        channel_config=channel_config,
        usage_mode=usage_mode,
    )
