import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional, Callable, Dict, Any
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class MultiSpectralNormalize:
    """
    多光谱数据标准化

    与普通的图像标准化不同，多光谱数据的每个通道都有不同的
    物理意义和数值分布。这个类为每个通道提供独立的标准化参数。
    其中，NDVI 和 SAR 插值通道的取值范围为 [-1, 1]，需要单独处理。？？
    """

    def __init__(self, means: List[float], stds: List[float]):
        """
        Args:
            means: 每个通道的均值列表
            stds: 每个通道的标准差列表
        """
        if len(means) != len(stds):
            raise ValueError("means and stds must have the same length")

        self.means = torch.tensor(means).view(-1, 1, 1)
        self.stds = torch.tensor(stds).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: 输入张量，形状为 (C, H, W)
        Returns:
            标准化后的张量
        """
        if tensor.size(0) != len(self.means):
            raise ValueError(f"Expected {len(self.means)} channels, got {tensor.size(0)}")

        return (tensor - self.means) / self.stds

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """反标准化，用于可视化"""
        return tensor * self.stds + self.means


class RemoteSensingRandomFlip:
    """
    遥感数据的随机翻转

    对于遥感数据，翻转通常是安全的，因为地理现象在不同方向上
    都可能出现。但要注意某些有方向性的特征（如河流、道路）。
    """

    def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.5):
        self.h_prob = horizontal_prob
        self.v_prob = vertical_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: 输入张量，形状为 (C, H, W)
        """
        # 水平翻转
        if random.random() < self.h_prob:
            tensor = torch.flip(tensor, dims=[2])

        # 垂直翻转
        if random.random() < self.v_prob:
            tensor = torch.flip(tensor, dims=[1])

        return tensor


class RemoteSensingRandomRotation:
    """
    遥感数据的随机旋转

    只支持90度的倍数旋转，避免插值带来的信息损失。
    对于遥感数据，保持像素值的准确性比平滑的旋转更重要。
    """

    def __init__(self, angles: List[int] = [0, 90, 180, 270], prob: float = 0.5):
        self.angles = angles
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: 输入张量，形状为 (C, H, W)
        """
        if random.random() < self.prob:
            angle = random.choice(self.angles)
            k = angle // 90  # 90度旋转的次数
            if k > 0:
                tensor = torch.rot90(tensor, k, dims=[1, 2])

        return tensor


class SpectralNoiseAugmentation:
    """
    光谱噪声增强

    为不同通道添加不同强度的噪声，模拟传感器噪声和大气干扰。
    这种增强有助于提高模型对真实世界数据变化的鲁棒性。
    """

    def __init__(self, noise_std: float = 0.01, channel_noise_weights: Optional[List[float]] = None, prob: float = 0.5):
        """
        Args:
            noise_std: 噪声标准差
            channel_noise_weights: 每个通道的噪声权重（如果为None，所有通道相同）
            prob: 应用噪声的概率
        """
        self.noise_std = noise_std
        self.channel_weights = channel_noise_weights
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            noise = torch.randn_like(tensor) * self.noise_std

            if self.channel_weights is not None:
                weights = torch.tensor(self.channel_weights).view(-1, 1, 1)
                if weights.size(0) != tensor.size(0):
                    logger.warning("Channel weights size mismatch, using uniform noise")
                else:
                    noise = noise * weights

            tensor = tensor + noise

        return tensor


class NDVIPreservingAugmentation:
    """
    保持NDVI物理意义的增强

    在对RGB和NIR通道进行增强时，确保NDVI通道（如果存在）
    保持其物理意义。这对于依赖植被指数的分析任务很重要。
    """

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.8, 1.2),
        ndvi_channel_idx: int = 4,  # NDVI通道通常是第5个（索引4）
        red_channel_idx: int = 0,
        nir_channel_idx: int = 3,
        prob: float = 0.4,
    ):
        self.intensity_range = intensity_range
        self.ndvi_idx = ndvi_channel_idx
        self.red_idx = red_channel_idx
        self.nir_idx = nir_channel_idx
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            random.random() < self.prob
            and self.ndvi_idx < tensor.size(0)
            and self.red_idx < tensor.size(0)
            and self.nir_idx < tensor.size(0)
        ):

            # 对Red和NIR通道应用相同的强度变化
            intensity_factor = random.uniform(*self.intensity_range)
            tensor[self.red_idx] = tensor[self.red_idx] * intensity_factor
            tensor[self.nir_idx] = tensor[self.nir_idx] * intensity_factor

            # 重新计算NDVI以保持物理意义
            red = tensor[self.red_idx]
            nir = tensor[self.nir_idx]
            epsilon = 1e-8
            new_ndvi = (nir - red) / (nir + red + epsilon)
            new_ndvi = torch.clip(new_ndvi, -1.0, 1.0)
            tensor[self.ndvi_idx] = new_ndvi

        return tensor


class RemoteSensingCompose:
    """
    组合多个变换的容器

    类似于torchvision.transforms.Compose，但专门为遥感数据设计。
    提供了更好的错误处理和调试信息。
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            try:
                tensor = transform(tensor)
            except Exception as e:
                logger.error(f"Error in transform {i} ({transform.__class__.__name__}): {e}")
                raise
        return tensor

    def __repr__(self):
        """__repr__的作用是返回一个对象的“官方”字符串表示"""
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


def get_train_transforms(cfg: DictConfig) -> Callable:
    """
    创建训练时的数据变换管道

    这个函数基于配置创建适合训练的数据变换序列。
    训练时的变换通常包括：
    1. 数据增强（翻转、旋转、噪声等）
    2. 标准化

    Args:
        cfg: 配置对象，包含变换参数

    Returns:
        组合的变换函数
    """
    transform_cfg = cfg.data.transforms.train

    transforms = []

    # 几何变换
    if transform_cfg.get("random_flip", True):
        transforms.append(
            RemoteSensingRandomFlip(
                horizontal_prob=transform_cfg.get("h_flip_prob", 0.5),
                vertical_prob=transform_cfg.get("v_flip_prob", 0.5),
            )
        )

    if transform_cfg.get("random_rotation", True):
        transforms.append(RemoteSensingRandomRotation(prob=transform_cfg.get("rotation_prob", 0.5)))

    # 光谱增强
    if transform_cfg.get("spectral_noise", True):
        transforms.append(
            SpectralNoiseAugmentation(
                noise_std=transform_cfg.get("noise_std", 0.01), prob=transform_cfg.get("noise_prob", 0.5)
            )
        )

    # # 大气效应模拟
    # if transform_cfg.get("atmospheric_scattering", False):
    #     transforms.append(
    #         AtmosphericScatteringSimulation(
    #             scattering_strength=transform_cfg.get("scattering_strength", 0.1),
    #             prob=transform_cfg.get("scattering_prob", 0.3),
    #         )
    #     )

    # NDVI保持增强
    if transform_cfg.get("ndvi_preserving", True):
        transforms.append(
            NDVIPreservingAugmentation(
                intensity_range=transform_cfg.get("intensity_range", [0.8, 1.2]),
                prob=transform_cfg.get("ndvi_aug_prob", 0.4),
            )
        )

    # 标准化（通常放在最后）
    if "normalization" in transform_cfg:
        norm_cfg = transform_cfg.normalization
        transforms.append(MultiSpectralNormalize(means=norm_cfg.means, stds=norm_cfg.stds))

    logger.info(f"Created training transforms with {len(transforms)} steps")
    return RemoteSensingCompose(transforms)


def get_val_transforms(cfg: DictConfig) -> Callable:
    """
    创建验证时的数据变换管道

    验证时通常只应用标准化，不使用数据增强，
    确保验证结果的一致性和可重现性。

    Args:
        cfg: 配置对象

    Returns:
        验证用的变换函数
    """
    transform_cfg = cfg.data.transforms

    transforms = []

    # 只添加标准化
    if "normalization" in transform_cfg:
        norm_cfg = transform_cfg.normalization
        transforms.append(MultiSpectralNormalize(means=norm_cfg.means, stds=norm_cfg.stds))

    logger.info(f"Created validation transforms with {len(transforms)} steps")
    return RemoteSensingCompose(transforms)


def get_test_transforms(cfg: DictConfig) -> Callable:
    """
    创建测试时的数据变换管道

    测试时的变换与验证时相同，只进行标准化。

    Args:
        cfg: 配置对象

    Returns:
        测试用的变换函数
    """
    return get_val_transforms(cfg)


# 预定义的标准化参数（基于常见的遥感数据统计）
STANDARD_OPTICAL_NORMALIZATION = {
    # 这些值应该根据您的具体数据统计来调整
    "means": [0.485, 0.456, 0.406, 0.5, 0.0],  # R, G, B, NIR, NDVI
    "stds": [0.229, 0.224, 0.225, 0.25, 0.5],  # 对应的标准差
}


def create_default_transforms(use_augmentation: bool = True) -> Tuple[Callable, Callable]:
    """
    创建默认的训练和验证变换

    这是一个便捷函数，为快速测试提供合理的默认变换。

    Args:
        use_augmentation: 是否在训练时使用数据增强

    Returns:
        (train_transform, val_transform) 元组
    """
    # 标准化变换
    normalize = MultiSpectralNormalize(
        means=STANDARD_OPTICAL_NORMALIZATION["means"], stds=STANDARD_OPTICAL_NORMALIZATION["stds"]
    )

    if use_augmentation:
        # 训练时使用增强
        train_transforms = [
            RemoteSensingRandomFlip(),
            RemoteSensingRandomRotation(prob=0.5),
            SpectralNoiseAugmentation(prob=0.3),
            NDVIPreservingAugmentation(prob=0.4),
            normalize,
        ]
        train_transform = RemoteSensingCompose(train_transforms)
    else:
        train_transform = normalize

    # 验证时只标准化
    val_transform = normalize

    return train_transform, val_transform


# 调试和可视化工具
def visualize_transform_effect(tensor: torch.Tensor, transform: Callable, num_samples: int = 5) -> List[torch.Tensor]:
    """
    可视化变换效果

    这个函数对同一个输入应用多次变换，显示变换的效果。
    对于调试和理解变换行为很有用。

    Args:
        tensor: 输入张量
        transform: 要测试的变换
        num_samples: 生成的样本数

    Returns:
        变换后的张量列表
    """
    results = []
    for _ in range(num_samples):
        transformed = transform(tensor.clone())
        results.append(transformed)

    return results


def compute_channel_statistics(dataset, num_samples: int = 1000) -> Dict[str, List[float]]:
    """
    计算数据集的通道统计信息

    这个函数用于计算标准化所需的均值和标准差。
    在设置数据预处理管道时，这些统计信息是必需的。

    Args:
        dataset: 数据集对象
        num_samples: 用于计算统计信息的样本数

    Returns:
        包含均值和标准差的字典
    """
    if len(dataset) == 0:
        return {"means": [], "stds": []}

    # 获取数据维度信息
    sample_data, _ = dataset[0]
    num_channels = sample_data.size(0)

    # 随机采样
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    # 收集所有像素值
    all_pixels = [[] for _ in range(num_channels)]

    for idx in indices:
        data, _ = dataset[idx]
        for c in range(num_channels):
            all_pixels[c].extend(data[c].flatten().tolist())

    # 计算统计信息
    means = []
    stds = []

    for c in range(num_channels):
        pixels = np.array(all_pixels[c])
        means.append(float(np.mean(pixels)))
        stds.append(float(np.std(pixels)))

    logger.info(f"Computed channel statistics from {len(indices)} samples:")
    logger.info(f"Means: {means}")
    logger.info(f"Stds: {stds}")

    return {"means": means, "stds": stds}
