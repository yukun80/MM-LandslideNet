import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig
from pathlib import Path

from .base import BaseModel

"""
python -m lightning_landslide.src.models.optical_swin
torch版本模型构建
"""

logger = logging.getLogger(__name__)  # 日志记录器, __name__ 是模块名


class OpticalSwinModel(BaseModel):
    """
    基于Swin Transformer的光学数据模型

    核心特性：
    1. 处理5通道输入：R, G, B, NIR, NDVI
    2. 动态上采样：从64x64自适应到224x224
    3. 特征分离：分离特征提取和分类决策
    """

    def __init__(
        self,
        model_name: str = "swinv2_small_patch4_window12_256",
        input_channels: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        pretrained_path: Optional[str] = None,  # 新增：本地权重路径
        img_size: int = 256,  # 新增：图像尺寸
    ):
        super().__init__()

        # 保存配置信息
        self.model_name = model_name
        self.input_channels = input_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.img_size = img_size  # 保存图像尺寸

        logger.info(f"Initializing OpticalSwinModel with {model_name}")
        logger.info(f"Target image size: {self.img_size}x{self.img_size}")

        # 如果提供了本地路径，则不使用timm的在线预训练
        use_timm_pretrained = pretrained and (pretrained_path is None)

        # 创建Swin Transformer骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_timm_pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # 如果提供了本地路径，加载本地权重
        if pretrained_path:
            self._load_local_weights(pretrained_path)

        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")

        self._modify_input_layer()

        self._model_info.update(
            {
                "backbone_name": model_name,
                "input_channels": input_channels,
                "pretrained": pretrained,
                "pretrained_path": pretrained_path,
                "dropout_rate": dropout_rate,
                "img_size": img_size,
            }
        )

        logger.info("OpticalSwinModel initialization completed successfully")

    def _load_local_weights(self, pretrained_path: str):
        """从本地文件加载预训练权重"""
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weights file not found at: {pretrained_path}")

        try:
            state_dict = torch.load(pretrained_path, map_location="cpu")

            # timm的权重通常在'model'键下
            if "model" in state_dict:
                state_dict = state_dict["model"]

            # 加载权重
            result = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Weight loading result: {result}")

            # 检查是否有未加载的键，这对于调试非常重要
            if result.missing_keys:
                logger.warning(f"Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys: {result.unexpected_keys}")

        except Exception as e:
            logger.error(f"Failed to load local weights from {pretrained_path}: {e}")
            raise

    def _modify_input_layer(self) -> None:
        """
        修改输入层以处理5通道输入

        这是您原有实现中最精彩的部分。我们不仅要支持5通道输入，
        还要智能地初始化权重，充分利用预训练的RGB权重。

        策略分解：
        1. 找到第一个卷积层
        2. 创建新的5通道卷积层
        3. 智能复制预训练权重：
           - 前3通道：直接复制RGB权重
           - 第4通道(NIR)：使用RGB权重的平均值
           - 第5通道(NDVI)：使用Red和NIR的组合
        4. 替换原有的卷积层

        为什么这样设计？
        这种初始化策略让模型在训练开始时就具有良好的特征提取能力，
        而不是从随机权重开始。这对于遥感数据特别有效。
        """
        logger.info("Modifying input layer for 5-channel input...")

        # 步骤1：找到第一个卷积层
        first_conv = None
        first_conv_name = ""

        # 遍历所有模块，找到第一个Conv2d层
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                logger.info(f"Found first conv layer: {name}, shape: {module.weight.shape}")
                break

        if first_conv is None:
            raise RuntimeError(f"Could not find first convolution layer in {self.model_name}")

        # 步骤2：创建新的5通道卷积层
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,  # 输入通道数
            out_channels=first_conv.out_channels,  # 输出通道数
            kernel_size=first_conv.kernel_size,  # 卷积核大小
            stride=first_conv.stride,  # 步长
            padding=first_conv.padding,  # 填充
            dilation=first_conv.dilation,  # 膨胀
            groups=first_conv.groups,  # 分组卷积
            bias=first_conv.bias is not None,  # 是否使用偏置
            padding_mode=first_conv.padding_mode,  # 填充模式
        )

        # 步骤3：智能权重初始化
        with torch.no_grad():
            old_weight = first_conv.weight
            # weight是卷积核的权重，形状: (out_channels, in_channels, kernel_h, kernel_w)

            try:
                assert old_weight.shape[1] == 3  # 确认原始权重是3通道
                # 前3通道：直接复制RGB预训练权重
                new_conv.weight[:, :3, :, :] = old_weight
                # 第4通道(NIR)：使用RGB权重的平均值作为初始化
                # 这基于假设：NIR与可见光有相似但不同的特征模式
                nir_init = old_weight.mean(dim=1, keepdim=True)  # 对RGB通道求平均
                new_conv.weight[:, 3:4, :, :] = nir_init
                logger.info("✓ Initialized NIR channel with RGB average")

                # 第5通道(NDVI)：使用Red+NIR的组合进行初始化
                # 这基于NDVI计算公式：(NIR-Red)/(NIR+Red)的特征模式
                if self.input_channels >= 5:
                    red_weight = old_weight[:, 0:1, :, :]  # Red通道权重
                    ndvi_init = (nir_init + red_weight) / 2  # Red和NIR的平均
                    new_conv.weight[:, 4:5, :, :] = ndvi_init
                    logger.info("✓ Initialized NDVI channel with Red+NIR combination")

                # 如果有更多通道，使用相同策略
                for i in range(5, self.input_channels):
                    new_conv.weight[:, i : i + 1, :, :] = nir_init
                    logger.info(f"✓ Initialized channel {i} with NIR pattern")
            except AssertionError:
                logger.warning(f"Unexpected pretrained weight channels: {old_weight.shape[1]}")
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            # 复制bias（如果存在）
            if new_conv.bias is not None and first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        # 步骤4：替换原有卷积层
        self._replace_layer(first_conv_name, new_conv)
        logger.info(f"✓ Successfully replaced input layer: {first_conv_name}")

    def _replace_layer(self, layer_name: str, new_layer: nn.Module) -> None:
        """
        替换指定名称的层

        这个方法通过层的全限定名称来定位和替换层。
        例如，如果层名是"patch_embed.proj"，则需要：
        1. 获取patch_embed模块
        2. 将其proj属性替换为新层

        Args:
            layer_name: 层的全限定名称，如"patch_embed.proj"
            new_layer: 新的层模块
        """
        # 将层名按点分割：["patch_embed", "proj"]
        names = layer_name.split(".")

        # 从backbone开始，逐层深入到目标层的父模块
        current_module = self.backbone
        for name in names[:-1]:
            current_module = getattr(current_module, name)

        # 替换最后一级的属性
        final_name = names[-1]
        setattr(current_module, final_name, new_layer)

        logger.info(f"Replaced layer: {layer_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        这个方法现在使用在初始化时定义的 self.img_size
        来进行动态上采样。
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channel, height, width), got {x.dim()}D")

        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")

        # 使用 self.img_size 进行动态上采样
        target_size = self.img_size
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

        features = self.backbone(x)
        return features

    def get_feature_dim(self) -> int:
        """
        返回特征维度

        这个信息用于构建合适的分类头。
        对于swin_tiny，通常是768维。

        Returns:
            特征维度
        """
        return self.feature_dim

    def get_features_with_intermediate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取中间层特征（用于可视化和分析）

        这个方法提供了更详细的特征信息，可用于：
        1. 特征可视化
        2. 模型解释
        3. 特征融合研究

        Args:
            x: 输入张量

        Returns:
            包含不同层级特征的字典
        """
        features = {}

        # 上采样
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # 获取patch embedding
        if hasattr(self.backbone, "patch_embed"):
            x = self.backbone.patch_embed(x)
            features["patch_embed"] = x

        # 通过各个stage（如果可以访问）
        if hasattr(self.backbone, "layers"):
            for i, layer in enumerate(self.backbone.layers):
                x = layer(x)
                features[f"stage_{i}"] = x

        # 最终特征
        if hasattr(self.backbone, "norm") and hasattr(self.backbone, "avgpool"):
            x = self.backbone.norm(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            features["final"] = x
        else:
            # 如果结构不同，使用完整的前向传播
            features["final"] = self.backbone(x)

        return features

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "OpticalSwinModel":
        """
        从配置创建模型实例

        这是工厂方法模式的实现。通过配置文件就能创建模型，
        而不需要在代码中硬编码模型参数。

        Args:
            cfg: 完整的配置对象

        Returns:
            配置好的OpticalSwinModel实例
        """
        model_cfg = cfg.model

        return cls(
            model_name=model_cfg.get("backbone_name", "swin_tiny_patch4_window7_224"),
            input_channels=model_cfg.get("input_channels", 5),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.2),
        )

    def freeze_patch_embed(self) -> None:
        """
        只冻结patch embedding层

        这是一个更精细的冻结策略，在某些场景下很有用：
        当我们想保持输入层的权重不变（特别是我们精心初始化的权重），
        但允许更深层的特征学习。
        """
        if hasattr(self.backbone, "patch_embed"):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            logger.info("Frozen patch embedding layer")
        else:
            logger.warning("patch_embed not found, cannot freeze")

    def unfreeze_patch_embed(self) -> None:
        """解冻patch embedding层"""
        if hasattr(self.backbone, "patch_embed"):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = True
            logger.info("Unfrozen patch embedding layer")

    def get_layer_wise_lr_groups(self, base_lr: float = 1e-4, decay_factor: float = 0.8):
        """
        为Swin Transformer创建层次化学习率

        Swin Transformer有明确的层次结构，我们可以为不同的stage
        设置不同的学习率。通常规律是：
        - patch_embed：最小学习率（最基础的特征）
        - 早期stage：较小学习率
        - 后期stage：较大学习率
        - 分类头：最大学习率

        Args:
            base_lr: 基础学习率
            decay_factor: 每向前一stage，学习率的衰减因子

        Returns:
            参数组列表，每组有不同的学习率
        """
        param_groups = []

        # Patch embedding - 最小学习率
        if hasattr(self.backbone, "patch_embed"):
            param_groups.append(
                {
                    "params": list(self.backbone.patch_embed.parameters()),
                    "lr": base_lr * (decay_factor**4),
                    "name": "patch_embed",
                }
            )

        # Swin stages - 递增学习率
        if hasattr(self.backbone, "layers"):
            for i, layer in enumerate(self.backbone.layers):
                param_groups.append(
                    {
                        "params": list(layer.parameters()),
                        "lr": base_lr * (decay_factor ** (3 - i)),
                        "name": f"stage_{i}",
                    }
                )

        # Norm层和其他剩余参数
        remaining_params = []
        processed_params = set()

        # 收集已处理的参数
        for group in param_groups:
            for param in group["params"]:
                processed_params.add(id(param))  # id() 返回对象的唯一标识符

        # 收集未处理的参数
        for param in self.backbone.parameters():
            if id(param) not in processed_params:
                remaining_params.append(param)

        if remaining_params:  # 如果存在未处理的参数，则添加到参数组中
            param_groups.append({"params": remaining_params, "lr": base_lr, "name": "others"})

        logger.info(f"Created {len(param_groups)} parameter groups with layer-wise learning rates")
        return param_groups


# 便捷函数：创建不同变体的模型
def create_swin_tiny(input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2) -> OpticalSwinModel:
    """创建Swin Tiny模型"""
    return OpticalSwinModel(
        model_name="swin_tiny_patch4_window7_224",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def create_swin_small(input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2) -> OpticalSwinModel:
    """创建Swin Small模型"""
    return OpticalSwinModel(
        model_name="swin_small_patch4_window7_224",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def create_swin_base(input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2) -> OpticalSwinModel:
    """创建Swin Base模型"""
    return OpticalSwinModel(
        model_name="swin_base_patch4_window7_224",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


# 测试函数
def test_optical_swin_model():
    """测试模型的基本功能"""
    import torch

    print("Testing OpticalSwinModel...")

    # 创建模型
    model = create_swin_tiny()
    model.eval()

    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, 5, 64, 64)

    # 前向传播测试
    with torch.no_grad():
        features = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output features shape: {features.shape}")
        print(f"Feature dimension: {model.get_feature_dim()}")

    # 模型信息测试
    model.summary()

    print("✓ OpticalSwinModel test passed!")


if __name__ == "__main__":
    test_optical_swin_model()
