# =============================================================================
# lightning_landslide/src/models/optical_convnext.py - ConvNextv2分类模型
# =============================================================================

"""
基于ConvNextv2的光学数据模型

这个模块是OpticalSwinModel的"兄弟"实现，遵循完全相同的接口设计，
让用户可以通过简单修改配置文件就能在Swin Transformer和ConvNextv2之间切换。

核心特性：
1. 处理5通道输入：R, G, B, NIR, NDVI
2. 动态上采样：从64x64自适应到目标尺寸
3. 特征分离：分离特征提取和分类决策
4. 智能权重初始化：充分利用预训练权重

设计哲学：
- 接口一致性：与OpticalSwinModel提供相同的接口
- 配置驱动：通过配置文件控制所有行为
- 即插即用：可以无缝替换现有的Swin模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)


class OpticalConvNextModel(BaseModel):
    """
    基于ConvNextv2的光学数据模型

    ConvNextv2相比Swin Transformer的优势：
    1. 更简单的架构，训练更稳定
    2. 更好的尺度不变性
    3. 在某些任务上精度更高
    4. 推理速度通常更快

    核心功能与OpticalSwinModel完全一致：
    - 5通道输入处理
    - 动态尺寸适配
    - 预训练权重智能初始化
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        input_channels: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        pretrained_path: Optional[str] = None,
        img_size: int = 256,
    ):
        super().__init__()

        # 保存配置信息
        self.model_name = model_name
        self.input_channels = input_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.img_size = img_size

        logger.info(f"Initializing OpticalConvNextModel with {model_name}")
        logger.info(f"Target image size: {self.img_size}x{self.img_size}")

        # 本地权重优先级处理
        use_timm_pretrained = pretrained and (pretrained_path is None)

        # 创建ConvNextv2骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_timm_pretrained,
            num_classes=0,  # 移除分类头，只要特征提取
            global_pool="avg",  # 全局平均池化
        )

        # 加载本地权重（如果提供）
        if pretrained_path:
            self._load_local_weights(pretrained_path)

        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")

        # 修改输入层以支持5通道
        self._modify_input_layer()

        # 保存模型信息
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

        logger.info("🚀 OpticalConvNextModel initialization completed successfully")

    def _load_local_weights(self, pretrained_path: str):
        """从本地文件加载预训练权重"""
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weights file not found at: {pretrained_path}")

        try:
            state_dict = torch.load(pretrained_path, map_location="cpu")

            # 处理不同格式的权重文件
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # 加载权重
            result = self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Local weight loading result: {result}")

            # 记录未匹配的键（用于调试）
            if result.missing_keys:
                logger.warning(f"Missing keys: {result.missing_keys[:5]}...")  # 只显示前5个
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys: {result.unexpected_keys[:5]}...")

        except Exception as e:
            logger.error(f"Failed to load local weights from {pretrained_path}: {e}")
            raise

    def _modify_input_layer(self) -> None:
        """
        修改输入层以处理5通道输入

        ConvNextv2的输入层通常是stem模块中的第一个卷积层。
        我们需要将其从3通道扩展到5通道，并智能地初始化新增通道的权重。

        策略：
        1. 找到第一个卷积层
        2. 创建新的5通道卷积层
        3. 智能复制和初始化权重
        4. 替换原有层
        """
        if self.input_channels == 3:
            logger.info("Input channels is 3, no modification needed")
            return

        # 寻找第一个卷积层
        first_conv_name, first_conv, old_weight = self._find_first_conv()
        if first_conv is None:
            raise RuntimeError("Cannot find the first convolutional layer")

        logger.info(f"Found first conv layer: {first_conv_name}")
        logger.info(f"Original weight shape: {old_weight.shape}")

        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # 智能初始化权重
        with torch.no_grad():
            if old_weight.shape[1] == 3:  # 预训练权重是3通道
                # 前3个通道直接复制RGB权重
                new_conv.weight[:, :3, :, :] = old_weight

                # NIR通道：使用红光通道的权重（相近波段）
                new_conv.weight[:, 3:4, :, :] = old_weight[:, 0:1, :, :] * 0.8  # 轻微衰减

                # NDVI通道：结合红光和近红外的特征
                if self.input_channels >= 5:
                    red_weight = old_weight[:, 0:1, :, :]  # R通道
                    nir_weight = old_weight[:, 0:1, :, :] * 0.8  # 模拟NIR
                    # NDVI通常是(NIR-R)/(NIR+R)的变化，这里简化为差异
                    new_conv.weight[:, 4:5, :, :] = (nir_weight - red_weight) * 0.5

                # 如果有更多通道，使用NIR的模式
                for i in range(5, self.input_channels):
                    new_conv.weight[:, i : i + 1, :, :] = new_conv.weight[:, 3:4, :, :]

                logger.info("✓ Successfully initialized 5-channel weights from 3-channel pretrained weights")

            else:
                # 如果预训练权重不是3通道，使用标准初始化
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                logger.warning(f"Unexpected pretrained weight channels: {old_weight.shape[1]}, using random init")

            # 复制bias
            if new_conv.bias is not None and first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        # 替换原有层
        self._replace_layer(first_conv_name, new_conv)
        logger.info(f"✓ Successfully replaced input layer: {first_conv_name}")

    def _find_first_conv(self) -> Tuple[str, Optional[nn.Conv2d], Optional[torch.Tensor]]:
        """
        寻找第一个卷积层

        ConvNextv2的架构中，第一个卷积层通常在：
        - stem.0 或 stem.conv
        - downsample_layers.0.0 或类似路径

        Returns:
            (layer_name, conv_layer, weight_tensor)
        """
        # 常见的第一层路径模式
        common_paths = [
            "stem.0",
            "stem.conv",
            "downsample_layers.0.0",
            "downsample_layers.0.conv",
            "features.0",
            "features.stem.0",
        ]

        # 首先尝试常见路径
        for path in common_paths:
            layer = self._get_layer_by_path(path)
            if isinstance(layer, nn.Conv2d):
                return path, layer, layer.weight.data

        # 如果常见路径找不到，遍历所有模块
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                logger.info(f"Found first conv layer through traversal: {name}")
                return name, module, module.weight.data

        return None, None, None

    def _get_layer_by_path(self, path: str) -> Optional[nn.Module]:
        """通过路径获取层"""
        try:
            current = self.backbone
            for part in path.split("."):
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current
        except (AttributeError, IndexError, KeyError):
            return None

    def _replace_layer(self, layer_name: str, new_layer: nn.Module) -> None:
        """
        替换指定名称的层

        支持嵌套路径，如 "stem.0" 或 "downsample_layers.0.0"
        """
        parts = layer_name.split(".")
        current = self.backbone

        # 导航到父模块
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        # 替换目标层
        final_part = parts[-1]
        if final_part.isdigit():
            current[int(final_part)] = new_layer
        else:
            setattr(current, final_part, new_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        处理流程：
        1. 验证输入维度
        2. 动态上采样到目标尺寸
        3. 通过骨干网络提取特征
        """
        # 输入验证
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channel, height, width), got {x.dim()}D")

        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")

        # 动态上采样
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # 特征提取
        features = self.backbone(x)
        return features

    def get_feature_dim(self) -> int:
        """
        返回特征维度

        ConvNextv2的不同变体有不同的特征维度：
        - tiny: 768
        - small: 768
        - base: 1024
        - large: 1536
        """
        return self.feature_dim

    def get_features_with_intermediate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取中间层特征（用于可视化和分析）

        ConvNextv2是分阶段的架构，我们可以提取每个阶段的特征。
        """
        features = {}

        # 输入处理
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # 尝试提取各阶段特征
        current = x
        stage_names = ["stem", "stage1", "stage2", "stage3", "stage4"]

        try:
            # Stem处理
            if hasattr(self.backbone, "stem"):
                current = self.backbone.stem(current)
                features["stem"] = current
            elif hasattr(self.backbone, "downsample_layers"):
                current = self.backbone.downsample_layers[0](current)
                features["stem"] = current

            # 各个stage
            if hasattr(self.backbone, "stages"):
                for i, stage in enumerate(self.backbone.stages):
                    current = stage(current)
                    features[f"stage_{i+1}"] = current

            # 最终特征
            if hasattr(self.backbone, "norm"):
                current = self.backbone.norm(current)
            if hasattr(self.backbone, "head"):
                if hasattr(self.backbone.head, "global_pool"):
                    current = self.backbone.head.global_pool(current)
                    current = current.flatten(1)

            features["final"] = current

        except Exception as e:
            logger.warning(f"Failed to extract intermediate features: {e}")
            # 回退到完整前向传播
            features["final"] = self.backbone(x)

        return features

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "OpticalConvNextModel":
        """
        从配置创建模型实例

        支持的配置参数：
        - model_name: ConvNextv2变体名称
        - input_channels: 输入通道数
        - pretrained: 是否使用预训练权重
        - dropout_rate: dropout比率
        - pretrained_path: 本地权重路径
        - img_size: 目标图像尺寸
        """
        # 从配置中提取参数，提供合理的默认值
        return cls(
            model_name=cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"),
            input_channels=cfg.get("input_channels", 5),
            pretrained=cfg.get("pretrained", True),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            pretrained_path=cfg.get("pretrained_path", None),
            img_size=cfg.get("img_size", 256),
        )

    def freeze_stem(self) -> None:
        """
        只冻结stem层

        在某些迁移学习场景中，我们可能只想冻结输入处理部分，
        而让后续的特征学习层继续训练。
        """
        if hasattr(self.backbone, "stem"):
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
            logger.info("Frozen stem layers")
        elif hasattr(self.backbone, "downsample_layers"):
            for param in self.backbone.downsample_layers[0].parameters():
                param.requires_grad = False
            logger.info("Frozen first downsample layer (stem equivalent)")

    def unfreeze_last_stages(self, num_stages: int = 2) -> None:
        """
        解冻最后几个stage

        这是一种常见的渐进解冻策略：先训练分类头，
        然后逐步解冻更多的层。

        Args:
            num_stages: 要解冻的最后几个stage数量
        """
        if hasattr(self.backbone, "stages"):
            total_stages = len(self.backbone.stages)
            start_stage = max(0, total_stages - num_stages)

            for i in range(start_stage, total_stages):
                for param in self.backbone.stages[i].parameters():
                    param.requires_grad = True

            logger.info(f"Unfroze last {num_stages} stages (stages {start_stage}-{total_stages-1})")

    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        获取分层的训练参数组

        ConvNextv2可以使用分层学习率：
        - stem: 最小学习率
        - early stages: 较小学习率
        - later stages: 较大学习率
        - classifier: 最大学习率（由外部分类头管理）
        """
        param_groups = []
        base_lr = 1e-4  # 这会被优化器配置覆盖

        # Stem参数组
        stem_params = []
        if hasattr(self.backbone, "stem"):
            stem_params.extend(self.backbone.stem.parameters())
        elif hasattr(self.backbone, "downsample_layers"):
            stem_params.extend(self.backbone.downsample_layers[0].parameters())

        if stem_params:
            param_groups.append({"params": stem_params, "lr": base_lr * 0.1, "name": "stem"})

        # Stage参数组
        if hasattr(self.backbone, "stages"):
            num_stages = len(self.backbone.stages)
            for i, stage in enumerate(self.backbone.stages):
                # 后面的stage使用更大的学习率
                lr_multiplier = 0.2 + 0.6 * (i / max(1, num_stages - 1))
                param_groups.append(
                    {"params": list(stage.parameters()), "lr": base_lr * lr_multiplier, "name": f"stage_{i}"}
                )

        # 其他参数（norm等）
        handled_params = set()
        for group in param_groups:
            handled_params.update(id(p) for p in group["params"])

        remaining_params = []
        for param in self.parameters():
            if id(param) not in handled_params:
                remaining_params.append(param)

        if remaining_params:
            param_groups.append({"params": remaining_params, "lr": base_lr, "name": "others"})

        logger.info(f"Created {len(param_groups)} parameter groups with layered learning rates")
        return param_groups


# 便捷函数：创建不同变体的ConvNextv2模型
def create_convnext_tiny(
    input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2
) -> OpticalConvNextModel:
    """创建ConvNextv2 Tiny模型"""
    return OpticalConvNextModel(
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def create_convnext_small(
    input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2
) -> OpticalConvNextModel:
    """创建ConvNextv2 Small模型"""
    return OpticalConvNextModel(
        model_name="convnextv2_small.fcmae_ft_in22k_in1k",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def create_convnext_base(
    input_channels: int = 5, pretrained: bool = True, dropout_rate: float = 0.2
) -> OpticalConvNextModel:
    """创建ConvNextv2 Base模型"""
    return OpticalConvNextModel(
        model_name="convnextv2_base.fcmae_ft_in22k_in1k",
        input_channels=input_channels,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


# 测试功能
def test_optical_convnext_model():
    """测试模型的基本功能"""
    print("Testing OpticalConvNextModel...")

    # 创建模型
    model = create_convnext_tiny()
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

    print("✓ OpticalConvNextModel test passed!")


if __name__ == "__main__":
    test_optical_convnext_model()
