from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """
    所有深度学习模型的抽象基类

    这个基类就像建筑设计中的"建筑规范"。无论您建造的是住宅楼还是
    办公楼，都必须有门、窗、电力接口等基础设施。同样，无论您使用的
    是Swin Transformer还是InternImage，都必须实现forward()和
    get_feature_dim()等基础接口。

    为什么这样设计？
    1. 让Lightning训练模块无需了解每个模型的内部细节
    2. 确保所有模型都能提供训练所需的基本信息
    3. 为模型集成和比较提供标准化接口
    """

    def __init__(self):
        super().__init__()
        # 用于存储模型的额外信息
        self._model_info = {}
        # 用于跟踪模型状态
        self._is_frozen = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型前向传播

        这是神经网络最核心的方法。我们在这里定义统一的接口，
        确保所有模型都能以相同的方式被调用。

        设计要点：
        - 输入：原始数据张量（具体维度由子类定义）
        - 输出：特征表示（而不是分类结果）
        - 原因：这样分类头可以独立配置和优化

        Args:
            x: 输入张量，具体形状由子类定义

        Returns:
            特征张量，用于分类头处理。形状通常是 (batch_size, feature_dim)

        Note:
            子类应该确保输出是特征表示，而不是最终分类结果。
            这种设计让我们可以轻松地切换不同的分类策略。
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        获取输出特征维度

        这个信息对于构建分类头至关重要。比如，如果这个方法返回768，
        Lightning模块就知道要创建一个输入维度为768的线性分类器。

        为什么这样设计？
        - 避免硬编码：分类头维度自动适配不同的骨干网络
        - 类型安全：编译时就能发现维度不匹配的问题
        - 信息透明：训练框架能够了解模型的输出规格

        Returns:
            特征维度（正整数）
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: DictConfig) -> "BaseModel":
        """
        从配置创建模型实例（工厂方法模式）

        这是设计模式中的"工厂方法"，它的威力在于：

        1. 配置驱动：通过修改配置文件就能创建不同的模型
        2. 延迟绑定：训练时才决定具体使用哪个模型
        3. 参数验证：在创建时就能检查配置的有效性

        实际应用场景：
        ```python
        # 在配置文件中指定模型类型
        cfg.model.type = "optical_swin"

        # 框架根据配置自动创建对应模型
        model = model_registry[cfg.model.type].from_config(cfg)
        ```

        Args:
            cfg: 完整的配置对象

        Returns:
            配置好的模型实例
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息（可选重写）

        这个方法让模型具备"自我介绍"的能力。在调试和优化时，
        我们经常需要了解模型的基本信息。

        返回的信息包括：
        - 参数数量：了解模型复杂度
        - 内存占用：估算资源需求
        - 特征维度：检查架构设计
        - 模型名称：便于日志记录
        """
        # 计算参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 估算模型大小（假设float32，每个参数4字节）
        model_size_mb = total_params * 4 / (1024 * 1024)

        # 打印模型信息
        base_info = {
            "model_name": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "feature_dim": self.get_feature_dim(),
            "is_frozen": self._is_frozen,
        }

        # 合并子类可能添加的额外信息
        base_info.update(self._model_info)

        return base_info

    def freeze_backbone(self) -> None:
        """
        冻结骨干网络参数

        这在迁移学习中非常有用。有时我们只想训练分类头，
        而保持预训练的特征提取器不变。

        使用场景：
        - 数据量较小时，避免过拟合
        - 快速实验时，减少训练时间
        - 渐进训练时，先训练分类头再微调整个网络
        """
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True
        logger.info(f"Frozen all parameters for {self.__class__.__name__}")

    def unfreeze_backbone(self) -> None:
        """解冻骨干网络参数，允许端到端训练"""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False
        logger.info(f"Unfrozen all parameters for {self.__class__.__name__}")

    def freeze_layers_except_classifier(self, classifier_names: list = ["classifier", "head", "fc"]):
        """
        只解冻分类头，冻结其他所有层

        这是一个更精细的冻结策略，经常用于迁移学习的初期阶段。

        Args:
            classifier_names: 分类头可能的名称列表
        """
        # 先冻结所有参数
        self.freeze_backbone()

        # 然后解冻分类头
        for name, module in self.named_modules():
            if any(cls_name in name for cls_name in classifier_names):
                for param in module.parameters():
                    param.requires_grad = True

        logger.info(f"Frozen backbone but kept classifier trainable for {self.__class__.__name__}")

    def get_layer_wise_lr_groups(self, base_lr: float = 1e-4, decay_factor: float = 0.8):
        """
        创建层次化学习率分组

        这是一个高级技巧：让模型的不同层使用不同的学习率。
        通常情况下，靠近输入的层（更通用的特征）使用较小的学习率，
        靠近输出的层（更任务特定的特征）使用较大的学习率。

        Args:
            base_lr: 分类头的学习率
            decay_factor: 每向前一层，学习率的衰减因子

        Returns:
            适用于优化器的参数组列表
        """
        # 这是一个模板实现，子类可以根据具体架构重写
        param_groups = []

        # 简单实现：骨干网络用较小学习率，分类头用基础学习率
        backbone_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if any(cls_name in name for cls_name in ["classifier", "head", "fc"]):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * decay_factor, "name": "backbone"})

        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": base_lr, "name": "classifier"})

        return param_groups

    def count_parameters(self) -> Tuple[int, int]:
        """
        统计模型参数数量

        Returns:
            (总参数数, 可训练参数数)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self) -> None:
        """打印模型摘要信息"""
        info = self.get_model_info()
        print("\n" + "=" * 50)
        print(f"Model: {info['model_name']}")
        print("=" * 50)
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print(f"Model size: {info['model_size_mb']:.2f} MB")
        print(f"Feature dimension: {info['feature_dim']}")
        print(f"Frozen status: {info['is_frozen']}")
        print("=" * 50 + "\n")
