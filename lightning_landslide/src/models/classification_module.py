# =============================================================================
# lightning_landslide/src/models/classification_module.py - 滑坡分类训练模块
# =============================================================================

"""
滑坡检测的核心Lightning训练模块

这个模块是整个框架的"大脑"，它统一管理所有模型的训练过程。
设计哲学与latent-diffusion相似："一次编写，处处运行"：

- 任何继承BaseModel的模型都能无缝接入
- 支持多种损失函数和优化策略
- 自动处理评估指标和日志记录
- 内置模型检查点和早停机制

这个设计的威力在于：当您添加新的模型（如InternImage、EfficientNet）时，
不需要重写任何训练逻辑，只需要实现BaseModel接口即可。

教学要点：
这个模块展示了面向对象设计的精髓：通过抽象接口实现多态性。
不同的模型有不同的内部实现，但对于训练框架来说，它们都是"模型"。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional, List, Tuple, Callable
from omegaconf import DictConfig
import logging
import numpy as np
from pathlib import Path

from .base import BaseModel
from .optical_swin import OpticalSwinModel

logger = logging.getLogger(__name__)


class LandslideClassificationModule(pl.LightningModule):
    """
    滑坡检测的核心Lightning模块

    这个类是整个训练框架的核心。它接受任何符合BaseModel接口的模型，
    并为其提供完整的训练、验证、测试支持。

    核心功能：
    1. 模型管理：动态创建和配置不同的模型
    2. 训练循环：标准化的训练、验证、测试流程
    3. 损失计算：支持多种损失函数
    4. 指标评估：全面的分类性能评估
    5. 优化器配置：灵活的优化策略

    设计优势：
    - 模型无关：支持任何BaseModel子类
    - 配置驱动：通过YAML文件控制所有行为
    - 可扩展：易于添加新的损失函数和指标
    - 可复现：自动保存配置和随机种子
    """

    def __init__(
        self,
        # 模型配置
        base_model: Dict[str, Any],
        # 训练配置
        loss_config: Dict[str, Any] = None,
        optimizer_config: Dict[str, Any] = None,
        scheduler_config: Dict[str, Any] = None,
        # 评估配置
        metrics_config: Dict[str, Any] = None,
        # 其他配置
        classifier_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        初始化Lightning模块

        Args:
            base_model: 基础模型配置
            loss_config: 损失函数配置
            optimizer_config: 优化器配置
            scheduler_config: 学习率调度器配置
            metrics_config: 评估指标配置
            classifier_config: 分类头配置
        """
        super().__init__()

        # 保存配置用于检查点恢复
        self.save_hyperparameters()

        # 解析配置
        self.base_model_config = base_model
        self.loss_config = loss_config or {"type": "bce"}
        self.optimizer_config = optimizer_config or {"type": "adamw", "adamw_params": {"lr": 1e-4}}
        self.scheduler_config = scheduler_config
        self.metrics_config = metrics_config or {"primary_metric": "f1"}
        self.classifier_config = classifier_config or {"type": "simple"}

        # 创建骨干模型
        self.model = self._build_model()

        # 创建分类头
        self.classifier = self._build_classifier()

        # 创建损失函数
        self.criterion = self._build_loss_function()

        # 初始化评估指标
        self._setup_metrics()

        # 训练状态跟踪
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        logger.info("🚀LandslideClassificationModule initialized" + "-" * 100)
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Feature dim: {self.model.get_feature_dim()}")
        logger.info(f"Loss function: {self.criterion.__class__.__name__}")
        logger.info("-" * 100)

    def _build_model(self) -> BaseModel:
        """
        根据配置构建骨干模型

        这是工厂方法模式的应用。我们根据配置文件中的模型类型，
        动态创建对应的模型实例。这种设计让添加新模型变得非常简单。

        Returns:
            配置好的模型实例
        """
        from ..utils.instantiate import instantiate_from_config

        logger.info("Building backbone model...")
        # 创建pytorch模型
        model = instantiate_from_config(self.base_model_config)

        if not isinstance(model, BaseModel):
            raise TypeError(f"Model must inherit from BaseModel, got {type(model)}")

        # 打印模型信息
        logger.info("📦model info:" + "-" * 100)
        model_info = model.get_model_info()
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        logger.info("-" * 100)

        return model

    def _build_classifier(self) -> nn.Module:
        """
        构建分类头

        分类头是连接特征提取器和最终预测的桥梁。
        我们根据任务类型和配置参数来设计分类头。

        Returns:
            配置好的分类头模块
        """
        feature_dim = self.model.get_feature_dim()
        num_classes = self.base_model_config["params"].get("num_classes", 1)
        dropout_rate = self.classifier_config.get("dropout_rate", 0.2)

        # 支持多种分类头设计
        classifier_type = self.classifier_config.get("type", "simple")

        if classifier_type == "simple":
            # 简单分类头：LayerNorm + Dropout + Linear
            classifier = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Dropout(dropout_rate), nn.Linear(feature_dim, num_classes)
            )
        elif classifier_type == "mlp":
            # MLP分类头：更复杂的多层感知机
            hidden_dim = self.classifier_config.get("hidden_dim", feature_dim // 2)
            classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        elif classifier_type == "attention":
            # 基于注意力机制的分类头（可以进一步实现）
            raise NotImplementedError("Attention classifier not implemented yet")
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        logger.info(f"Built {classifier_type} classifier: {feature_dim} -> {num_classes}")
        return classifier

    def _build_loss_function(self) -> nn.Module:
        """
        构建损失函数

        支持多种损失函数，特别针对类别不平衡问题提供了多种解决方案。
        这种设计让我们可以轻松尝试不同的损失函数来找到最适合的方案。

        Returns:
            配置好的损失函数
        """
        loss_type = self.loss_config["type"]

        if loss_type == "bce":
            # 标准二元交叉熵损失
            return nn.BCEWithLogitsLoss()

        elif loss_type == "weighted_bce":
            # 加权二元交叉熵，处理类别不平衡
            pos_weight = torch.tensor(self.loss_config.get("pos_weight", 1.0))
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        elif loss_type == "focal":
            # Focal Loss，专门处理难样本和类别不平衡
            focal_params = self.loss_config.get("focal_params", {})
            return FocalLoss(alpha=focal_params.get("alpha", 1.0), gamma=focal_params.get("gamma", 2.0))

        elif loss_type == "dice":
            # Dice Loss，常用于分割任务，也适用于不平衡分类
            return DiceLoss(smooth=self.loss_config.get("smooth", 1.0))

        elif loss_type == "combined":
            # 组合损失：可以结合多种损失函数的优势
            # 例如：BCE + Dice，或者 Focal + Dice
            raise NotImplementedError("Combined loss not implemented yet")

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _setup_metrics(self) -> None:
        """
        设置评估指标

        为训练和验证设置全面的分类性能评估指标。
        这些指标帮助我们从多个角度评估模型性能。

        在滑坡检测任务中，我们特别关注：
        - F1分数：平衡精确率和召回率，适合不平衡数据
        - AUROC：模型区分能力的综合评估
        - 精确率：预测为滑坡的准确程度
        - 召回率：真实滑坡被检测出的比例
        """
        task_type = "binary"  # 假设是二分类任务
        num_classes = 2

        # 训练指标（计算开销较小的指标）
        self.train_acc = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)

        # 验证指标（更全面的评估）
        self.val_acc = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task=task_type, num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

        # 测试指标
        self.test_acc = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.test_precision = torchmetrics.Precision(task=task_type, num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

        # 混淆矩阵（用于详细分析）
        self.val_confmat = torchmetrics.ConfusionMatrix(task=task_type, num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(task=task_type, num_classes=num_classes)

        logger.info(f"Setup metrics for {task_type} classification")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, channels, height, width]

        Returns:
            预测logits [batch_size, num_classes]
        """
        # 通过骨干网络提取特征
        features = self.model(x)

        # 通过分类头得到预测
        logits = self.classifier(features)

        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤

        这个方法定义了单个训练批次的处理逻辑。PyTorch Lightning
        会自动调用这个方法，并处理梯度计算、反向传播等细节。

        Args:
            batch: 输入批次 (x, y)
            batch_idx: 批次索引

        Returns:
            损失值
        """
        x, y = batch

        # 前向传播
        logits = self(x)

        # 确保标签维度正确
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        # 计算损失
        loss = self.criterion(logits, y)

        # 计算预测概率
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # 更新训练指标
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        # 记录指标
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        验证步骤

        验证过程不需要梯度计算，主要用于评估模型在未见过数据上的性能。
        我们使用更全面的指标来评估模型。

        Args:
            batch: 输入批次 (x, y)
            batch_idx: 批次索引
        """
        x, y = batch

        # 前向传播
        logits = self(x)

        # 确保标签维度正确
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        # 计算损失
        loss = self.criterion(logits, y)

        # 计算预测概率和预测类别
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # 更新验证指标
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_auroc(probs, y.long())
        self.val_confmat(preds, y)

        # 记录指标
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        测试步骤

        测试阶段提供最终的模型性能评估，通常在模型选择和结果报告中使用。

        Args:
            batch: 输入批次 (x, y)
            batch_idx: 批次索引
        """
        x, y = batch

        # 前向传播
        logits = self(x)

        # 确保标签维度正确
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        # 计算损失
        loss = self.criterion(logits, y)

        # 计算预测概率和预测类别
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # 更新测试指标
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_auroc(probs, y.long())
        self.test_confmat(preds, y)

        # 记录指标
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        预测步骤

        用于生成预测结果，通常用于推理和竞赛提交。

        Args:
            batch: 输入批次
            batch_idx: 批次索引

        Returns:
            包含预测结果的字典
        """
        x = batch if isinstance(batch, torch.Tensor) else batch[0]

        # 前向传播
        logits = self(x)

        # 确保logits维度正确
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        # 计算预测概率和预测类别
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        return {"logits": logits, "probabilities": probs, "predictions": preds}

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器

        这个方法展示了如何灵活配置不同的优化策略。
        它支持差分学习率、学习率调度等高级技巧。

        Returns:
            优化器和调度器配置
        """
        # 获取模型参数
        if self.optimizer_config.get("differential_lr", {}).get("enable", False):
            # 差分学习率：骨干网络使用较小的学习率，分类头使用较大的学习率
            optimizer_params = self._create_param_groups()
        else:
            # 统一学习率
            optimizer_params = self.parameters()

        # 创建优化器
        optimizer_type = self.optimizer_config["type"]
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(optimizer_params, **self.optimizer_config["adamw_params"])
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(optimizer_params, **self.optimizer_config["adam_params"])
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_params, **self.optimizer_config["sgd_params"])
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # 配置学习率调度器（如果有的话）
        if self.scheduler_config is None:
            return optimizer

        scheduler_type = self.scheduler_config["type"]
        if scheduler_type == "cosine_with_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            scheduler = CosineAnnealingWarmRestarts(optimizer, **self.scheduler_config["cosine_params"])
        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR

            scheduler = StepLR(optimizer, **self.scheduler_config["step_params"])
        elif scheduler_type == "reduce_on_plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_config["plateau_params"])
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.scheduler_config.get("monitor", "val_loss"),
                "frequency": self.scheduler_config.get("frequency", 1),
            },
        }

    def _create_param_groups(self) -> List[Dict[str, Any]]:
        """
        创建差分学习率参数组

        这是一个高级技巧：对预训练的骨干网络使用较小的学习率，
        对新的分类头使用较大的学习率。这样可以在微调时获得更好的效果。

        Returns:
            参数组列表
        """
        differential_config = self.optimizer_config["differential_lr"]
        base_lr = self.optimizer_config["adamw_params"]["lr"]

        backbone_lr = base_lr * differential_config.get("backbone_lr_ratio", 0.1)
        classifier_lr = base_lr * differential_config.get("classifier_lr_ratio", 1.0)

        param_groups = [
            {"params": self.model.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": self.classifier.parameters(), "lr": classifier_lr, "name": "classifier"},
        ]

        logger.info(f"Created parameter groups with differential learning rates:")
        logger.info(f"  Backbone LR: {backbone_lr}")
        logger.info(f"  Classifier LR: {classifier_lr}")

        return param_groups


class FocalLoss(nn.Module):
    """
    Focal Loss实现 - 专门处理类别不平衡问题

    Focal Loss是Facebook AI Research提出的损失函数，特别适合处理
    极度不平衡的分类问题。在遥感滑坡检测中，负样本（非滑坡）通常
    远多于正样本（滑坡），这正是Focal Loss的用武之地。

    核心思想：
    1. 降低易分类样本的权重（让模型专注于困难样本）
    2. 增加少数类的权重（解决类别不平衡）

    公式：FL(p_t) = -α(1-p_t)^γ * log(p_t)
    其中：p_t是模型对正确类别的预测概率
    α控制类别权重，γ控制困难样本的权重
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()  # 将Long类型转换为Float类型

        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # 计算预测概率
        p_t = torch.exp(-bce_loss)

        # 应用focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # 应用类别权重alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 计算focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss实现 - 适用于分割任务的损失函数

    Dice Loss基于Dice系数（也称为F1分数），特别适合处理
    分割任务中的类别不平衡问题。在滑坡检测中，它能够更好地
    关注预测区域与真实区域的重叠程度。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 应用sigmoid获得概率
        inputs = torch.sigmoid(inputs)

        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算交集和并集
        intersection = (inputs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff
