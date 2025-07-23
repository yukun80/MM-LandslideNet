"""
核心Lightning训练模块

这是整个框架的"大脑"，负责统一管理所有模型的训练过程。
它的设计哲学是"一次编写，处处运行"：

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

    def __init__(self, cfg: DictConfig):
        """
        初始化Lightning模块

        Args:
            cfg: 完整的配置对象
        """
        super().__init__()
        self.cfg = cfg

        # 保存超参数（用于检查点恢复和记录）
        self.save_hyperparameters(cfg)

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

        logger.info("LandslideClassificationModule initialized")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Feature dim: {self.model.get_feature_dim()}")
        logger.info(f"Loss function: {self.criterion.__class__.__name__}")

    def _build_model(self) -> BaseModel:
        """
        根据配置构建骨干模型

        这是工厂方法模式的应用。我们根据配置文件中的模型类型，
        动态创建对应的模型实例。

        Returns:
            配置好的模型实例
        """
        model_type = self.cfg.model.type

        # 模型注册表：添加新模型时只需要在这里注册
        model_registry = {
            "optical_swin": OpticalSwinModel,
            # 未来可以添加更多模型：
            # 'intern_image': InternImageModel,
            # 'efficientnet_dual': EfficientNetDualModel,
            # 'tnf_fusion': TNFFusionModel,
        }

        if model_type not in model_registry:
            available_models = list(model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")

        model_class = model_registry[model_type]
        model = model_class.from_config(self.cfg)

        logger.info(f"Built model: {model_type}")

        # 打印模型信息
        model_info = model.get_model_info()
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")

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
        num_classes = self.cfg.model.num_classes
        dropout_rate = self.cfg.model.get("dropout_rate", 0.2)

        # 支持多种分类头设计
        classifier_type = self.cfg.model.get("classifier_type", "simple")

        if classifier_type == "simple":
            # 简单分类头：LayerNorm + Dropout + Linear
            classifier = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Dropout(dropout_rate), nn.Linear(feature_dim, num_classes)
            )
        elif classifier_type == "mlp":
            # MLP分类头：更复杂的多层感知机
            hidden_dim = self.cfg.model.get("classifier_hidden_dim", feature_dim // 2)
            classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        logger.info(f"Built {classifier_type} classifier: {feature_dim} -> {num_classes}")
        return classifier

    def _build_loss_function(self) -> nn.Module:
        """
        构建损失函数

        支持多种损失函数，特别针对类别不平衡问题提供了多种解决方案。

        Returns:
            配置好的损失函数
        """
        loss_config = self.cfg.training.loss
        loss_type = loss_config.type

        if loss_type == "bce":
            # 标准二元交叉熵损失
            return nn.BCEWithLogitsLoss()

        elif loss_type == "weighted_bce":
            # 加权二元交叉熵，处理类别不平衡
            pos_weight = torch.tensor(loss_config.pos_weight, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        elif loss_type == "focal":
            # Focal Loss，专门处理难样本和类别不平衡
            return FocalLoss(alpha=loss_config.get("alpha", 1.0), gamma=loss_config.get("gamma", 2.0))

        elif loss_type == "dice":
            # Dice Loss，常用于分割任务，也适用于不平衡分类
            return DiceLoss(smooth=loss_config.get("smooth", 1.0))

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _setup_metrics(self) -> None:
        """
        设置评估指标

        为训练和验证设置全面的分类性能评估指标。
        这些指标帮助我们从多个角度评估模型性能。
        """
        task_type = "binary" if self.cfg.model.num_classes == 1 else "multiclass"
        num_classes = self.cfg.model.num_classes if task_type == "multiclass" else 2

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
            x: 输入张量

        Returns:
            预测logits
        """
        # 通过骨干网络提取特征
        features = self.model(x)

        # 通过分类头得到预测
        logits = self.classifier(features)

        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤

        这是训练循环的核心方法。每个训练批次都会调用这个方法。

        Args:
            batch: (数据, 标签) 元组
            batch_idx: 批次索引

        Returns:
            损失值
        """
        x, y = batch

        # 前向传播
        logits = self(x)

        # 计算损失
        loss = self._compute_loss(logits, y)

        # 计算预测概率和二分类结果
        if self.cfg.model.num_classes == 1:
            # 二分类
            probs = torch.sigmoid(logits.squeeze(-1))
            targets = y.int()
        else:
            # 多分类
            probs = torch.softmax(logits, dim=-1)
            targets = y

        # 更新训练指标
        self.train_acc(probs, targets)
        self.train_f1(probs, targets)

        # 记录训练日志（每步记录损失，每个epoch记录指标）
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # 记录学习率
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        验证步骤

        验证时我们计算更全面的指标，用于模型选择和性能评估。

        Args:
            batch: (数据, 标签) 元组
            batch_idx: 批次索引
        """
        x, y = batch

        # 前向传播（验证时不需要梯度）
        logits = self(x)

        # 计算损失
        loss = self._compute_loss(logits, y)

        # 计算预测概率和分类结果
        if self.cfg.model.num_classes == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            targets = y.int()
        else:
            probs = torch.softmax(logits, dim=-1)
            targets = y

        # 更新验证指标
        self.val_acc(probs, targets)
        self.val_f1(probs, targets)
        self.val_precision(probs, targets)
        self.val_recall(probs, targets)
        self.val_auroc(probs, targets)
        self.val_confmat(probs, targets)

        # 记录验证日志
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, logger=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, logger=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        测试步骤

        测试时计算最终的性能评估指标。

        Args:
            batch: (数据, 标签) 元组
            batch_idx: 批次索引
        """
        x, y = batch

        logits = self(x)
        loss = self._compute_loss(logits, y)

        if self.cfg.model.num_classes == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            targets = y.int()
        else:
            probs = torch.softmax(logits, dim=-1)
            targets = y

        # 更新测试指标
        self.test_acc(probs, targets)
        self.test_f1(probs, targets)
        self.test_precision(probs, targets)
        self.test_recall(probs, targets)
        self.test_auroc(probs, targets)
        self.test_confmat(probs, targets)

        # 记录测试日志
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, logger=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, logger=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True, logger=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True, logger=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        统一的损失计算接口，处理不同的任务类型和损失函数。

        Args:
            logits: 模型预测logits
            targets: 真实标签

        Returns:
            损失值
        """
        if self.cfg.model.num_classes == 1:
            # 二分类：确保目标是float类型
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            targets = targets.float()
        else:
            # 多分类：确保目标是long类型
            targets = targets.long()

        return self.criterion(logits, targets)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        这个方法让Lightning知道如何优化模型参数。
        支持多种优化器和学习率调度策略。
        """
        # 获取优化器配置
        opt_cfg = self.cfg.training.optimizer

        # 支持层次化学习率（如果模型提供）
        if hasattr(self.model, "get_layer_wise_lr_groups") and opt_cfg.get("layer_wise_lr", False):
            param_groups = self.model.get_layer_wise_lr_groups(
                base_lr=opt_cfg.lr, decay_factor=opt_cfg.get("lr_decay_factor", 0.8)
            )
            # 添加分类头参数组
            param_groups.append({"params": list(self.classifier.parameters()), "lr": opt_cfg.lr, "name": "classifier"})
            logger.info(f"Using layer-wise learning rates with {len(param_groups)} groups")
        else:
            # 标准参数组
            param_groups = self.parameters()

        # 创建优化器
        if opt_cfg.type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
                eps=opt_cfg.get("eps", 1e-8),
            )
        elif opt_cfg.type == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            )
        elif opt_cfg.type == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=opt_cfg.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.weight_decay,
                nesterov=opt_cfg.get("nesterov", False),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")

        # 配置学习率调度器（如果指定）
        if "scheduler" in self.cfg.training:
            scheduler_cfg = self.cfg.training.scheduler
            scheduler = self._create_scheduler(optimizer, scheduler_cfg)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_cfg.get("monitor", "val_f1"),
                    "interval": scheduler_cfg.get("interval", "epoch"),
                    "frequency": scheduler_cfg.get("frequency", 1),
                },
            }

        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, scheduler_cfg: DictConfig):
        """
        创建学习率调度器

        Args:
            optimizer: 优化器
            scheduler_cfg: 调度器配置

        Returns:
            学习率调度器
        """
        scheduler_type = scheduler_cfg.type

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get("t_max", self.cfg.training.max_epochs),
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_cfg.get("step_size", 30), gamma=scheduler_cfg.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "max"),
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 10),
                verbose=True,
            )
        elif scheduler_type == "linear_warmup":
            return LinearWarmupScheduler(
                optimizer,
                warmup_epochs=scheduler_cfg.get("warmup_epochs", 5),
                total_epochs=self.cfg.training.max_epochs,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def on_validation_epoch_end(self) -> None:
        """
        验证epoch结束时的回调

        在这里我们可以执行一些epoch级别的操作，
        如保存混淆矩阵、计算额外的指标等。
        """
        # 记录混淆矩阵
        if hasattr(self.val_confmat, "compute"):
            confmat = self.val_confmat.compute()
            logger.info(f"Validation Confusion Matrix:\n{confmat}")

    def on_test_epoch_end(self) -> None:
        """
        测试epoch结束时的回调
        """
        # 记录测试混淆矩阵
        if hasattr(self.test_confmat, "compute"):
            confmat = self.test_confmat.compute()
            logger.info(f"Test Confusion Matrix:\n{confmat}")

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        预测步骤（用于推理）

        Args:
            batch: 输入批次
            batch_idx: 批次索引

        Returns:
            预测概率
        """
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        logits = self(x)

        if self.cfg.model.num_classes == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.softmax(logits, dim=-1)

        return probs


# 自定义损失函数
class FocalLoss(nn.Module):
    """
    Focal Loss实现

    Focal Loss专门设计用于处理类别极度不平衡的问题。
    它的核心思想是降低简单样本的权重，让模型更关注困难样本。
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算标准BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # 计算概率
        pt = torch.exp(-bce_loss)

        # 计算focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # 计算focal loss
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss实现

    Dice Loss来自医学图像分割，对类别不平衡问题也有很好的效果。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 将logits转换为概率
        probs = torch.sigmoid(inputs)

        # 计算Dice系数
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # 返回Dice Loss
        return 1 - dice


class LinearWarmupScheduler:
    """
    线性预热学习率调度器

    在训练初期使用较小的学习率，然后线性增加到目标学习率。
    这有助于模型在训练初期的稳定性。
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            # 预热后：余弦退火
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * lr_scale


import numpy as np
