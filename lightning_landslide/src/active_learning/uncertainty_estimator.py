# =============================================================================
# lightning_landslide/src/active_learning/uncertainty_estimator.py
# =============================================================================

"""
不确定性估计器 - 评估模型预测的不确定性

这个模块实现了多种不确定性估计方法，用于主动学习中的样本选择。
设计思路：结合认知不确定性和偶然不确定性，全面评估模型的置信度。

核心方法：
1. MC Dropout: 通过多次dropout推理估计认知不确定性
2. Deep Ensemble: 多模型投票评估模型不确定性
3. Temperature Scaling: 校准置信度分数
4. Prediction Entropy: 衡量预测分布的不确定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResults:
    """不确定性估计结果容器"""

    sample_ids: List[str]
    predictions: np.ndarray  # (N, num_classes)
    uncertainty_scores: np.ndarray  # (N,)
    epistemic_uncertainty: np.ndarray  # 认知不确定性
    aleatoric_uncertainty: np.ndarray  # 偶然不确定性
    prediction_entropy: np.ndarray  # 预测熵
    confidence_scores: np.ndarray  # 置信度分数
    calibrated_confidence: np.ndarray  # 校准后置信度


class BaseUncertaintyEstimator(ABC):
    """不确定性估计器基类"""

    def __init__(self, device: str = "auto"):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        )

    @abstractmethod
    def estimate_uncertainty(self, model, dataloader, **kwargs) -> UncertaintyResults:
        """估计不确定性的抽象方法"""
        pass

    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """计算预测熵"""
        # 避免log(0)的情况
        probs = np.clip(probs, 1e-8, 1.0)
        return -np.sum(probs * np.log(probs), axis=1)

    def _compute_confidence(self, probs: np.ndarray) -> np.ndarray:
        """计算置信度(最大概率)"""
        return np.max(probs, axis=1)


class MCDropoutEstimator(BaseUncertaintyEstimator):
    """
    Monte Carlo Dropout不确定性估计器

    原理：在推理时保持dropout开启，通过多次前向传播的变化
    来估计模型的认知不确定性。
    """

    def __init__(
        self, n_forward_passes: int = 50, use_temperature_scaling: bool = False, device: str = "auto"  # 添加这个参数
    ):
        super().__init__(device)
        self.n_forward_passes = n_forward_passes
        self.use_temperature_scaling = use_temperature_scaling  # 保存参数

        # 初始化温度缩放器
        if self.use_temperature_scaling:
            self.temperature_scaler = TemperatureScaling()
        else:
            self.temperature_scaler = None

    def fit_temperature_scaling(self, model, val_dataloader):
        """拟合温度缩放参数"""
        if not self.use_temperature_scaling or self.temperature_scaler is None:
            return

        logger.info("📏 Fitting temperature scaling...")

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits = model(data)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.temperature_scaler.fit(all_logits, all_labels)

    def estimate_uncertainty(self, model, dataloader, **kwargs) -> UncertaintyResults:
        """使用MC Dropout估计不确定性"""
        logger.info(f"🎲 Running MC Dropout with {self.n_forward_passes} forward passes...")

        # === 关键修复：确保模型在正确设备上 ===
        model = model.to(self.device)
        model.eval()
        self._enable_dropout(model)

        all_predictions = []
        all_sample_ids = []

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                # === 关键修复：确保数据在正确设备上 ===
                data = data.to(self.device)
                sample_ids = [f"sample_{batch_idx}_{i}" for i in range(len(data))]

                batch_predictions = []
                for _ in range(self.n_forward_passes):
                    logits = model(data)

                    if self.use_temperature_scaling and self.temperature_scaler is not None:
                        probs = self.temperature_scaler.calibrate(logits)
                    else:
                        probs = F.softmax(logits, dim=1)

                    batch_predictions.append(probs.cpu().numpy())

                batch_predictions = np.stack(batch_predictions, axis=0)
                all_predictions.append(batch_predictions)
                all_sample_ids.extend(sample_ids)

        # 合并所有批次的预测
        all_predictions = np.concatenate(all_predictions, axis=1)  # (n_passes, total_samples, num_classes)

        # 计算各种不确定性指标
        mean_predictions = np.mean(all_predictions, axis=0)  # (total_samples, num_classes)
        prediction_variance = np.var(all_predictions, axis=0)  # (total_samples, num_classes)

        # 认知不确定性：多次预测的方差
        epistemic_uncertainty = np.mean(prediction_variance, axis=1)

        # 偶然不确定性：平均预测的熵
        aleatoric_uncertainty = self._compute_entropy(mean_predictions)

        # 预测熵
        prediction_entropy = self._compute_entropy(mean_predictions)

        # 置信度
        confidence_scores = self._compute_confidence(mean_predictions)

        # 总不确定性：认知 + 偶然
        uncertainty_scores = epistemic_uncertainty + aleatoric_uncertainty

        # 校准后置信度（如果使用温度缩放）
        calibrated_confidence = confidence_scores  # 已经在前向传播中应用了温度缩放

        self._disable_dropout(model)  # 恢复正常状态

        return UncertaintyResults(
            sample_ids=all_sample_ids,
            predictions=mean_predictions,
            uncertainty_scores=uncertainty_scores,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            prediction_entropy=prediction_entropy,
            confidence_scores=confidence_scores,
            calibrated_confidence=calibrated_confidence,
        )

    def _enable_dropout(self, model):
        """在推理时启用dropout"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self, model):
        """恢复正常推理状态"""
        model.eval()


class DeepEnsembleEstimator(BaseUncertaintyEstimator):
    """
    Deep Ensemble不确定性估计器

    原理：训练多个不同初始化的模型，通过模型间的预测差异
    来估计不确定性。
    """

    def __init__(self, model_paths: List[str], device: str = "auto"):
        super().__init__(device)
        self.model_paths = model_paths
        self.models = []

    def load_ensemble_models(self, model_class, **model_kwargs):
        """加载集成模型"""
        logger.info(f"🎯 Loading {len(self.model_paths)} models for ensemble...")

        for path in self.model_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            model.eval()
            self.models.append(model)

        logger.info(f"✓ Loaded {len(self.models)} models for ensemble")

    def estimate_uncertainty(self, model=None, dataloader=None, **kwargs) -> UncertaintyResults:
        """使用模型集成估计不确定性"""
        if not self.models:
            raise ValueError("No models loaded. Call load_ensemble_models() first.")

        logger.info(f"🎪 Running Deep Ensemble with {len(self.models)} models...")

        all_predictions = []
        all_sample_ids = []

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                sample_ids = [f"sample_{batch_idx}_{i}" for i in range(len(data))]

                # 每个模型的预测
                batch_predictions = []
                for model in self.models:
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())

                batch_predictions = np.stack(batch_predictions, axis=0)  # (n_models, batch_size, num_classes)
                all_predictions.append(batch_predictions)
                all_sample_ids.extend(sample_ids)

        # 合并所有批次
        all_predictions = np.concatenate(all_predictions, axis=1)  # (n_models, total_samples, num_classes)

        # 计算不确定性指标
        mean_predictions = np.mean(all_predictions, axis=0)
        prediction_variance = np.var(all_predictions, axis=0)

        # 模型间的不一致性作为不确定性
        epistemic_uncertainty = np.mean(prediction_variance, axis=1)
        aleatoric_uncertainty = self._compute_entropy(mean_predictions)
        prediction_entropy = self._compute_entropy(mean_predictions)
        confidence_scores = self._compute_confidence(mean_predictions)
        uncertainty_scores = epistemic_uncertainty + aleatoric_uncertainty

        return UncertaintyResults(
            sample_ids=all_sample_ids,
            predictions=mean_predictions,
            uncertainty_scores=uncertainty_scores,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            prediction_entropy=prediction_entropy,
            confidence_scores=confidence_scores,
            calibrated_confidence=confidence_scores,
        )


class TemperatureScaling:
    """
    温度缩放校准器

    原理：通过学习一个温度参数T，将模型输出除以T来校准置信度，
    使得预测概率更准确地反映真实的正确率。
    """

    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_fitted = False

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        """在验证集上拟合温度参数"""
        logger.info("🌡️ Fitting temperature scaling...")

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.is_fitted = True

        logger.info(f"✓ Temperature scaling fitted: T = {self.temperature.item():.3f}")

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """应用温度缩放"""
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, using T=1.0")
            return F.softmax(logits, dim=1)

        return F.softmax(logits / self.temperature, dim=1)


class HybridUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    混合不确定性估计器

    结合多种方法的优势，提供更robust的不确定性估计。
    """

    def __init__(
        self,
        use_mc_dropout: bool = True,
        n_forward_passes: int = 30,
        use_temperature_scaling: bool = True,
        device: str = "auto",
    ):
        super().__init__(device)
        self.use_mc_dropout = use_mc_dropout
        self.n_forward_passes = n_forward_passes
        self.use_temperature_scaling = use_temperature_scaling

        if use_mc_dropout:
            self.mc_estimator = MCDropoutEstimator(n_forward_passes, device)
        if use_temperature_scaling:
            self.temperature_scaler = TemperatureScaling()

    def fit_temperature_scaling(self, model, val_dataloader):
        """拟合温度缩放参数"""
        if not self.use_temperature_scaling:
            return

        logger.info("📏 Fitting temperature scaling on validation set...")

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits = model(data)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.temperature_scaler.fit(all_logits, all_labels)

    def estimate_uncertainty(self, model, dataloader, **kwargs) -> UncertaintyResults:
        """使用混合方法估计不确定性"""
        logger.info("🔀 Running hybrid uncertainty estimation...")

        if self.use_mc_dropout:
            # 使用MC Dropout
            mc_results = self.mc_estimator.estimate_uncertainty(model, dataloader)
            return mc_results
        else:
            # 单次推理 + 温度缩放
            model.eval()
            all_predictions = []
            all_sample_ids = []

            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(dataloader):
                    data = data.to(self.device)
                    sample_ids = [f"sample_{batch_idx}_{i}" for i in range(len(data))]

                    logits = model(data)

                    if self.use_temperature_scaling:
                        probs = self.temperature_scaler.calibrate(logits)
                    else:
                        probs = F.softmax(logits, dim=1)

                    all_predictions.append(probs.cpu().numpy())
                    all_sample_ids.extend(sample_ids)

            predictions = np.concatenate(all_predictions, axis=0)

            # 计算基于熵的不确定性
            prediction_entropy = self._compute_entropy(predictions)
            confidence_scores = self._compute_confidence(predictions)

            # 简单的不确定性估计
            uncertainty_scores = prediction_entropy

            return UncertaintyResults(
                sample_ids=all_sample_ids,
                predictions=predictions,
                uncertainty_scores=uncertainty_scores,
                epistemic_uncertainty=uncertainty_scores * 0.5,  # 简化
                aleatoric_uncertainty=uncertainty_scores * 0.5,
                prediction_entropy=prediction_entropy,
                confidence_scores=confidence_scores,
                calibrated_confidence=confidence_scores,
            )


def create_uncertainty_estimator(method: str, **kwargs) -> BaseUncertaintyEstimator:
    """工厂函数：创建不确定性估计器"""
    estimators = {
        "mc_dropout": MCDropoutEstimator,
        "deep_ensemble": DeepEnsembleEstimator,
        "hybrid": HybridUncertaintyEstimator,
    }

    if method not in estimators:
        raise ValueError(f"Unknown uncertainty method: {method}. Available: {list(estimators.keys())}")

    return estimators[method](**kwargs)
