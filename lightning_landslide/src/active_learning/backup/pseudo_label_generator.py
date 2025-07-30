# =============================================================================
# lightning_landslide/src/active_learning/pseudo_label_generator.py
# =============================================================================

"""
伪标签生成器 - 基于置信度和不确定性生成高质量伪标签

这个模块实现了智能的伪标签生成策略，通过多重质量控制机制
确保生成的伪标签不会污染训练数据。

核心策略：
1. 置信度筛选：只选择高置信度的预测
2. 不确定性过滤：排除高不确定性的样本
3. 类别平衡：确保伪标签的类别分布合理
4. 一致性检查：多模型预测一致性验证
5. 质量评估：动态调整伪标签质量要求
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from .uncertainty_estimator import UncertaintyResults

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelSample:
    """单个伪标签样本的信息"""

    sample_id: str
    predicted_label: int
    confidence: float
    uncertainty: float
    quality_score: float
    source_model: str = "single"


@dataclass
class PseudoLabelResults:
    """伪标签生成结果"""

    high_confidence_samples: List[PseudoLabelSample]
    low_confidence_samples: List[PseudoLabelSample]
    excluded_samples: List[PseudoLabelSample]
    statistics: Dict
    quality_distribution: Dict

    def to_dict(self):
        """转换为字典格式以便保存"""
        return {
            "high_confidence_samples": [asdict(s) for s in self.high_confidence_samples],
            "low_confidence_samples": [asdict(s) for s in self.low_confidence_samples],
            "excluded_samples": [asdict(s) for s in self.excluded_samples],
            "statistics": self.statistics,
            "quality_distribution": self.quality_distribution,
        }


class AdaptiveThresholdScheduler:
    """
    自适应阈值调度器

    根据训练进展和模型性能动态调整置信度阈值，
    在训练早期使用较低阈值，随着模型改进逐步提高要求。
    """

    def __init__(self, initial_threshold: float = 0.8, final_threshold: float = 0.95, max_iterations: int = 5):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.max_iterations = max_iterations

    def get_threshold(
        self, current_iteration: int, validation_performance: float = None, pseudo_label_quality: float = None
    ) -> float:
        """计算当前迭代的置信度阈值"""

        # 基础线性调度
        progress = min(current_iteration / self.max_iterations, 1.0)
        base_threshold = self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress

        # 根据验证性能调整
        if validation_performance is not None:
            if validation_performance > 0.85:  # 模型表现好时提高要求
                base_threshold += 0.02
            elif validation_performance < 0.75:  # 模型表现差时降低要求
                base_threshold -= 0.02

        # 根据伪标签质量调整
        if pseudo_label_quality is not None:
            if pseudo_label_quality > 0.9:  # 伪标签质量高时可以适当降低阈值
                base_threshold -= 0.01
            elif pseudo_label_quality < 0.7:  # 伪标签质量低时提高阈值
                base_threshold += 0.03

        # 确保阈值在合理范围内
        return np.clip(base_threshold, 0.6, 0.98)


class ClassBalanceController:
    """
    类别平衡控制器

    确保伪标签的类别分布合理，避免某个类别的伪标签过多
    导致模型偏向某个类别。
    """

    def __init__(self, original_class_distribution: Dict[int, int], balance_ratio: float = 2.0):
        """
        Args:
            original_class_distribution: 原始训练集的类别分布
            balance_ratio: 允许的最大类别不平衡比例
        """
        self.original_distribution = original_class_distribution
        self.balance_ratio = balance_ratio
        self.total_original = sum(original_class_distribution.values())

        # 计算每个类别的目标比例
        self.target_ratios = {cls: count / self.total_original for cls, count in original_class_distribution.items()}

        logger.info(f"📊 Original class distribution: {original_class_distribution}")
        logger.info(f"🎯 Target ratios: {self.target_ratios}")

    def filter_balanced_samples(
        self, samples: List[PseudoLabelSample], max_samples_per_class: Optional[int] = None
    ) -> List[PseudoLabelSample]:
        """根据类别平衡策略筛选样本"""

        # 统计每个类别的样本数量
        class_counts = Counter(s.predicted_label for s in samples)
        logger.info(f"📈 Pseudo-label class distribution: {dict(class_counts)}")

        if max_samples_per_class is None:
            # 根据原始分布计算每个类别的最大样本数
            max_samples_per_class = {
                cls: int(self.total_original * ratio * self.balance_ratio) for cls, ratio in self.target_ratios.items()
            }
        else:
            max_samples_per_class = {cls: max_samples_per_class for cls in self.target_ratios.keys()}

        # 按类别分组并排序
        class_groups = {}
        for sample in samples:
            if sample.predicted_label not in class_groups:
                class_groups[sample.predicted_label] = []
            class_groups[sample.predicted_label].append(sample)

        # 对每个类别按质量分数排序，选择最好的样本
        balanced_samples = []
        for cls, cls_samples in class_groups.items():
            # 按质量分数降序排序
            cls_samples.sort(key=lambda x: x.quality_score, reverse=True)

            # 选择前N个最好的样本
            max_count = max_samples_per_class.get(cls, len(cls_samples))
            selected = cls_samples[:max_count]
            balanced_samples.extend(selected)

            logger.info(f"🏷️ Class {cls}: {len(cls_samples)} → {len(selected)} samples selected")

        return balanced_samples


class PseudoLabelGenerator:
    """
    智能伪标签生成器

    这个类整合了多种质量控制策略，生成高质量的伪标签。
    设计理念：宁缺毋滥 - 只选择最有把握的样本作为伪标签。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        uncertainty_threshold: float = 0.1,
        use_adaptive_threshold: bool = True,
        use_class_balance: bool = True,
        quality_weight_config: Dict[str, float] = None,
    ):
        """
        初始化伪标签生成器

        Args:
            confidence_threshold: 基础置信度阈值
            uncertainty_threshold: 最大允许不确定性
            use_adaptive_threshold: 是否使用自适应阈值
            use_class_balance: 是否进行类别平衡
            quality_weight_config: 质量分数权重配置
        """
        self.base_confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_class_balance = use_class_balance

        # 质量分数权重配置
        default_weights = {
            "confidence_weight": 0.4,
            "uncertainty_weight": 0.3,
            "consistency_weight": 0.2,
            "calibration_weight": 0.1,
        }
        self.quality_weights = quality_weight_config or default_weights

        # 初始化组件
        if use_adaptive_threshold:
            self.threshold_scheduler = AdaptiveThresholdScheduler()

        self.class_balance_controller = None  # 将在设置类别分布时初始化

        logger.info("🏷️ PseudoLabelGenerator initialized")
        logger.info(f"📏 Base confidence threshold: {confidence_threshold}")
        logger.info(f"🔧 Quality weights: {self.quality_weights}")

    def set_class_distribution(self, class_distribution: Dict[int, int]):
        """设置原始训练集的类别分布"""
        if self.use_class_balance:
            self.class_balance_controller = ClassBalanceController(class_distribution)

    def generate_pseudo_labels(
        self, uncertainty_results: UncertaintyResults, current_iteration: int = 0, validation_performance: float = None
    ) -> PseudoLabelResults:
        """
        生成伪标签

        Args:
            uncertainty_results: 不确定性估计结果
            current_iteration: 当前迭代轮次
            validation_performance: 验证集性能

        Returns:
            伪标签生成结果
        """
        logger.info(f"🎯 Generating pseudo labels for iteration {current_iteration}...")

        # 确定当前迭代的置信度阈值
        if self.use_adaptive_threshold:
            current_threshold = self.threshold_scheduler.get_threshold(
                current_iteration=current_iteration, validation_performance=validation_performance
            )
        else:
            current_threshold = self.base_confidence_threshold

        logger.info(f"📊 Using confidence threshold: {current_threshold:.3f}")

        # 生成所有样本的质量分数
        all_samples = self._evaluate_sample_quality(uncertainty_results)

        # 根据阈值分类样本
        high_confidence_samples = []
        low_confidence_samples = []
        excluded_samples = []

        for sample in all_samples:
            if sample.confidence >= current_threshold and sample.uncertainty <= self.uncertainty_threshold:
                high_confidence_samples.append(sample)
            elif sample.confidence >= 0.6:  # 中等置信度样本，可能用于主动学习
                low_confidence_samples.append(sample)
            else:
                excluded_samples.append(sample)

        # 类别平衡处理
        if self.use_class_balance and self.class_balance_controller:
            high_confidence_samples = self.class_balance_controller.filter_balanced_samples(high_confidence_samples)

        # 计算统计信息
        statistics = self._compute_statistics(
            all_samples, high_confidence_samples, low_confidence_samples, excluded_samples
        )

        # 质量分布分析
        quality_distribution = self._analyze_quality_distribution(all_samples)

        results = PseudoLabelResults(
            high_confidence_samples=high_confidence_samples,
            low_confidence_samples=low_confidence_samples,
            excluded_samples=excluded_samples,
            statistics=statistics,
            quality_distribution=quality_distribution,
        )

        logger.info(f"✅ Generated {len(high_confidence_samples)} high-quality pseudo labels")
        logger.info(f"🤔 Found {len(low_confidence_samples)} samples for potential active learning")
        logger.info(f"❌ Excluded {len(excluded_samples)} low-quality samples")

        return results

    def _evaluate_sample_quality(self, uncertainty_results: UncertaintyResults) -> List[PseudoLabelSample]:
        """评估每个样本的质量分数"""
        samples = []

        for i, sample_id in enumerate(uncertainty_results.sample_ids):
            # 获取预测标签
            predicted_label = int(np.argmax(uncertainty_results.predictions[i]))

            # 基础指标
            confidence = uncertainty_results.confidence_scores[i]
            uncertainty = uncertainty_results.uncertainty_scores[i]
            calibrated_confidence = uncertainty_results.calibrated_confidence[i]

            # 计算质量分数（归一化到0-1）
            quality_score = self._compute_quality_score(
                confidence=confidence, uncertainty=uncertainty, calibrated_confidence=calibrated_confidence
            )

            sample = PseudoLabelSample(
                sample_id=sample_id,
                predicted_label=predicted_label,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                quality_score=float(quality_score),
            )

            samples.append(sample)

        return samples

    def _compute_quality_score(self, confidence: float, uncertainty: float, calibrated_confidence: float) -> float:
        """计算综合质量分数"""

        # 归一化指标到0-1范围
        normalized_confidence = confidence  # 已经是0-1
        normalized_uncertainty = 1.0 - min(uncertainty, 1.0)  # 转换为确定性
        normalized_calibrated = calibrated_confidence  # 已经是0-1

        # 加权组合
        quality_score = (
            self.quality_weights["confidence_weight"] * normalized_confidence
            + self.quality_weights["uncertainty_weight"] * normalized_uncertainty
            + self.quality_weights["calibration_weight"] * normalized_calibrated
        )

        # 一致性分数（简化版，实际可以用多模型一致性）
        consistency_score = min(confidence, normalized_uncertainty)
        quality_score += self.quality_weights["consistency_weight"] * consistency_score

        return np.clip(quality_score, 0.0, 1.0)

    def _compute_statistics(self, all_samples, high_conf, low_conf, excluded) -> Dict:
        """计算详细统计信息"""
        total = len(all_samples)

        stats = {
            "total_samples": total,
            "high_confidence_count": len(high_conf),
            "low_confidence_count": len(low_conf),
            "excluded_count": len(excluded),
            "high_confidence_ratio": len(high_conf) / total if total > 0 else 0,
            "low_confidence_ratio": len(low_conf) / total if total > 0 else 0,
            "excluded_ratio": len(excluded) / total if total > 0 else 0,
            "average_quality_score": np.mean([s.quality_score for s in all_samples]),
            "high_conf_avg_quality": np.mean([s.quality_score for s in high_conf]) if high_conf else 0,
            "class_distribution_high_conf": dict(Counter(s.predicted_label for s in high_conf)),
            "class_distribution_low_conf": dict(Counter(s.predicted_label for s in low_conf)),
        }

        return stats

    def _analyze_quality_distribution(self, samples: List[PseudoLabelSample]) -> Dict:
        """分析质量分数分布"""
        quality_scores = [s.quality_score for s in samples]

        if not quality_scores:
            return {}

        quality_stats = {
            "mean": np.mean(quality_scores),
            "std": np.std(quality_scores),
            "min": np.min(quality_scores),
            "max": np.max(quality_scores),
            "percentiles": {
                "25th": np.percentile(quality_scores, 25),
                "50th": np.percentile(quality_scores, 50),
                "75th": np.percentile(quality_scores, 75),
                "90th": np.percentile(quality_scores, 90),
                "95th": np.percentile(quality_scores, 95),
            },
        }

        return quality_stats

    def save_results(self, results: PseudoLabelResults, output_path: Path):
        """保存伪标签结果到文件"""
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存JSON格式的结果
        with open(output_path / "pseudo_label_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # 保存CSV格式的高置信度样本
        if results.high_confidence_samples:
            df = pd.DataFrame([asdict(s) for s in results.high_confidence_samples])
            df.to_csv(output_path / "high_confidence_pseudo_labels.csv", index=False)

        # 保存统计报告
        with open(output_path / "pseudo_label_statistics.json", "w") as f:
            json.dump(results.statistics, f, indent=2)

        logger.info(f"📁 Pseudo label results saved to {output_path}")

    def create_visualization(self, results: PseudoLabelResults, output_path: Path):
        """创建可视化报告"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 样本分布饼图
        labels = ["High Confidence", "Low Confidence", "Excluded"]
        sizes = [
            len(results.high_confidence_samples),
            len(results.low_confidence_samples),
            len(results.excluded_samples),
        ]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        axes[0, 0].set_title("Sample Distribution")

        # 2. 质量分数分布直方图
        all_quality_scores = (
            [s.quality_score for s in results.high_confidence_samples]
            + [s.quality_score for s in results.low_confidence_samples]
            + [s.quality_score for s in results.excluded_samples]
        )

        axes[0, 1].hist(all_quality_scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 1].set_title("Quality Score Distribution")
        axes[0, 1].set_xlabel("Quality Score")
        axes[0, 1].set_ylabel("Count")

        # 3. 类别分布对比
        high_conf_classes = [s.predicted_label for s in results.high_confidence_samples]
        class_counts = Counter(high_conf_classes)

        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            axes[1, 0].bar(classes, counts, color="lightgreen", alpha=0.7)
            axes[1, 0].set_title("High Confidence Class Distribution")
            axes[1, 0].set_xlabel("Class")
            axes[1, 0].set_ylabel("Count")

        # 4. 置信度vs不确定性散点图
        confidences = [s.confidence for s in results.high_confidence_samples]
        uncertainties = [s.uncertainty for s in results.high_confidence_samples]

        if confidences and uncertainties:
            axes[1, 1].scatter(confidences, uncertainties, alpha=0.6, color="purple")
            axes[1, 1].set_title("Confidence vs Uncertainty")
            axes[1, 1].set_xlabel("Confidence")
            axes[1, 1].set_ylabel("Uncertainty")

        plt.tight_layout()
        plt.savefig(output_path / "pseudo_label_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Visualization saved to {output_path / 'pseudo_label_analysis.png'}")


def create_pseudo_label_generator(config: Dict) -> PseudoLabelGenerator:
    """工厂函数：从配置创建伪标签生成器"""
    return PseudoLabelGenerator(
        confidence_threshold=config.get("confidence_threshold", 0.9),
        uncertainty_threshold=config.get("uncertainty_threshold", 0.1),
        use_adaptive_threshold=config.get("use_adaptive_threshold", True),
        use_class_balance=config.get("use_class_balance", True),
        quality_weight_config=config.get("quality_weights", None),
    )
