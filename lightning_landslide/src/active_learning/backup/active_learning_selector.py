# =============================================================================
# lightning_landslide/src/active_learning/active_learning_selector.py
# =============================================================================

"""
主动学习选择器 - 智能选择最有价值的样本进行人工标注

这个模块实现了多种主动学习策略，从无标注数据中选择最有价值的样本
进行人工标注，以最小的标注成本获得最大的模型性能提升。

核心策略：
1. 不确定性采样：选择模型最不确定的样本
2. 多样性采样：避免选择冗余样本，保证代表性
3. 预期模型改变：选择对模型参数影响最大的样本
4. 查询-by-committee：利用模型差异选择争议样本
5. 混合策略：结合多种方法的优势
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from .uncertainty_estimator import UncertaintyResults
from .pseudo_label_generator import PseudoLabelSample

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningQuery:
    """主动学习查询结果"""

    sample_id: str
    uncertainty_score: float
    diversity_score: float
    expected_impact: float
    combined_score: float
    selection_reason: str


@dataclass
class ActiveLearningResults:
    """主动学习选择结果"""

    selected_samples: List[ActiveLearningQuery]
    candidate_pool: List[ActiveLearningQuery]
    selection_statistics: Dict
    diversity_analysis: Dict

    def to_dict(self):
        """转换为字典格式"""
        return {
            "selected_samples": [asdict(s) for s in self.selected_samples],
            "candidate_pool": [asdict(s) for s in self.candidate_pool],
            "selection_statistics": self.selection_statistics,
            "diversity_analysis": self.diversity_analysis,
        }


class BaseActiveLearningStrategy(ABC):
    """主动学习策略基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def score_samples(
        self, uncertainty_results: UncertaintyResults, feature_embeddings: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """为样本计算主动学习分数"""
        pass


class UncertaintyStrategy(BaseActiveLearningStrategy):
    """
    不确定性采样策略

    选择模型预测不确定性最高的样本。这是最直观和常用的策略，
    基于的假设是：模型不确定的样本往往是最难的，学习这些样本
    能最大化模型的改进。
    """

    def __init__(self, uncertainty_method: str = "entropy"):
        super().__init__("uncertainty")
        self.uncertainty_method = uncertainty_method

    def score_samples(
        self, uncertainty_results: UncertaintyResults, feature_embeddings: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """基于不确定性为样本评分"""

        if self.uncertainty_method == "entropy":
            return uncertainty_results.prediction_entropy
        elif self.uncertainty_method == "confidence":
            return 1.0 - uncertainty_results.confidence_scores  # 低置信度 = 高不确定性
        elif self.uncertainty_method == "combined":
            return uncertainty_results.uncertainty_scores
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")


class DiversityStrategy(BaseActiveLearningStrategy):
    """
    多样性采样策略

    选择在特征空间中彼此差异较大的样本，避免选择过于相似的样本。
    这有助于获得更具代表性的训练数据。
    """

    def __init__(self, distance_metric: str = "cosine", n_clusters: int = None):
        super().__init__("diversity")
        self.distance_metric = distance_metric
        self.n_clusters = n_clusters

    def score_samples(
        self, uncertainty_results: UncertaintyResults, feature_embeddings: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """基于多样性为样本评分"""

        if feature_embeddings is None:
            logger.warning("No feature embeddings provided for diversity strategy")
            return np.ones(len(uncertainty_results.sample_ids))

        # 计算样本间的距离矩阵
        if self.distance_metric == "cosine":
            similarity_matrix = cosine_similarity(feature_embeddings)
            distance_matrix = 1.0 - similarity_matrix
        elif self.distance_metric == "euclidean":
            distance_matrix = euclidean_distances(feature_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # 计算每个样本的多样性分数（到其他样本的平均距离）
        diversity_scores = np.mean(distance_matrix, axis=1)

        # 归一化到0-1范围
        diversity_scores = (diversity_scores - diversity_scores.min()) / (
            diversity_scores.max() - diversity_scores.min() + 1e-8
        )

        return diversity_scores


class ClusterBasedStrategy(BaseActiveLearningStrategy):
    """
    基于聚类的采样策略

    先将样本聚类，然后从每个聚类中选择最不确定的样本。
    这确保了选择的样本既有代表性又有挑战性。
    """

    def __init__(self, n_clusters: int = 10, cluster_method: str = "kmeans"):
        super().__init__("cluster_based")
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method

    def score_samples(
        self, uncertainty_results: UncertaintyResults, feature_embeddings: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """基于聚类的采样评分"""

        if feature_embeddings is None:
            logger.warning("No feature embeddings provided for cluster-based strategy")
            return uncertainty_results.uncertainty_scores

        # 聚类
        if self.cluster_method == "kmeans":
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(feature_embeddings)
        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")

        # 为每个聚类内的样本计算相对分数
        cluster_scores = np.zeros(len(uncertainty_results.sample_ids))
        uncertainty_scores = uncertainty_results.uncertainty_scores

        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) == 0:
                continue

            cluster_uncertainties = uncertainty_scores[cluster_mask]
            # 在聚类内按不确定性排序
            cluster_ranks = np.argsort(cluster_uncertainties)[::-1]  # 降序

            # 分配分数：聚类内排名靠前的样本得分更高
            for i, sample_idx in enumerate(np.where(cluster_mask)[0]):
                rank_in_cluster = np.where(cluster_ranks == i)[0][0]
                cluster_scores[sample_idx] = 1.0 - (rank_in_cluster / len(cluster_ranks))

        return cluster_scores


class QueryByCommitteeStrategy(BaseActiveLearningStrategy):
    """
    Query-by-Committee策略

    利用多个模型之间的预测差异来选择样本。当多个模型对某个样本
    的预测分歧很大时，说明这个样本很有学习价值。
    """

    def __init__(self, committee_predictions: List[np.ndarray] = None):
        super().__init__("query_by_committee")
        self.committee_predictions = committee_predictions or []

    def set_committee_predictions(self, predictions: List[np.ndarray]):
        """设置委员会模型的预测结果"""
        self.committee_predictions = predictions

    def score_samples(
        self, uncertainty_results: UncertaintyResults, feature_embeddings: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """基于模型间分歧为样本评分"""

        if not self.committee_predictions:
            logger.warning("No committee predictions available, falling back to uncertainty")
            return uncertainty_results.uncertainty_scores

        n_samples = len(uncertainty_results.sample_ids)
        disagreement_scores = np.zeros(n_samples)

        # 计算模型间的平均分歧
        for i in range(n_samples):
            sample_predictions = [pred[i] for pred in self.committee_predictions]

            # 计算模型间的KL散度平均值
            kl_divergences = []
            for j in range(len(sample_predictions)):
                for k in range(j + 1, len(sample_predictions)):
                    p = sample_predictions[j] + 1e-8  # 避免log(0)
                    q = sample_predictions[k] + 1e-8
                    kl_div = np.sum(p * np.log(p / q))
                    kl_divergences.append(kl_div)

            disagreement_scores[i] = np.mean(kl_divergences) if kl_divergences else 0.0

        # 归一化
        if disagreement_scores.max() > disagreement_scores.min():
            disagreement_scores = (disagreement_scores - disagreement_scores.min()) / (
                disagreement_scores.max() - disagreement_scores.min()
            )

        return disagreement_scores


class HybridActiveLearningSelector:
    """
    混合主动学习选择器

    结合多种主动学习策略的优势，通过加权组合不同策略的分数
    来选择最有价值的样本。
    """

    def __init__(
        self, strategies: Dict[str, float] = None, feature_extractor: Callable = None, budget_per_iteration: int = 50
    ):
        """
        初始化混合选择器

        Args:
            strategies: 策略权重字典，例如 {"uncertainty": 0.6, "diversity": 0.4}
            feature_extractor: 特征提取函数
            budget_per_iteration: 每轮迭代的标注预算
        """
        # 默认策略权重
        default_strategies = {"uncertainty": 0.5, "diversity": 0.3, "cluster_based": 0.2}
        self.strategy_weights = strategies or default_strategies
        self.feature_extractor = feature_extractor
        self.budget_per_iteration = budget_per_iteration

        # 初始化策略实例
        self.strategies = {
            "uncertainty": UncertaintyStrategy(),
            "diversity": DiversityStrategy(),
            "cluster_based": ClusterBasedStrategy(),
            "query_by_committee": QueryByCommitteeStrategy(),
        }

        logger.info(f"🎯 HybridActiveLearningSelector initialized with strategies: {self.strategy_weights}")

    def select_samples(
        self,
        uncertainty_results: UncertaintyResults,
        candidate_samples: List[PseudoLabelSample],
        feature_embeddings: Optional[np.ndarray] = None,
        committee_predictions: Optional[List[np.ndarray]] = None,
        budget: Optional[int] = None,
    ) -> ActiveLearningResults:
        """
        选择最有价值的样本进行主动学习

        Args:
            uncertainty_results: 不确定性估计结果
            candidate_samples: 候选样本列表（低置信度样本）
            feature_embeddings: 特征嵌入
            committee_predictions: 委员会模型预测
            budget: 本轮标注预算

        Returns:
            主动学习选择结果
        """
        budget = budget or self.budget_per_iteration
        logger.info(f"🔍 Selecting {budget} samples from {len(candidate_samples)} candidates...")

        # 构建候选样本的映射
        candidate_indices = []
        candidate_id_to_idx = {}

        for i, sample in enumerate(candidate_samples):
            if sample.sample_id in uncertainty_results.sample_ids:
                idx = uncertainty_results.sample_ids.index(sample.sample_id)
                candidate_indices.append(idx)
                candidate_id_to_idx[sample.sample_id] = idx

        if not candidate_indices:
            logger.warning("No matching candidates found in uncertainty results")
            return ActiveLearningResults([], [], {}, {})

        # 过滤相关的不确定性结果和特征
        filtered_uncertainty = self._filter_uncertainty_results(uncertainty_results, candidate_indices)
        filtered_features = feature_embeddings[candidate_indices] if feature_embeddings is not None else None

        # 计算各策略的分数
        strategy_scores = {}
        for strategy_name, weight in self.strategy_weights.items():
            if weight <= 0:
                continue

            strategy = self.strategies[strategy_name]

            # 特殊处理query_by_committee
            if strategy_name == "query_by_committee" and committee_predictions:
                filtered_committee_preds = [pred[candidate_indices] for pred in committee_predictions]
                strategy.set_committee_predictions(filtered_committee_preds)

            scores = strategy.score_samples(filtered_uncertainty, filtered_features)
            strategy_scores[strategy_name] = scores

            logger.debug(f"📊 {strategy_name} scores: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

        # 加权组合策略分数
        combined_scores = self._combine_strategy_scores(strategy_scores, self.strategy_weights)

        # 创建查询对象
        queries = []
        for i, sample in enumerate(candidate_samples):
            if sample.sample_id not in candidate_id_to_idx:
                continue

            uncertainty_idx = candidate_id_to_idx[sample.sample_id]
            local_idx = candidate_indices.index(uncertainty_idx)

            query = ActiveLearningQuery(
                sample_id=sample.sample_id,
                uncertainty_score=float(filtered_uncertainty.uncertainty_scores[local_idx]),
                diversity_score=(
                    float(strategy_scores.get("diversity", [0])[local_idx]) if "diversity" in strategy_scores else 0.0
                ),
                expected_impact=float(combined_scores[local_idx]),
                combined_score=float(combined_scores[local_idx]),
                selection_reason=self._get_selection_reason(strategy_scores, local_idx),
            )
            queries.append(query)

        # 按组合分数排序并选择top-k
        queries.sort(key=lambda x: x.combined_score, reverse=True)
        selected_samples = queries[:budget]

        # 计算统计信息
        selection_stats = self._compute_selection_statistics(queries, selected_samples)
        diversity_analysis = (
            self._analyze_diversity(selected_samples, filtered_features) if filtered_features is not None else {}
        )

        results = ActiveLearningResults(
            selected_samples=selected_samples,
            candidate_pool=queries,
            selection_statistics=selection_stats,
            diversity_analysis=diversity_analysis,
        )

        logger.info(f"✅ Selected {len(selected_samples)} samples for annotation")
        logger.info(f"📈 Average combined score: {np.mean([s.combined_score for s in selected_samples]):.3f}")

        return results

    def _filter_uncertainty_results(
        self, uncertainty_results: UncertaintyResults, indices: List[int]
    ) -> UncertaintyResults:
        """过滤不确定性结果以匹配候选样本"""
        return UncertaintyResults(
            sample_ids=[uncertainty_results.sample_ids[i] for i in indices],
            predictions=uncertainty_results.predictions[indices],
            uncertainty_scores=uncertainty_results.uncertainty_scores[indices],
            epistemic_uncertainty=uncertainty_results.epistemic_uncertainty[indices],
            aleatoric_uncertainty=uncertainty_results.aleatoric_uncertainty[indices],
            prediction_entropy=uncertainty_results.prediction_entropy[indices],
            confidence_scores=uncertainty_results.confidence_scores[indices],
            calibrated_confidence=uncertainty_results.calibrated_confidence[indices],
        )

    def _combine_strategy_scores(self, strategy_scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """加权组合多个策略的分数"""
        if not strategy_scores:
            return np.array([])

        # 归一化每个策略的分数到0-1范围
        normalized_scores = {}
        for name, scores in strategy_scores.items():
            if len(scores) == 0:
                continue
            score_min, score_max = scores.min(), scores.max()
            if score_max > score_min:
                normalized_scores[name] = (scores - score_min) / (score_max - score_min)
            else:
                normalized_scores[name] = np.ones_like(scores)

        # 加权组合
        n_samples = len(list(normalized_scores.values())[0])
        combined = np.zeros(n_samples)
        total_weight = 0

        for name, scores in normalized_scores.items():
            weight = weights.get(name, 0)
            combined += weight * scores
            total_weight += weight

        # 归一化权重
        if total_weight > 0:
            combined /= total_weight

        return combined

    def _get_selection_reason(self, strategy_scores: Dict[str, np.ndarray], idx: int) -> str:
        """生成样本选择的原因描述"""
        reasons = []

        for strategy_name, scores in strategy_scores.items():
            if len(scores) <= idx:
                continue
            score = scores[idx]
            percentile = (scores <= score).mean() * 100

            if percentile >= 90:
                reasons.append(f"high_{strategy_name}")
            elif percentile >= 70:
                reasons.append(f"medium_{strategy_name}")

        return ", ".join(reasons) if reasons else "low_priority"

    def _compute_selection_statistics(
        self, all_queries: List[ActiveLearningQuery], selected: List[ActiveLearningQuery]
    ) -> Dict:
        """计算选择统计信息"""
        if not all_queries:
            return {}

        all_scores = [q.combined_score for q in all_queries]
        selected_scores = [q.combined_score for q in selected]

        stats = {
            "total_candidates": len(all_queries),
            "selected_count": len(selected),
            "selection_ratio": len(selected) / len(all_queries),
            "score_statistics": {
                "all_mean": np.mean(all_scores),
                "all_std": np.std(all_scores),
                "selected_mean": np.mean(selected_scores) if selected_scores else 0,
                "selected_min": np.min(selected_scores) if selected_scores else 0,
                "selected_max": np.max(selected_scores) if selected_scores else 0,
            },
            "strategy_contributions": self.strategy_weights.copy(),
        }

        return stats

    def _analyze_diversity(self, selected_samples: List[ActiveLearningQuery], feature_embeddings: np.ndarray) -> Dict:
        """分析选择样本的多样性"""
        if len(selected_samples) < 2 or feature_embeddings is None:
            return {}

        # 提取选择样本的特征
        selected_indices = list(range(len(selected_samples)))  # 这里需要实际的索引映射

        # 计算样本间距离
        if len(selected_indices) >= 2:
            selected_features = feature_embeddings[: len(selected_samples)]  # 简化处理
            distances = euclidean_distances(selected_features)

            # 去除对角线（自身距离）
            mask = np.ones(distances.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            inter_distances = distances[mask]

            diversity_analysis = {
                "mean_distance": float(np.mean(inter_distances)),
                "min_distance": float(np.min(inter_distances)),
                "max_distance": float(np.max(inter_distances)),
                "std_distance": float(np.std(inter_distances)),
                "diversity_score": float(np.mean(inter_distances)),  # 平均距离作为多样性指标
            }

            return diversity_analysis

        return {}

    def save_results(self, results: ActiveLearningResults, output_path: Path):
        """保存主动学习结果"""
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存完整结果
        with open(output_path / "active_learning_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # 保存选择的样本ID列表（用于后续标注）
        selected_ids = [s.sample_id for s in results.selected_samples]
        with open(output_path / "samples_for_annotation.txt", "w") as f:
            f.write("\n".join(selected_ids))

        # 保存详细的选择结果CSV
        if results.selected_samples:
            df = pd.DataFrame([asdict(s) for s in results.selected_samples])
            df.to_csv(output_path / "selected_samples_detailed.csv", index=False)

        logger.info(f"📁 Active learning results saved to {output_path}")

    def create_visualization(self, results: ActiveLearningResults, output_path: Path):
        """创建主动学习选择的可视化"""
        if not results.selected_samples:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 分数分布对比
        all_scores = [q.combined_score for q in results.candidate_pool]
        selected_scores = [q.combined_score for q in results.selected_samples]

        axes[0, 0].hist(all_scores, bins=30, alpha=0.5, label="All Candidates", color="lightblue")
        axes[0, 0].hist(selected_scores, bins=20, alpha=0.7, label="Selected", color="red")
        axes[0, 0].set_title("Score Distribution")
        axes[0, 0].set_xlabel("Combined Score")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # 2. 不确定性vs多样性散点图
        uncertainty_scores = [q.uncertainty_score for q in results.candidate_pool]
        diversity_scores = [q.diversity_score for q in results.candidate_pool]
        selected_uncertainty = [q.uncertainty_score for q in results.selected_samples]
        selected_diversity = [q.diversity_score for q in results.selected_samples]

        axes[0, 1].scatter(uncertainty_scores, diversity_scores, alpha=0.5, color="lightgray", label="Candidates")
        axes[0, 1].scatter(selected_uncertainty, selected_diversity, alpha=0.8, color="red", s=50, label="Selected")
        axes[0, 1].set_title("Uncertainty vs Diversity")
        axes[0, 1].set_xlabel("Uncertainty Score")
        axes[0, 1].set_ylabel("Diversity Score")
        axes[0, 1].legend()

        # 3. 选择原因饼图
        reasons = [q.selection_reason for q in results.selected_samples]
        reason_counts = pd.Series(reasons).value_counts()

        if len(reason_counts) > 0:
            axes[1, 0].pie(reason_counts.values, labels=reason_counts.index, autopct="%1.1f%%")
            axes[1, 0].set_title("Selection Reasons")

        # 4. 策略贡献权重
        if results.selection_statistics.get("strategy_contributions"):
            strategies = list(results.selection_statistics["strategy_contributions"].keys())
            weights = list(results.selection_statistics["strategy_contributions"].values())

            axes[1, 1].bar(strategies, weights, color="skyblue")
            axes[1, 1].set_title("Strategy Weights")
            axes[1, 1].set_xlabel("Strategy")
            axes[1, 1].set_ylabel("Weight")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "active_learning_visualization.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Visualization saved to {output_path / 'active_learning_visualization.png'}")


def create_active_learning_selector(config: Dict) -> HybridActiveLearningSelector:
    """工厂函数：从配置创建主动学习选择器"""
    return HybridActiveLearningSelector(
        strategies=config.get("strategies", None), budget_per_iteration=config.get("budget_per_iteration", 50)
    )
