# =============================================================================
# lightning_landslide/src/training/active_kfold_trainer.py
# =============================================================================

"""
主动学习+K折交叉验证融合训练器

这个类结合了K折交叉验证的稳定性和主动学习的数据效率优势。
在每个fold中都应用主动学习策略，最后聚合所有fold的结果。

核心策略：
1. 对每个fold都进行主动学习训练
2. 使用一致的主动学习策略确保公平比较
3. 聚合所有fold的不确定性估计
4. 提供更robust的性能评估

设计思想：
"每个fold都是一个完整的主动学习实验，最终结果是所有实验的综合"
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import pytorch_lightning as pl

from ..data.kfold_extension import create_kfold_wrapper
from ..utils.instantiate import instantiate_from_config
from .simple_kfold_trainer import SimpleKFoldTrainer

logger = logging.getLogger(__name__)


@dataclass
class ActiveKFoldResults:
    """主动学习+K折交叉验证的结果"""

    aggregated_performance: Dict[str, float]
    cross_fold_analysis: Dict[str, Any]
    best_fold_index: int
    ensemble_model_paths: List[str]
    total_training_time: float
    data_efficiency_analysis: Dict[str, Any]

    def to_dict(self):
        """转换为字典格式"""
        return {
            "fold_results": [result.to_dict() for result in self.fold_results],
            "aggregated_performance": self.aggregated_performance,
            "cross_fold_analysis": self.cross_fold_analysis,
            "best_fold_index": self.best_fold_index,
            "ensemble_model_paths": self.ensemble_model_paths,
            "total_training_time": self.total_training_time,
            "data_efficiency_analysis": self.data_efficiency_analysis,
        }


class ActiveKFoldTrainer:
    """
    主动学习+K折交叉验证融合训练器

    这个类是SimpleKFoldTrainer和ActivePseudoTrainer的高级组合，
    在每个fold中都运行完整的主动学习流程。
    """

    def __init__(self, config: Dict[str, Any], experiment_name: str = None, output_dir: str = None):
        """
        初始化主动K折训练器

        Args:
            config: 完整配置（包含kfold和active_pseudo_learning配置）
            experiment_name: 实验名称
            output_dir: 输出目录
        """
        self.config = config
        self.kfold_config = config.get("kfold", {})
        self.active_config = config.get("active_pseudo_learning", {})

        # K折配置
        self.n_splits = self.kfold_config.get("n_splits", 5)
        self.stratified = self.kfold_config.get("stratified", True)
        self.seed = config.get("seed", 3407)

        # 实验管理
        self.experiment_name = experiment_name or config.get("experiment_name", f"active_kfold_{int(time.time())}")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 结果存储
        self.fold_results = []

        logger.info(f"🔄🎯 ActiveKFoldTrainer initialized: {self.experiment_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎲 Using {self.n_splits}-fold cross-validation with active learning")

    def run(self) -> ActiveKFoldResults:
        """运行主动学习+K折交叉验证"""
        logger.info("🚀 Starting Active Learning + K-fold Cross-validation...")
        start_time = time.time()

        # 创建K折数据包装器
        kfold_wrapper = self._create_kfold_wrapper()

        # 为每个fold运行主动学习
        for fold_idx in range(self.n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 FOLD {fold_idx + 1}/{self.n_splits}")
            logger.info(f"{'='*60}")

            fold_result = self._run_active_learning_for_fold(fold_idx, kfold_wrapper)
            self.fold_results.append(fold_result)

            # 保存单折结果
            self._save_fold_result(fold_result, fold_idx)

        # 聚合所有fold的结果
        total_time = time.time() - start_time
        final_results = self._aggregate_fold_results(total_time)

        # 保存最终结果和可视化
        self._save_final_results(final_results)
        self._create_comprehensive_visualization(final_results)

        logger.info(f"\n🎉 Active K-fold training completed!")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")
        logger.info(f"🏆 Mean CV performance: {final_results.aggregated_performance.get('mean_val_f1', 0):.4f}")

        return final_results

    def _create_kfold_wrapper(self):
        """创建K折数据包装器"""
        return create_kfold_wrapper(
            base_datamodule_config=self.config["data"]["params"],
            n_splits=self.n_splits,
            stratified=self.stratified,
            seed=self.seed,
            output_dir=str(self.output_dir / "kfold_info"),
        )

    def _create_fold_config(self, fold_idx: int, kfold_wrapper) -> Dict:
        """为特定fold创建配置"""
        fold_config = self.config.copy()

        # 修改数据配置以使用当前fold的数据
        fold_datamodule = kfold_wrapper.get_fold_datamodule(fold_idx)

        # 这里需要将fold_datamodule转换为配置格式
        # 简化处理：直接使用原配置，在实际数据加载时处理fold分割
        fold_config["fold_index"] = fold_idx
        fold_config["kfold_wrapper"] = kfold_wrapper

        return fold_config

    def _aggregate_fold_results(self, total_time: float) -> ActiveKFoldResults:
        """聚合所有fold的结果"""
        logger.info("📊 Aggregating results from all folds...")

        if not self.fold_results:
            raise ValueError("No fold results to aggregate")

        # 计算聚合性能指标
        aggregated_performance = self._compute_aggregated_performance()

        # 交叉fold分析
        cross_fold_analysis = self._perform_cross_fold_analysis()

        # 找到最佳fold
        best_fold_index = self._find_best_fold()

        # 收集所有最佳模型路径
        ensemble_model_paths = [result.best_model_path for result in self.fold_results]

        # 数据效率分析
        data_efficiency_analysis = self._analyze_data_efficiency()

        return ActiveKFoldResults(
            fold_results=self.fold_results,
            aggregated_performance=aggregated_performance,
            cross_fold_analysis=cross_fold_analysis,
            best_fold_index=best_fold_index,
            ensemble_model_paths=ensemble_model_paths,
            total_training_time=total_time,
            data_efficiency_analysis=data_efficiency_analysis,
        )

    def _compute_aggregated_performance(self) -> Dict[str, float]:
        """计算聚合性能指标"""
        # 收集所有fold的最终性能
        final_performances = []
        best_performances = []

        for result in self.fold_results:
            if result.performance_history["val_f1"]:
                final_f1 = result.performance_history["val_f1"][-1]
                best_f1 = max(result.performance_history["val_f1"])

                final_performances.append(final_f1)
                best_performances.append(best_f1)

        if not final_performances:
            return {}

        final_performances = np.array(final_performances)
        best_performances = np.array(best_performances)

        return {
            "mean_final_f1": float(np.mean(final_performances)),
            "std_final_f1": float(np.std(final_performances)),
            "mean_best_f1": float(np.mean(best_performances)),
            "std_best_f1": float(np.std(best_performances)),
            "min_f1": float(np.min(best_performances)),
            "max_f1": float(np.max(best_performances)),
            "mean_val_f1": float(np.mean(best_performances)),  # 主要指标
            "std_cv_score": float(np.std(best_performances)),  # CV标准差
        }

    def _perform_cross_fold_analysis(self) -> Dict[str, Any]:
        """执行交叉fold分析"""
        analysis = {}

        # 1. 收敛迭代次数分析
        convergence_iterations = [result.convergence_iteration for result in self.fold_results]
        analysis["convergence_analysis"] = {
            "mean_iterations": np.mean(convergence_iterations),
            "std_iterations": np.std(convergence_iterations),
            "min_iterations": np.min(convergence_iterations),
            "max_iterations": np.max(convergence_iterations),
        }

        # 2. 数据使用效率分析
        total_samples_used = []
        pseudo_labels_used = []

        for result in self.fold_results:
            if result.data_usage_history["training_samples"]:
                total_samples_used.append(result.data_usage_history["training_samples"][-1])
                pseudo_labels_used.append(result.data_usage_history["pseudo_labels"][-1])

        if total_samples_used:
            analysis["data_usage_consistency"] = {
                "mean_total_samples": np.mean(total_samples_used),
                "std_total_samples": np.std(total_samples_used),
                "mean_pseudo_labels": np.mean(pseudo_labels_used),
                "std_pseudo_labels": np.std(pseudo_labels_used),
                "pseudo_label_ratio": (
                    np.mean(pseudo_labels_used) / np.mean(total_samples_used) if np.mean(total_samples_used) > 0 else 0
                ),
            }

        # 3. 性能稳定性分析
        if len(self.fold_results) > 1:
            fold_performances = [
                max(result.performance_history["val_f1"]) if result.performance_history["val_f1"] else 0
                for result in self.fold_results
            ]

            analysis["performance_stability"] = {
                "coefficient_of_variation": (
                    np.std(fold_performances) / np.mean(fold_performances) if np.mean(fold_performances) > 0 else 0
                ),
                "performance_range": np.max(fold_performances) - np.min(fold_performances),
                "consistency_score": (
                    1.0 - (np.std(fold_performances) / np.mean(fold_performances))
                    if np.mean(fold_performances) > 0
                    else 0
                ),
            }

        return analysis

    def _find_best_fold(self) -> int:
        """找到表现最好的fold"""
        best_performance = -1
        best_fold_idx = 0

        for i, result in enumerate(self.fold_results):
            if result.performance_history["val_f1"]:
                fold_best = max(result.performance_history["val_f1"])
                if fold_best > best_performance:
                    best_performance = fold_best
                    best_fold_idx = i

        return best_fold_idx

    def _analyze_data_efficiency(self) -> Dict[str, Any]:
        """分析数据使用效率"""
        efficiency_analysis = {}

        # 计算每个fold的数据效率指标
        fold_efficiencies = []

        for i, result in self.fold_results:
            if result.performance_history["val_f1"] and result.data_usage_history["training_samples"]:

                initial_performance = result.performance_history["val_f1"][0]
                final_performance = max(result.performance_history["val_f1"])
                initial_samples = result.data_usage_history["training_samples"][0]
                final_samples = result.data_usage_history["training_samples"][-1]

                # 计算性能提升 / 数据增长比率
                performance_gain = final_performance - initial_performance
                data_growth_ratio = final_samples / initial_samples if initial_samples > 0 else 1

                efficiency = performance_gain / (data_growth_ratio - 1) if data_growth_ratio > 1 else 0
                fold_efficiencies.append(efficiency)

        if fold_efficiencies:
            efficiency_analysis["fold_efficiencies"] = fold_efficiencies
            efficiency_analysis["mean_efficiency"] = np.mean(fold_efficiencies)
            efficiency_analysis["std_efficiency"] = np.std(fold_efficiencies)

        return efficiency_analysis

    def _save_final_results(self, results: ActiveKFoldResults):
        """保存最终聚合结果"""
        # 保存完整结果
        with open(self.output_dir / "active_kfold_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        # 保存性能摘要
        performance_summary = {
            "experiment_name": self.experiment_name,
            "n_splits": self.n_splits,
            "aggregated_performance": results.aggregated_performance,
            "best_fold_index": results.best_fold_index,
            "total_training_time": results.total_training_time,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.output_dir / "performance_summary.json", "w") as f:
            json.dump(performance_summary, f, indent=2)

        # 保存CSV格式的性能历史
        self._save_performance_history_csv(results)

        logger.info(f"📁 Final results saved to {self.output_dir}")

    def _save_performance_history_csv(self, results: ActiveKFoldResults):
        """保存性能历史的CSV文件"""
        # 收集所有fold的性能历史
        all_history = []

        for fold_idx, fold_result in enumerate(results.fold_results):
            for iter_idx, val_f1 in enumerate(fold_result.performance_history["val_f1"]):
                all_history.append(
                    {
                        "fold": fold_idx,
                        "iteration": iter_idx + 1,
                        "val_f1": val_f1,
                        "train_f1": (
                            fold_result.performance_history["train_f1"][iter_idx]
                            if iter_idx < len(fold_result.performance_history["train_f1"])
                            else None
                        ),
                        "val_loss": (
                            fold_result.performance_history["val_loss"][iter_idx]
                            if iter_idx < len(fold_result.performance_history["val_loss"])
                            else None
                        ),
                        "training_samples": (
                            fold_result.data_usage_history["training_samples"][iter_idx]
                            if iter_idx < len(fold_result.data_usage_history["training_samples"])
                            else None
                        ),
                    }
                )

        df = pd.DataFrame(all_history)
        df.to_csv(self.output_dir / "performance_history.csv", index=False)

    def _create_comprehensive_visualization(self, results: ActiveKFoldResults):
        """创建综合可视化报告"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Active Learning + K-fold Cross-validation Results\n{self.experiment_name}", fontsize=16)

        # 1. 每个fold的性能曲线
        for fold_idx, fold_result in enumerate(results.fold_results):
            if fold_result.performance_history["val_f1"]:
                iterations = list(range(1, len(fold_result.performance_history["val_f1"]) + 1))
                axes[0, 0].plot(
                    iterations,
                    fold_result.performance_history["val_f1"],
                    label=f"Fold {fold_idx}",
                    marker="o",
                    alpha=0.7,
                )

        axes[0, 0].set_title("Validation F1 Score by Fold")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. fold性能分布箱线图
        fold_performances = [
            max(result.performance_history["val_f1"]) if result.performance_history["val_f1"] else 0
            for result in results.fold_results
        ]

        axes[0, 1].boxplot(fold_performances)
        axes[0, 1].scatter(range(1, len(fold_performances) + 1), fold_performances, color="red", alpha=0.7)
        axes[0, 1].set_title("Performance Distribution Across Folds")
        axes[0, 1].set_xlabel("Fold")
        axes[0, 1].set_ylabel("Best F1 Score")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 数据使用效率
        for fold_idx, fold_result in enumerate(results.fold_results):
            if fold_result.data_usage_history["training_samples"]:
                iterations = list(range(1, len(fold_result.data_usage_history["training_samples"]) + 1))
                axes[0, 2].plot(
                    iterations,
                    fold_result.data_usage_history["training_samples"],
                    label=f"Fold {fold_idx}",
                    marker="s",
                    alpha=0.7,
                )

        axes[0, 2].set_title("Training Data Growth")
        axes[0, 2].set_xlabel("Iteration")
        axes[0, 2].set_ylabel("Total Training Samples")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 收敛迭代次数
        convergence_iters = [result.convergence_iteration for result in results.fold_results]
        axes[1, 0].bar(range(len(convergence_iters)), convergence_iters, color="skyblue", alpha=0.7)
        axes[1, 0].set_title("Convergence Iterations by Fold")
        axes[1, 0].set_xlabel("Fold Index")
        axes[1, 0].set_ylabel("Iterations to Convergence")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 伪标签使用统计
        pseudo_label_counts = []
        for fold_result in results.fold_results:
            if fold_result.data_usage_history["pseudo_labels"]:
                pseudo_label_counts.append(fold_result.data_usage_history["pseudo_labels"][-1])
            else:
                pseudo_label_counts.append(0)

        axes[1, 1].bar(range(len(pseudo_label_counts)), pseudo_label_counts, color="orange", alpha=0.7)
        axes[1, 1].set_title("Pseudo Labels Used by Fold")
        axes[1, 1].set_xlabel("Fold Index")
        axes[1, 1].set_ylabel("Number of Pseudo Labels")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 综合性能摘要
        metrics = ["Mean F1", "Std F1", "Min F1", "Max F1"]
        values = [
            results.aggregated_performance.get("mean_best_f1", 0),
            results.aggregated_performance.get("std_best_f1", 0),
            results.aggregated_performance.get("min_f1", 0),
            results.aggregated_performance.get("max_f1", 0),
        ]

        bars = axes[1, 2].bar(metrics, values, color=["green", "orange", "red", "blue"], alpha=0.7)
        axes[1, 2].set_title("Aggregated Performance Metrics")
        axes[1, 2].set_ylabel("F1 Score")
        axes[1, 2].tick_params(axis="x", rotation=45)

        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{value:.3f}", ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / "active_kfold_comprehensive_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Comprehensive visualization saved")


def create_active_kfold_trainer(config: Dict, **kwargs) -> ActiveKFoldTrainer:
    """工厂函数：创建主动K折训练器"""
    return ActiveKFoldTrainer(config, **kwargs)
