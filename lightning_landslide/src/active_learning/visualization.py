# =============================================================================
# lightning_landslide/src/active_learning/visualization.py
# =============================================================================

"""
主动学习可视化分析和监控工具

这个模块提供了丰富的可视化功能，帮助理解和监控主动学习过程：
1. 实时训练监控
2. 不确定性分析可视化
3. 数据分布变化追踪
4. 性能趋势分析
5. 交互式报告生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """可视化配置"""

    style: str = "seaborn"
    color_palette: str = "husl"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    interactive: bool = True


class ActiveLearningVisualizer:
    """
    主动学习可视化器

    提供全面的可视化分析功能，包括静态图表和交互式报告。
    """

    def __init__(self, output_dir: Path, config: VisualizationConfig = None):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
            config: 可视化配置
        """
        self.output_dir = Path(output_dir)
        self.config = config or VisualizationConfig()

        # 创建可视化目录
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # 设置样式
        self._setup_style()

        logger.info(f"📊 ActiveLearningVisualizer initialized: {self.viz_dir}")

    def _setup_style(self):
        """设置可视化样式"""
        if self.config.style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_palette(self.config.color_palette)

        plt.rcParams["figure.figsize"] = self.config.figure_size
        plt.rcParams["figure.dpi"] = self.config.dpi

    def create_training_overview(
        self,
        performance_history: Dict[str, List[float]],
        data_usage_history: Dict[str, List[int]],
        iteration_results: List[Dict],
    ) -> str:
        """
        创建训练过程总览

        Args:
            performance_history: 性能历史数据
            data_usage_history: 数据使用历史
            iteration_results: 迭代结果列表

        Returns:
            保存的图片路径
        """
        logger.info("📈 Creating training overview visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Active Learning Training Overview", fontsize=16, fontweight="bold")

        iterations = list(range(1, len(performance_history["val_f1"]) + 1))

        # 1. 性能改进曲线
        axes[0, 0].plot(
            iterations, performance_history["val_f1"], "b-o", linewidth=2, markersize=6, label="Validation F1"
        )
        axes[0, 0].plot(
            iterations, performance_history.get("train_f1", []), "g-s", linewidth=2, markersize=6, label="Training F1"
        )
        axes[0, 0].set_title("Model Performance Over Iterations", fontweight="bold")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)

        # 添加性能提升标注
        if len(performance_history["val_f1"]) > 1:
            improvement = performance_history["val_f1"][-1] - performance_history["val_f1"][0]
            axes[0, 0].annotate(
                f"Improvement: +{improvement:.3f}",
                xy=(iterations[-1], performance_history["val_f1"][-1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        # 2. 数据增长趋势
        axes[0, 1].plot(
            iterations, data_usage_history["training_samples"], "r-o", linewidth=2, markersize=6, label="Total Training"
        )
        axes[0, 1].plot(
            iterations,
            data_usage_history.get("pseudo_labels", []),
            "orange",
            marker="s",
            linewidth=2,
            markersize=6,
            label="Pseudo Labels",
        )
        axes[0, 1].plot(
            iterations,
            data_usage_history.get("new_annotations", []),
            "purple",
            marker="^",
            linewidth=2,
            markersize=6,
            label="New Annotations",
        )
        axes[0, 1].set_title("Data Usage Over Iterations", fontweight="bold")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Number of Samples")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 损失下降
        if "val_loss" in performance_history:
            axes[0, 2].plot(
                iterations, performance_history["val_loss"], "r-o", linewidth=2, markersize=6, label="Validation Loss"
            )
            axes[0, 2].set_title("Loss Reduction Over Iterations", fontweight="bold")
            axes[0, 2].set_xlabel("Iteration")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. 数据效率分析
        if len(performance_history["val_f1"]) > 1:
            sample_counts = data_usage_history["training_samples"]
            f1_scores = performance_history["val_f1"]

            axes[1, 0].scatter(
                sample_counts,
                f1_scores,
                c=range(len(sample_counts)),
                cmap="viridis",
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=1,
            )
            axes[1, 0].plot(sample_counts, f1_scores, "b--", alpha=0.5, linewidth=1)
            axes[1, 0].set_title("Performance vs Data Usage", fontweight="bold")
            axes[1, 0].set_xlabel("Training Samples")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].grid(True, alpha=0.3)

            # 添加颜色条
            cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
            cbar.set_label("Iteration")

        # 5. 训练时间分析
        if iteration_results:
            training_times = [result.get("training_time", 0) for result in iteration_results]
            cumulative_times = np.cumsum(training_times)

            axes[1, 1].bar(
                iterations,
                training_times,
                alpha=0.7,
                color="lightblue",
                edgecolor="navy",
                linewidth=1,
                label="Per Iteration",
            )
            axes[1, 1].plot(iterations, cumulative_times, "r-o", linewidth=2, markersize=6, label="Cumulative")
            axes[1, 1].set_title("Training Time Analysis", fontweight="bold")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Time (seconds)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. 数据组成饼图
        if iteration_results:
            latest_result = iteration_results[-1]
            total_samples = latest_result.get("total_training_samples", 0)
            pseudo_count = latest_result.get("pseudo_label_count", 0)
            new_annotations = latest_result.get("new_annotations_count", 0)
            original_count = total_samples - pseudo_count - new_annotations

            sizes = [original_count, pseudo_count, new_annotations]
            labels = ["Original", "Pseudo Labels", "New Annotations"]
            colors = ["#ff9999", "#66b3ff", "#99ff99"]

            wedges, texts, autotexts = axes[1, 2].pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, explode=(0.05, 0.05, 0.05)
            )
            axes[1, 2].set_title("Final Data Composition", fontweight="bold")

            # 美化饼图
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

        plt.tight_layout()

        # 保存图片
        output_path = self.viz_dir / "training_overview.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        logger.info(f"💾 Training overview saved: {output_path}")
        return str(output_path)

    def create_uncertainty_analysis(self, uncertainty_results: Dict, pseudo_label_results: Dict, iteration: int) -> str:
        """
        创建不确定性分析图表

        Args:
            uncertainty_results: 不确定性估计结果
            pseudo_label_results: 伪标签生成结果
            iteration: 迭代轮次

        Returns:
            保存的图片路径
        """
        logger.info(f"🎲 Creating uncertainty analysis for iteration {iteration}...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Uncertainty Analysis - Iteration {iteration}", fontsize=16, fontweight="bold")

        # 从结果中提取数据
        uncertainty_scores = uncertainty_results.get("uncertainty_scores", [])
        confidence_scores = uncertainty_results.get("confidence_scores", [])
        predictions = uncertainty_results.get("predictions", [])

        if len(uncertainty_scores) == 0:
            logger.warning("No uncertainty data available for visualization")
            return ""

        # 1. 不确定性分布
        axes[0, 0].hist(uncertainty_scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black", density=True)
        axes[0, 0].axvline(
            np.mean(uncertainty_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(uncertainty_scores):.3f}",
        )
        axes[0, 0].set_title("Uncertainty Score Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Uncertainty Score")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 置信度分布
        axes[0, 1].hist(confidence_scores, bins=30, alpha=0.7, color="lightgreen", edgecolor="black", density=True)
        axes[0, 1].axvline(
            np.mean(confidence_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(confidence_scores):.3f}",
        )
        axes[0, 1].set_title("Confidence Score Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Confidence Score")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 置信度vs不确定性散点图
        scatter = axes[0, 2].scatter(
            confidence_scores,
            uncertainty_scores,
            alpha=0.6,
            c=confidence_scores,
            cmap="viridis",
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[0, 2].set_title("Confidence vs Uncertainty", fontweight="bold")
        axes[0, 2].set_xlabel("Confidence Score")
        axes[0, 2].set_ylabel("Uncertainty Score")
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label="Confidence")

        # 4. 伪标签质量分析
        if pseudo_label_results and "high_confidence_samples" in pseudo_label_results:
            high_conf_samples = pseudo_label_results["high_confidence_samples"]
            if high_conf_samples:
                quality_scores = [s.get("quality_score", 0) for s in high_conf_samples]
                axes[1, 0].hist(quality_scores, bins=20, alpha=0.7, color="gold", edgecolor="black", density=True)
                axes[1, 0].axvline(
                    np.mean(quality_scores),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {np.mean(quality_scores):.3f}",
                )
                axes[1, 0].set_title("Pseudo Label Quality Distribution", fontweight="bold")
                axes[1, 0].set_xlabel("Quality Score")
                axes[1, 0].set_ylabel("Density")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

        # 5. 预测类别分布
        if len(predictions) > 0:
            if predictions[0].ndim > 1:  # 多类别预测
                predicted_classes = np.argmax(predictions, axis=1)
            else:  # 二分类
                predicted_classes = (np.array(predictions) > 0.5).astype(int)

            class_counts = np.bincount(predicted_classes)
            class_labels = [f"Class {i}" for i in range(len(class_counts))]

            bars = axes[1, 1].bar(
                class_labels,
                class_counts,
                alpha=0.7,
                color=["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][: len(class_counts)],
                edgecolor="black",
                linewidth=1,
            )
            axes[1, 1].set_title("Predicted Class Distribution", fontweight="bold")
            axes[1, 1].set_xlabel("Class")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(True, alpha=0.3)

            # 添加数值标签
            for bar, count in zip(bars, class_counts):
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(class_counts) * 0.01,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # 6. 样本选择统计
        if pseudo_label_results:
            stats = pseudo_label_results.get("statistics", {})
            categories = ["High Conf.", "Low Conf.", "Excluded"]
            counts = [
                stats.get("high_confidence_count", 0),
                stats.get("low_confidence_count", 0),
                stats.get("excluded_count", 0),
            ]
            colors = ["green", "orange", "red"]

            bars = axes[1, 2].bar(categories, counts, alpha=0.7, color=colors, edgecolor="black", linewidth=1)
            axes[1, 2].set_title("Sample Selection Results", fontweight="bold")
            axes[1, 2].set_xlabel("Category")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].grid(True, alpha=0.3)

            # 添加百分比标签
            total = sum(counts) if sum(counts) > 0 else 1
            for bar, count in zip(bars, counts):
                percentage = count / total * 100
                axes[1, 2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f"{count}\n({percentage:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()

        # 保存图片
        output_path = self.viz_dir / f"uncertainty_analysis_iter_{iteration}.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        logger.info(f"💾 Uncertainty analysis saved: {output_path}")
        return str(output_path)

    def create_interactive_dashboard(self, complete_results: Dict, experiment_name: str) -> str:
        """
        创建交互式仪表板

        Args:
            complete_results: 完整的实验结果
            experiment_name: 实验名称

        Returns:
            保存的HTML文件路径
        """
        if not self.config.interactive:
            logger.info("Interactive visualization disabled")
            return ""

        logger.info("🎛️ Creating interactive dashboard...")

        # 创建子图
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Performance Over Time",
                "Data Usage Growth",
                "Training Time Analysis",
                "Uncertainty Distribution",
                "Data Composition",
                "Performance Correlation",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "pie"}, {"secondary_y": False}],
            ],
        )

        # 提取数据
        performance_history = complete_results.get("performance_history", {})
        data_usage_history = complete_results.get("data_usage_history", {})
        iteration_results = complete_results.get("iteration_results", [])

        iterations = list(range(1, len(performance_history.get("val_f1", [])) + 1))

        if len(iterations) == 0:
            logger.warning("No data available for interactive dashboard")
            return ""

        # 1. 性能随时间变化
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=performance_history.get("val_f1", []),
                mode="lines+markers",
                name="Validation F1",
                line=dict(color="blue", width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )

        if "train_f1" in performance_history:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=performance_history["train_f1"],
                    mode="lines+markers",
                    name="Training F1",
                    line=dict(color="green", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )

        # 2. 数据使用增长
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=data_usage_history.get("training_samples", []),
                mode="lines+markers",
                name="Total Training",
                line=dict(color="red", width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=2,
        )

        if "pseudo_labels" in data_usage_history:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=data_usage_history["pseudo_labels"],
                    mode="lines+markers",
                    name="Pseudo Labels",
                    line=dict(color="orange", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=2,
            )

        # 3. 训练时间分析
        if iteration_results:
            training_times = [result.get("training_time", 0) for result in iteration_results]
            fig.add_trace(
                go.Bar(x=iterations, y=training_times, name="Training Time", marker_color="lightblue"), row=2, col=1
            )

        # 4. 不确定性分布（最新迭代）
        if iteration_results:
            latest_uncertainty = iteration_results[-1].get("uncertainty_results", {})
            uncertainty_scores = latest_uncertainty.get("uncertainty_scores", [])

            if len(uncertainty_scores) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=uncertainty_scores, name="Uncertainty Distribution", marker_color="skyblue", opacity=0.7
                    ),
                    row=2,
                    col=2,
                )

        # 5. 数据组成饼图
        if iteration_results:
            latest_result = iteration_results[-1]
            total_samples = latest_result.get("total_training_samples", 0)
            pseudo_count = latest_result.get("pseudo_label_count", 0)
            new_annotations = latest_result.get("new_annotations_count", 0)
            original_count = total_samples - pseudo_count - new_annotations

            fig.add_trace(
                go.Pie(
                    values=[original_count, pseudo_count, new_annotations],
                    labels=["Original", "Pseudo Labels", "New Annotations"],
                    name="Data Composition",
                ),
                row=3,
                col=1,
            )

        # 6. 性能相关性分析
        if len(iterations) > 1:
            sample_counts = data_usage_history.get("training_samples", [])
            f1_scores = performance_history.get("val_f1", [])

            fig.add_trace(
                go.Scatter(
                    x=sample_counts,
                    y=f1_scores,
                    mode="markers+lines",
                    name="Performance vs Data",
                    marker=dict(
                        size=10,
                        color=iterations,
                        colorscale="viridis",
                        showscale=True,
                        colorbar=dict(title="Iteration"),
                    ),
                    line=dict(color="gray", width=1, dash="dash"),
                ),
                row=3,
                col=2,
            )

        # 更新布局
        fig.update_layout(
            title_text=f"Active Learning Interactive Dashboard - {experiment_name}",
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=1200,
            template="plotly_white",
        )

        # 保存交互式图表
        output_path = self.viz_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        logger.info(f"💾 Interactive dashboard saved: {output_path}")
        return str(output_path)

    def create_comparison_report(self, multiple_results: Dict[str, Dict], baseline_name: str = "baseline") -> str:
        """
        创建多实验对比报告

        Args:
            multiple_results: 多个实验结果字典
            baseline_name: 基线实验名称

        Returns:
            保存的图片路径
        """
        logger.info("⚖️ Creating comparison report...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Active Learning Methods Comparison", fontsize=16, fontweight="bold")

        experiment_names = list(multiple_results.keys())
        n_experiments = len(experiment_names)

        if n_experiments < 2:
            logger.warning("Need at least 2 experiments for comparison")
            return ""

        # 提取对比数据
        final_performances = []
        total_training_times = []
        data_efficiency_scores = []
        convergence_iterations = []

        for name, result in multiple_results.items():
            # 最终性能
            performance_history = result.get("performance_history", {})
            val_f1_history = performance_history.get("val_f1", [])
            if val_f1_history:
                final_performances.append(val_f1_history[-1])
            else:
                final_performances.append(0)

            # 总训练时间
            total_time = result.get("total_training_time", 0)
            total_training_times.append(total_time)

            # 数据效率（性能提升/数据增长比例）
            data_usage = result.get("data_usage_history", {})
            training_samples = data_usage.get("training_samples", [])
            if len(val_f1_history) > 1 and len(training_samples) > 1:
                perf_improvement = val_f1_history[-1] - val_f1_history[0]
                data_growth_ratio = training_samples[-1] / training_samples[0]
                efficiency = perf_improvement / (data_growth_ratio - 1) if data_growth_ratio > 1 else 0
                data_efficiency_scores.append(efficiency)
            else:
                data_efficiency_scores.append(0)

            # 收敛迭代次数
            convergence_iter = result.get("convergence_iteration", len(val_f1_history))
            convergence_iterations.append(convergence_iter)

        # 1. 最终性能对比
        colors = plt.cm.Set3(np.linspace(0, 1, n_experiments))
        bars1 = axes[0, 0].bar(
            experiment_names, final_performances, color=colors, alpha=0.8, edgecolor="black", linewidth=1
        )
        axes[0, 0].set_title("Final Performance Comparison", fontweight="bold")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 添加数值标签
        for bar, perf in zip(bars1, final_performances):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{perf:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. 训练时间对比
        bars2 = axes[0, 1].bar(
            experiment_names, total_training_times, color=colors, alpha=0.8, edgecolor="black", linewidth=1
        )
        axes[0, 1].set_title("Training Time Comparison", fontweight="bold")
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 数据效率对比
        bars3 = axes[1, 0].bar(
            experiment_names, data_efficiency_scores, color=colors, alpha=0.8, edgecolor="black", linewidth=1
        )
        axes[1, 0].set_title("Data Efficiency Comparison", fontweight="bold")
        axes[1, 0].set_ylabel("Efficiency Score")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 收敛速度对比
        bars4 = axes[1, 1].bar(
            experiment_names, convergence_iterations, color=colors, alpha=0.8, edgecolor="black", linewidth=1
        )
        axes[1, 1].set_title("Convergence Speed Comparison", fontweight="bold")
        axes[1, 1].set_ylabel("Iterations to Convergence")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存对比图表
        output_path = self.viz_dir / "methods_comparison.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        # 创建对比表格
        comparison_df = pd.DataFrame(
            {
                "Experiment": experiment_names,
                "Final F1": final_performances,
                "Training Time (s)": total_training_times,
                "Data Efficiency": data_efficiency_scores,
                "Convergence Iterations": convergence_iterations,
            }
        )

        # 计算相对于基线的改进
        if baseline_name in experiment_names:
            baseline_idx = experiment_names.index(baseline_name)
            baseline_f1 = final_performances[baseline_idx]
            comparison_df["F1 Improvement"] = comparison_df["Final F1"] - baseline_f1
            comparison_df["F1 Improvement %"] = (comparison_df["F1 Improvement"] / baseline_f1 * 100).round(2)

        # 保存表格
        table_path = self.viz_dir / "methods_comparison.csv"
        comparison_df.to_csv(table_path, index=False)

        logger.info(f"💾 Comparison report saved: {output_path}")
        logger.info(f"📊 Comparison table saved: {table_path}")

        return str(output_path)

    def create_summary_report(self, complete_results: Dict, experiment_name: str) -> str:
        """
        创建实验总结报告

        Args:
            complete_results: 完整实验结果
            experiment_name: 实验名称

        Returns:
            保存的报告路径
        """
        logger.info("📋 Creating summary report...")

        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Active Learning Experiment Report - {experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background-color: #e8f4fd; border-radius: 5px; min-width: 120px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c5aa0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .improvement {{ color: green; font-weight: bold; }}
                .decline {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
        """

        # 报告头部
        html_content += f"""
        <div class="header">
            <h1>🎯 Active Learning Experiment Report</h1>
            <h2>{experiment_name}</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

        # 关键指标
        performance_history = complete_results.get("performance_history", {})
        data_usage_history = complete_results.get("data_usage_history", {})

        if performance_history.get("val_f1"):
            initial_f1 = performance_history["val_f1"][0]
            final_f1 = performance_history["val_f1"][-1]
            improvement = final_f1 - initial_f1

            html_content += f"""
            <div class="section">
                <h3>📊 Key Performance Metrics</h3>
                <div class="metric">
                    <div class="metric-value">{final_f1:.4f}</div>
                    <div class="metric-label">Final F1 Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'improvement' if improvement > 0 else 'decline'}">
                        {improvement:+.4f}
                    </div>
                    <div class="metric-label">Performance Improvement</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{complete_results.get('convergence_iteration', 0)}</div>
                    <div class="metric-label">Iterations to Convergence</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{complete_results.get('total_training_time', 0):.0f}s</div>
                    <div class="metric-label">Total Training Time</div>
                </div>
            </div>
            """

        # 数据使用统计
        if data_usage_history.get("training_samples"):
            initial_samples = data_usage_history["training_samples"][0]
            final_samples = data_usage_history["training_samples"][-1]
            data_growth = final_samples - initial_samples

            html_content += f"""
            <div class="section">
                <h3>📈 Data Usage Analysis</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Initial Training Samples</td><td>{initial_samples:,}</td><td>Original labeled data</td></tr>
                    <tr><td>Final Training Samples</td><td>{final_samples:,}</td><td>Total after augmentation</td></tr>
                    <tr><td>Data Growth</td><td class="improvement">+{data_growth:,}</td><td>Additional samples added</td></tr>
                    <tr><td>Pseudo Labels</td><td>{data_usage_history.get('pseudo_labels', [0])[-1]:,}</td><td>High-confidence predictions</td></tr>
                    <tr><td>New Annotations</td><td>{data_usage_history.get('new_annotations', [0])[-1]:,}</td><td>Human-labeled samples</td></tr>
                </table>
            </div>
            """

        # 迭代详情
        iteration_results = complete_results.get("iteration_results", [])
        if iteration_results:
            html_content += """
            <div class="section">
                <h3>🔄 Iteration Details</h3>
                <table>
                    <tr><th>Iteration</th><th>Val F1</th><th>Training Samples</th><th>Pseudo Labels</th><th>Training Time</th></tr>
            """

            for i, result in enumerate(iteration_results):
                val_f1 = (
                    performance_history.get("val_f1", [0] * (i + 1))[i]
                    if i < len(performance_history.get("val_f1", []))
                    else 0
                )
                html_content += f"""
                <tr>
                    <td>{result.get('iteration', i+1)}</td>
                    <td>{val_f1:.4f}</td>
                    <td>{result.get('total_training_samples', 0):,}</td>
                    <td>{result.get('pseudo_label_count', 0):,}</td>
                    <td>{result.get('training_time', 0):.1f}s</td>
                </tr>
                """

            html_content += "</table></div>"

        # 结论和建议
        html_content += f"""
        <div class="section">
            <h3>💡 Conclusions and Recommendations</h3>
            <ul>
                <li><strong>Performance:</strong> {'Successful improvement' if improvement > 0 else 'Limited improvement'} 
                    achieved through active learning strategy</li>
                <li><strong>Data Efficiency:</strong> Added {data_growth:,} samples with 
                    {data_usage_history.get('new_annotations', [0])[-1]:,} requiring human annotation</li>
                <li><strong>Convergence:</strong> Model converged after {complete_results.get('convergence_iteration', 0)} iterations</li>
                <li><strong>Scalability:</strong> Method {'scales well' if complete_results.get('convergence_iteration', 0) <= 5 else 'may need optimization'} 
                    for production use</li>
            </ul>
        </div>
        """

        html_content += """
        </body>
        </html>
        """

        # 保存HTML报告
        report_path = self.viz_dir / f"{experiment_name}_summary_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"📋 Summary report saved: {report_path}")
        return str(report_path)


def create_visualizer(output_dir: Path, config: Dict = None) -> ActiveLearningVisualizer:
    """
    工厂函数：创建可视化器

    Args:
        output_dir: 输出目录
        config: 可视化配置

    Returns:
        可视化器实例
    """
    viz_config = VisualizationConfig()
    if config:
        for key, value in config.items():
            if hasattr(viz_config, key):
                setattr(viz_config, key, value)

    return ActiveLearningVisualizer(output_dir, viz_config)
