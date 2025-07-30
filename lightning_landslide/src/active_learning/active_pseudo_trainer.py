# =============================================================================
# lightning_landslide/src/active_learning/active_pseudo_trainer.py
# =============================================================================

"""
主动学习+伪标签融合训练器

这是整个主动学习系统的核心协调器，它整合了：
1. 不确定性估计
2. 伪标签生成
3. 主动学习样本选择
4. 迭代训练流程
5. 数据管理

设计理念："渐进增强" - 通过多轮迭代，逐步改善模型性能，
最大化利用无标注数据，最小化人工标注成本。
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import json
import time
from datetime import datetime
import copy
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ..utils.instantiate import instantiate_from_config
from ..data.multimodal_datamodule import MultiModalDataModule
from ..data.multimodal_dataset import MultiModalDataset
from .uncertainty_estimator import create_uncertainty_estimator, UncertaintyResults
from .pseudo_label_generator import create_pseudo_label_generator, PseudoLabelResults
from .active_learning_selector import create_active_learning_selector, ActiveLearningResults

logger = logging.getLogger(__name__)


@dataclass
class IterationResults:
    """单次迭代的结果"""

    iteration: int
    model_performance: Dict[str, float]
    uncertainty_results: UncertaintyResults
    pseudo_label_results: PseudoLabelResults
    active_learning_results: ActiveLearningResults
    training_time: float
    total_training_samples: int
    pseudo_label_count: int
    new_annotations_count: int


@dataclass
class ActivePseudoTrainingResults:
    """完整的主动+伪标签训练结果"""

    iteration_results: List[IterationResults]
    final_model_path: str
    best_model_path: str
    performance_history: Dict[str, List[float]]
    data_usage_history: Dict[str, List[int]]
    total_training_time: float
    convergence_iteration: int

    def to_dict(self):
        """转换为字典格式"""
        return {
            "iteration_results": [asdict(r) for r in self.iteration_results],
            "final_model_path": self.final_model_path,
            "best_model_path": self.best_model_path,
            "performance_history": self.performance_history,
            "data_usage_history": self.data_usage_history,
            "total_training_time": self.total_training_time,
            "convergence_iteration": self.convergence_iteration,
        }


class EnhancedDataManager:
    """
    增强的数据管理器

    管理主动学习过程中的复杂数据状态：
    - 原始标注数据
    - 伪标签数据
    - 新增标注数据
    - 测试集数据
    """

    def __init__(self, base_config: Dict, output_dir: Path):
        self.base_config = base_config
        self.output_dir = output_dir

        # 数据状态管理
        self.original_train_data = None
        self.pseudo_labeled_data = []
        self.newly_annotated_data = []
        self.test_data = None

        # 创建数据版本控制目录
        self.data_versions_dir = output_dir / "data_versions"
        self.data_versions_dir.mkdir(parents=True, exist_ok=True)

        logger.info("📊 EnhancedDataManager initialized")

    def load_initial_data(self):
        """加载初始数据"""
        logger.info("📂 Loading initial data...")

        # 使用配置创建初始数据模块
        self.base_datamodule = instantiate_from_config(self.base_config["data"])
        self.base_datamodule.setup("fit")

        # 保存原始数据引用
        self.original_train_data = self.base_datamodule.train_dataset

        # 获取类别分布
        self.original_class_distribution = self._get_class_distribution(self.original_train_data)

        logger.info(f"✅ Loaded {len(self.original_train_data)} original training samples")
        logger.info(f"📊 Class distribution: {self.original_class_distribution}")

    def create_enhanced_datamodule(self, iteration: int) -> MultiModalDataModule:
        """创建包含伪标签的增强数据模块"""
        logger.info(f"🔧 Creating enhanced datamodule for iteration {iteration}...")

        # 合并所有训练数据
        combined_dataset = self._combine_training_data()

        # 创建新的数据模块
        enhanced_config = copy.deepcopy(self.base_config["data"])
        enhanced_datamodule = instantiate_from_config(enhanced_config)

        # 替换训练数据集
        enhanced_datamodule.train_dataset = combined_dataset
        enhanced_datamodule.setup("fit")  # 重新设置验证集分割

        # 保存数据版本
        self._save_data_version(combined_dataset, iteration)

        logger.info(f"📈 Enhanced dataset size: {len(combined_dataset)} samples")
        return enhanced_datamodule

    def add_pseudo_labels(self, pseudo_label_results: PseudoLabelResults, test_dataloader):
        """添加伪标签数据"""
        logger.info(f"🏷️ Adding {len(pseudo_label_results.high_confidence_samples)} pseudo labels...")

        # 创建伪标签数据集
        pseudo_dataset = self._create_pseudo_dataset(pseudo_label_results, test_dataloader)
        self.pseudo_labeled_data.append(pseudo_dataset)

        logger.info(f"✅ Added pseudo-labeled dataset with {len(pseudo_dataset)} samples")

    def add_new_annotations(self, annotated_samples: List[str], test_dataloader):
        """添加新标注的数据"""
        if not annotated_samples:
            return

        logger.info(f"📝 Adding {len(annotated_samples)} newly annotated samples...")

        # 这里假设有一个人工标注接口
        # 在实际实现中，可能需要与标注工具集成
        annotated_dataset = self._create_annotated_dataset(annotated_samples, test_dataloader)
        self.newly_annotated_data.append(annotated_dataset)

        logger.info(f"✅ Added {len(annotated_dataset)} newly annotated samples")

    def get_data_statistics(self) -> Dict:
        """获取当前数据统计信息"""
        original_count = len(self.original_train_data) if self.original_train_data else 0
        pseudo_count = sum(len(dataset) for dataset in self.pseudo_labeled_data)
        new_annotation_count = sum(len(dataset) for dataset in self.newly_annotated_data)

        return {
            "original_training_samples": original_count,
            "pseudo_labeled_samples": pseudo_count,
            "newly_annotated_samples": new_annotation_count,
            "total_training_samples": original_count + pseudo_count + new_annotation_count,
            "data_augmentation_ratio": (pseudo_count + new_annotation_count) / max(original_count, 1),
        }

    def _combine_training_data(self):
        """合并所有训练数据"""
        combined_data = []

        # 添加原始数据
        if self.original_train_data:
            combined_data.extend(self._dataset_to_list(self.original_train_data))

        # 添加伪标签数据
        for pseudo_dataset in self.pseudo_labeled_data:
            combined_data.extend(self._dataset_to_list(pseudo_dataset))

        # 添加新标注数据
        for new_dataset in self.newly_annotated_data:
            combined_data.extend(self._dataset_to_list(new_dataset))

        # 创建新的数据集
        return self._create_combined_dataset(combined_data)

    def _get_class_distribution(self, dataset) -> Dict[int, int]:
        """获取数据集的类别分布"""
        class_counts = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_int = int(label.item())
            class_counts[label_int] = class_counts.get(label_int, 0) + 1
        return class_counts

    def _create_pseudo_dataset(self, pseudo_results: PseudoLabelResults, test_dataloader):
        """从伪标签结果创建数据集"""
        # 简化实现 - 实际中需要根据具体的数据集类型来实现
        pseudo_samples = []

        # 这里需要根据sample_id从test_dataloader中提取对应的数据
        # 并用伪标签替换原始标签
        # 具体实现取决于数据集的结构

        return pseudo_samples  # 返回伪标签数据集

    def _dataset_to_list(self, dataset):
        """将数据集转换为列表"""
        return [dataset[i] for i in range(len(dataset))]

    def _create_combined_dataset(self, data_list):
        """从数据列表创建组合数据集"""
        # 简化实现 - 需要根据具体数据集类型实现
        return data_list

    def _save_data_version(self, dataset, iteration: int):
        """保存数据版本"""
        version_info = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(dataset),
            "class_distribution": self._get_class_distribution(dataset),
        }

        with open(self.data_versions_dir / f"iteration_{iteration}_info.json", "w") as f:
            json.dump(version_info, f, indent=2)


class ActivePseudoTrainer:
    """
    主动学习+伪标签融合训练器

    这是整个系统的核心，协调所有组件完成迭代训练过程。
    """

    def __init__(self, config: Dict, experiment_name: str = None, output_dir: str = None):
        """
        初始化主动伪标签训练器

        Args:
            config: 完整配置（包含model, data, trainer, active_pseudo_learning等）
            experiment_name: 实验名称
            output_dir: 输出目录
        """
        self.config = config
        self.active_config = config.get("active_pseudo_learning", {})

        # 实验配置
        self.experiment_name = experiment_name or config.get("experiment_name", f"active_pseudo_{int(time.time())}")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练配置
        self.max_iterations = self.active_config.get("max_iterations", 5)
        self.convergence_threshold = self.active_config.get("convergence_threshold", 0.01)
        self.min_improvement_iterations = self.active_config.get("min_improvement_iterations", 2)

        # 创建组件
        self._initialize_components()

        # 数据管理器
        self.data_manager = EnhancedDataManager(config, self.output_dir)

        # 结果跟踪
        self.iteration_results = []
        self.performance_history = {"val_f1": [], "val_loss": [], "train_f1": []}
        self.data_usage_history = {"training_samples": [], "pseudo_labels": [], "new_annotations": []}

        logger.info(f"🚀 ActivePseudoTrainer initialized: {self.experiment_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔄 Max iterations: {self.max_iterations}")

    def _initialize_components(self):
        """初始化各个组件"""
        # 不确定性估计器
        uncertainty_config = self.active_config.get("uncertainty_estimation", {})
        self.uncertainty_estimator = create_uncertainty_estimator(
            method=uncertainty_config.get("method", "mc_dropout"), **uncertainty_config.get("params", {})
        )

        # 伪标签生成器
        pseudo_config = self.active_config.get("pseudo_labeling", {})
        self.pseudo_generator = create_pseudo_label_generator(pseudo_config)

        # 主动学习选择器
        active_config = self.active_config.get("active_learning", {})
        self.active_selector = create_active_learning_selector(active_config)

        logger.info("🔧 All components initialized successfully")

    def run(self) -> ActivePseudoTrainingResults:
        """运行完整的主动+伪标签学习流程"""
        logger.info("🎯 Starting Active + Pseudo Label Learning...")
        start_time = time.time()

        # 初始化数据
        self.data_manager.load_initial_data()
        self.pseudo_generator.set_class_distribution(self.data_manager.original_class_distribution)

        # 训练基线模型
        logger.info("🏁 Training baseline model...")
        baseline_model = self._train_baseline_model()
        current_model = baseline_model
        best_performance = 0.0
        best_model_path = None
        no_improvement_count = 0

        # 迭代训练循环
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ITERATION {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            iteration_start = time.time()

            # 1. 在测试集上进行不确定性估计
            logger.info("📊 Step 1: Uncertainty estimation on test set...")
            test_dataloader = self._create_test_dataloader()
            uncertainty_results = self.uncertainty_estimator.estimate_uncertainty(current_model, test_dataloader)

            # 2. 生成伪标签
            logger.info("🏷️ Step 2: Generating pseudo labels...")
            pseudo_results = self.pseudo_generator.generate_pseudo_labels(
                uncertainty_results=uncertainty_results,
                current_iteration=iteration,
                validation_performance=best_performance,
            )

            # 3. 主动学习样本选择
            logger.info("🎯 Step 3: Active learning sample selection...")
            active_results = self.active_selector.select_samples(
                uncertainty_results=uncertainty_results,
                candidate_samples=pseudo_results.low_confidence_samples,
                budget=self.active_config.get("annotation_budget", 50),
            )

            # 4. 数据增强（添加伪标签和新标注）
            logger.info("📈 Step 4: Data augmentation...")
            self.data_manager.add_pseudo_labels(pseudo_results, test_dataloader)

            # 模拟人工标注过程（实际中需要人工介入）
            selected_ids = [s.sample_id for s in active_results.selected_samples]
            self.data_manager.add_new_annotations(selected_ids, test_dataloader)

            # 5. 重新训练模型
            logger.info("🚀 Step 5: Retraining model with augmented data...")
            enhanced_datamodule = self.data_manager.create_enhanced_datamodule(iteration)
            new_model, performance = self._train_enhanced_model(enhanced_datamodule, iteration)

            # 6. 评估性能改进
            current_f1 = performance.get("val_f1", 0.0)
            improvement = current_f1 - best_performance

            logger.info(f"📈 Performance: {current_f1:.4f} (improvement: {improvement:+.4f})")

            # 更新最佳模型
            if current_f1 > best_performance:
                best_performance = current_f1
                best_model_path = self.output_dir / f"models/best_model_iter_{iteration}.ckpt"
                torch.save(new_model.state_dict(), best_model_path)
                no_improvement_count = 0
                logger.info(f"🏆 New best model saved: {best_model_path}")
            else:
                no_improvement_count += 1

            # 记录迭代结果
            iteration_time = time.time() - iteration_start
            data_stats = self.data_manager.get_data_statistics()

            iter_result = IterationResults(
                iteration=iteration + 1,
                model_performance=performance,
                uncertainty_results=uncertainty_results,
                pseudo_label_results=pseudo_results,
                active_learning_results=active_results,
                training_time=iteration_time,
                total_training_samples=data_stats["total_training_samples"],
                pseudo_label_count=data_stats["pseudo_labeled_samples"],
                new_annotations_count=data_stats["newly_annotated_samples"],
            )

            self.iteration_results.append(iter_result)

            # 更新历史记录
            self.performance_history["val_f1"].append(current_f1)
            self.performance_history["val_loss"].append(performance.get("val_loss", 0.0))
            self.performance_history["train_f1"].append(performance.get("train_f1", 0.0))

            self.data_usage_history["training_samples"].append(data_stats["total_training_samples"])
            self.data_usage_history["pseudo_labels"].append(data_stats["pseudo_labeled_samples"])
            self.data_usage_history["new_annotations"].append(data_stats["newly_annotated_samples"])

            # 保存中间结果
            self._save_iteration_results(iter_result, iteration)

            # 收敛检查
            if improvement < self.convergence_threshold and no_improvement_count >= self.min_improvement_iterations:
                logger.info(f"🎯 Convergence reached after {iteration + 1} iterations")
                break

            current_model = new_model

        # 训练完成
        total_time = time.time() - start_time

        # 创建最终结果
        final_results = ActivePseudoTrainingResults(
            iteration_results=self.iteration_results,
            final_model_path=str(self.output_dir / f"models/final_model.ckpt"),
            best_model_path=str(best_model_path) if best_model_path else "",
            performance_history=self.performance_history,
            data_usage_history=self.data_usage_history,
            total_training_time=total_time,
            convergence_iteration=len(self.iteration_results),
        )

        # 保存最终模型
        torch.save(current_model.state_dict(), final_results.final_model_path)

        # 保存完整结果和可视化
        self._save_final_results(final_results)
        self._create_training_visualization(final_results)

        logger.info(f"\n🎉 Active + Pseudo Label Learning completed!")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")
        logger.info(f"🏆 Best F1 score: {best_performance:.4f}")
        logger.info(f"📁 Results saved to: {self.output_dir}")

        return final_results

    def _train_baseline_model(self):
        """训练基线模型"""
        baseline_trainer = self._create_trainer("baseline")
        model = instantiate_from_config(self.config["model"])

        baseline_trainer.fit(model, self.data_manager.base_datamodule)

        # 保存基线模型
        baseline_path = self.output_dir / "models/baseline_model.ckpt"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_trainer.save_checkpoint(str(baseline_path))

        return model

    def _train_enhanced_model(self, datamodule, iteration: int):
        """训练增强数据的模型"""
        trainer = self._create_trainer(f"iteration_{iteration}")
        model = instantiate_from_config(self.config["model"])

        trainer.fit(model, datamodule)

        # 评估模型性能
        val_results = trainer.validate(model, datamodule, verbose=False)
        performance = val_results[0] if val_results else {}

        return model, performance

    def _create_trainer(self, name: str) -> pl.Trainer:
        """创建Lightning训练器"""
        trainer_config = self.config["trainer"]["params"].copy()

        # 设置日志和回调
        logger_instance = TensorBoardLogger(save_dir=str(self.output_dir / "logs"), name=name, version="")

        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.output_dir / "checkpoints" / name),
                filename="{epoch}-{val_f1:.4f}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_f1", patience=10, mode="max"),
        ]

        trainer = pl.Trainer(**trainer_config)
        trainer.callbacks = callbacks
        trainer.logger = logger_instance

        return trainer

    def _create_test_dataloader(self):
        """创建测试数据加载器"""
        # 使用基础数据模块的测试集
        self.data_manager.base_datamodule.setup("test")
        return self.data_manager.base_datamodule.test_dataloader()

    def _save_iteration_results(self, result: IterationResults, iteration: int):
        """保存单次迭代结果"""
        iter_dir = self.output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # 保存迭代摘要
        with open(iter_dir / "iteration_summary.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # 保存各个组件的详细结果
        self.pseudo_generator.save_results(result.pseudo_label_results, iter_dir / "pseudo_labels")
        self.active_selector.save_results(result.active_learning_results, iter_dir / "active_learning")

        # 创建可视化
        self.pseudo_generator.create_visualization(result.pseudo_label_results, iter_dir / "pseudo_labels")
        self.active_selector.create_visualization(result.active_learning_results, iter_dir / "active_learning")

    def _save_final_results(self, results: ActivePseudoTrainingResults):
        """保存最终完整结果"""
        with open(self.output_dir / "final_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        # 保存性能历史CSV
        history_df = pd.DataFrame(results.performance_history)
        history_df.to_csv(self.output_dir / "performance_history.csv", index=False)

        # 保存数据使用历史CSV
        data_usage_df = pd.DataFrame(results.data_usage_history)
        data_usage_df.to_csv(self.output_dir / "data_usage_history.csv", index=False)

    def _create_training_visualization(self, results: ActivePseudoTrainingResults):
        """创建训练过程的可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        iterations = list(range(1, len(results.performance_history["val_f1"]) + 1))

        # 1. 性能改进曲线
        axes[0, 0].plot(iterations, results.performance_history["val_f1"], "b-o", label="Validation F1")
        axes[0, 0].plot(iterations, results.performance_history["train_f1"], "g-s", label="Training F1")
        axes[0, 0].set_title("Model Performance Over Iterations")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 数据增长趋势
        axes[0, 1].plot(iterations, results.data_usage_history["training_samples"], "r-o", label="Total Training")
        axes[0, 1].plot(
            iterations, results.data_usage_history["pseudo_labels"], "orange", marker="s", label="Pseudo Labels"
        )
        axes[0, 1].plot(
            iterations, results.data_usage_history["new_annotations"], "purple", marker="^", label="New Annotations"
        )
        axes[0, 1].set_title("Data Usage Over Iterations")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Number of Samples")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 损失曲线
        axes[1, 0].plot(iterations, results.performance_history["val_loss"], "r-o", label="Validation Loss")
        axes[1, 0].set_title("Validation Loss Over Iterations")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 数据增强效率
        original_samples = (
            results.data_usage_history["training_samples"][0] if results.data_usage_history["training_samples"] else 0
        )
        augmentation_ratios = [
            (total - original_samples) / original_samples * 100
            for total in results.data_usage_history["training_samples"]
        ]

        axes[1, 1].bar(iterations, augmentation_ratios, color="skyblue", alpha=0.7)
        axes[1, 1].set_title("Data Augmentation Ratio")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Augmentation %")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Training visualization saved to {self.output_dir / 'training_overview.png'}")


def create_active_pseudo_trainer(config: Dict, **kwargs) -> ActivePseudoTrainer:
    """工厂函数：创建主动伪标签训练器"""
    return ActivePseudoTrainer(config, **kwargs)
