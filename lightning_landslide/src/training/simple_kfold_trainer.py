# =============================================================================
# lightning_landslide/src/training/simple_kfold_trainer.py - 简化K折训练器
# =============================================================================

"""
简化的K折交叉验证训练器

设计哲学："简单即美"
这个训练器的设计原则是最大化重用现有的训练流程，
最小化新增的复杂性。

核心思想：
K折训练 = 标准训练 × N次 + 结果聚合
就像做N个蛋糕，每个蛋糕的制作流程完全相同，
只是原料（数据）分配不同，最后统计所有蛋糕的质量。
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
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.kfold_extension import create_kfold_wrapper
from ..utils.instantiate import instantiate_from_config

logger = logging.getLogger(__name__)


class SimpleKFoldTrainer:
    """
    简化的K折交叉验证训练器

    这个类的设计遵循"组合优于继承"的原则。
    它不是一个Lightning训练器的子类，而是一个协调器，
    负责管理多个标准训练过程。

    就像一个项目经理，不需要亲自做每一项具体工作，
    但需要协调各个部门（数据、模型、训练）完成整体目标。
    """

    def __init__(self, config: Dict[str, Any], experiment_name: str = None, output_dir: str = None):
        """
        初始化简化K折训练器

        Args:
            config: 完整的训练配置（包含model, data, trainer, kfold等）
            experiment_name: 实验名称
            output_dir: 实验输出目录的完整路径（与基础训练保持一致）
        """
        self.config = config
        self.kfold_config = config.get("kfold", {})

        # K折基本配置
        self.n_splits = self.kfold_config.get("n_splits", 5)
        self.stratified = self.kfold_config.get("stratified", True)
        self.primary_metric = self.kfold_config.get("primary_metric", "f1")
        self.save_oof = self.kfold_config.get("save_oof_predictions", True)
        self.save_models = self.kfold_config.get("save_fold_models", True)

        # 实验管理
        self.experiment_name = experiment_name or config.get("experiment_name", f"kfold_{int(time.time())}")

        # 使用传入的output_dir（完整路径），而不是再次嵌套experiment_name
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            # 如果没有提供output_dir，回退到默认路径（向后兼容）
            self.output_dir = Path("outputs") / self.experiment_name

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 结果存储
        self.fold_results = []
        self.oof_predictions = None
        self.oof_targets = None
        self.test_predictions = []

        # 全局种子
        self.seed = config.get("seed", 3407)
        pl.seed_everything(self.seed, workers=True)

        logger.info(f"🚀 SimpleKFoldTrainer initialized")
        logger.info(f"📁 Experiment: {self.experiment_name}")
        logger.info(f"🎯 {self.n_splits}-fold cross validation")
        logger.info(f"📊 Primary metric: {self.primary_metric}")

        # 创建输出目录结构
        self._setup_directories()

    def _setup_directories(self):
        """创建输出目录结构"""
        dirs = ["models", "predictions", "logs", "plots", "reports"]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(exist_ok=True)

        logger.info(f"📂 Output directories created in: {self.output_dir}")

    def run_kfold_training(self) -> Dict[str, Any]:
        """
        执行完整的K折交叉验证训练

        这是整个类的核心方法。它的逻辑非常直观：
        1. 准备数据分割
        2. 循环训练每一折
        3. 收集和分析结果
        4. 生成最终报告

        Returns:
            包含所有结果的字典
        """
        logger.info(f"🎯 Starting {self.n_splits}-fold cross validation")
        start_time = time.time()

        # 第一步：准备K折数据分割
        logger.info("📊 Step 1: Preparing K-fold data splits...")
        kfold_wrapper = self._create_kfold_wrapper()
        kfold_wrapper.prepare_fold_splits()

        # 第二步：训练每一折
        logger.info("🔄 Step 2: Training individual folds...")
        for fold in range(self.n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Training Fold {fold + 1}/{self.n_splits}")
            logger.info(f"{'='*60}")

            fold_result = self._train_single_fold(fold, kfold_wrapper)
            self.fold_results.append(fold_result)

            # 打印当前折结果
            val_score = fold_result["val_metrics"].get(f"val_{self.primary_metric}", 0)
            logger.info(f"✅ Fold {fold + 1} - {self.primary_metric}: {val_score:.4f}")

        # 第三步：生成OOF预测（如果启用）
        if self.save_oof:
            logger.info("🎯 Step 3: Generating OOF predictions...")
            self._generate_oof_predictions(kfold_wrapper)

        # 第四步：生成最终结果和报告
        logger.info("📊 Step 4: Generating final results...")
        final_results = self._generate_final_results(time.time() - start_time)

        # 第五步：保存结果
        logger.info("💾 Step 5: Saving results...")
        self._save_results(final_results)

        # 第六步：生成可视化报告
        logger.info("📈 Step 6: Generating visualization report...")
        self._generate_report()

        logger.info(f"\n🎉 K-Fold training completed!")
        logger.info(f"⏱️  Total time: {final_results['total_time']:.2f}s")
        logger.info(f"📈 Mean CV score: {final_results['mean_cv_score']:.4f} ± {final_results['std_cv_score']:.4f}")

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

    def _train_single_fold(self, fold_idx: int, kfold_wrapper) -> Dict[str, Any]:
        """
        训练单个折

        这个方法展示了设计的核心思想：重用现有训练流程。
        我们不需要重新发明训练逻辑，只需要：
        1. 获取当前折的数据模块
        2. 创建模型
        3. 配置回调和日志
        4. 执行标准训练

        Args:
            fold_idx: 折索引
            kfold_wrapper: K折数据包装器

        Returns:
            单折训练结果
        """
        # 设置折特定的随机种子
        fold_seed = self.seed + fold_idx
        pl.seed_everything(fold_seed, workers=True)

        # 获取当前折的数据模块
        datamodule = kfold_wrapper.get_fold_datamodule(fold_idx)

        # 创建模型（每折都是全新的模型）
        model = instantiate_from_config(self.config["model"])

        # 设置回调函数
        callbacks = self._create_fold_callbacks(fold_idx)

        # 设置日志记录器
        logger_instance = TensorBoardLogger(
            save_dir=str(self.output_dir / "logs"),
            name=f"fold_{fold_idx}",
            version="",
        )

        # 创建训练器 - 模仿标准训练的正确做法
        trainer_config = self.config["trainer"]["params"].copy()
        trainer = pl.Trainer(**trainer_config)

        # 然后设置复杂对象（就像标准训练中的做法）
        if callbacks:
            trainer.callbacks = callbacks
        if logger_instance:
            trainer.logger = logger_instance

        # 执行训练 - 这里就是标准的Lightning训练流程
        trainer.fit(model, datamodule)

        # 执行验证获取最终指标
        val_results = trainer.validate(model, datamodule, verbose=False)
        val_metrics = val_results[0] if val_results else {}

        # 保存模型（如果启用）
        model_path = None
        if self.save_models:
            model_path = self.output_dir / "models" / f"fold_{fold_idx}_model.ckpt"
            trainer.save_checkpoint(str(model_path))

        return {
            "fold": fold_idx,
            "val_metrics": val_metrics,
            "model_path": str(model_path) if model_path else None,
            "trainer": trainer,
            "model": model,
        }

    def _create_fold_callbacks(self, fold_idx: int) -> List[pl.Callback]:
        """
        为单折创建回调函数

        这个方法展示了如何重用配置中的回调设置，
        但为每一折进行个性化配置。
        """
        callbacks = []

        # 模型检查点回调
        if "callbacks" in self.config and "model_checkpoint" in self.config["callbacks"]:
            checkpoint_config = self.config["callbacks"]["model_checkpoint"]

            # 从配置中读取所有参数，而不是硬编码
            monitor_metric = checkpoint_config.get("monitor", f"val_{self.primary_metric}")
            mode = checkpoint_config.get("mode", "max")
            save_top_k = checkpoint_config.get("save_top_k", 1)
            save_last = checkpoint_config.get("save_last", True)
            verbose = checkpoint_config.get("verbose", True)

            # 关键修复：正确读取min_delta参数
            min_delta = checkpoint_config.get("min_delta", 0.0)  # 默认值应该和Lightning一致

            checkpoint_callback = ModelCheckpoint(
                dirpath=str(self.output_dir / "models"),
                filename=f"fold_{fold_idx}_best_{{epoch:02d}}_{{val_{self.primary_metric}:.4f}}",
                monitor=monitor_metric,
                mode=mode,
                save_top_k=save_top_k,
                save_last=save_last,
                verbose=verbose,
                min_delta=min_delta,  # 使用配置文件中的值
            )
            callbacks.append(checkpoint_callback)

        # 早停回调
        if "callbacks" in self.config and "early_stopping" in self.config["callbacks"]:
            early_stop_config = self.config["callbacks"]["early_stopping"]

            # 从配置中读取所有参数
            monitor_metric = early_stop_config.get("monitor", f"val_{self.primary_metric}")
            mode = early_stop_config.get("mode", "max")
            patience = early_stop_config.get("patience", 10)
            verbose = early_stop_config.get("verbose", True)

            # 关键修复：同样需要读取min_delta参数
            min_delta = early_stop_config.get("min_delta", 0.0)

            early_stopping = EarlyStopping(
                monitor=monitor_metric,
                mode=mode,
                patience=patience,
                verbose=verbose,
                min_delta=min_delta,  # 使用配置文件中的值
            )
            callbacks.append(early_stopping)

        return callbacks

    def _generate_oof_predictions(self, kfold_wrapper):
        """
        生成OOF（Out-of-Fold）预测

        OOF预测是k折交叉验证的精髓之一。
        它的思想是：对于每个样本，我们使用没有见过它的模型来预测。
        这样得到的预测结果更能反映模型的真实泛化能力。

        就像考试一样，我们用学生没有见过的题目来测试他们的真实水平。
        """
        logger.info("🎯 Generating OOF predictions...")

        # 准备OOF数组
        # 我们需要知道总共有多少训练样本
        temp_datamodule = kfold_wrapper.get_fold_datamodule(0)

        # 从折分割信息中获取总样本数
        total_samples = sum(len(train_idx) + len(val_idx) for train_idx, val_idx in kfold_wrapper.fold_splits)

        self.oof_predictions = np.zeros(total_samples)
        self.oof_targets = np.zeros(total_samples)

        # 为每一折生成OOF预测
        for fold_idx in range(self.n_splits):
            logger.info(f"Generating OOF predictions for fold {fold_idx}...")

            # 获取验证集索引
            _, val_indices = kfold_wrapper.fold_splits[fold_idx]

            # 获取对应的模型和数据
            fold_result = self.fold_results[fold_idx]
            model = fold_result["model"]
            trainer = fold_result["trainer"]

            # 获取验证数据
            datamodule = kfold_wrapper.get_fold_datamodule(fold_idx)

            # 生成预测
            model.eval()
            predictions = []
            targets = []

            with torch.no_grad():
                for batch in datamodule.val_dataloader():
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        x, y = batch
                        targets.extend(y.cpu().numpy())
                    else:
                        x = batch

                    # 生成预测
                    logits = model(x.to(model.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    predictions.extend(probs)

            # 存储OOF预测
            self.oof_predictions[val_indices] = predictions
            if targets:  # 如果有标签
                self.oof_targets[val_indices] = targets

        logger.info("✅ OOF predictions generated!")

    def _generate_final_results(self, total_time: float) -> Dict[str, Any]:
        """生成最终结果"""
        # 收集所有折的验证指标
        cv_scores = []
        all_metrics = {}

        for fold_result in self.fold_results:
            val_metrics = fold_result["val_metrics"]
            fold_score = val_metrics.get(f"val_{self.primary_metric}", 0)
            cv_scores.append(fold_score)

            # 收集所有指标
            for metric_name, value in val_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # 计算统计信息
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        # 计算每个指标的统计信息
        metrics_stats = {}
        for metric_name, values in all_metrics.items():
            metrics_stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

        # 组装最终结果
        final_results = {
            "experiment_name": self.experiment_name,
            "n_splits": self.n_splits,
            "primary_metric": self.primary_metric,
            "mean_cv_score": float(mean_cv_score),
            "std_cv_score": float(std_cv_score),
            "cv_scores": cv_scores,
            "metrics_stats": metrics_stats,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

        return final_results

    def _save_results(self, results: Dict[str, Any]):
        """保存结果到JSON文件"""
        results_file = self.output_dir / "kfold_results.json"

        # 创建一个可序列化的结果副本
        serializable_results = results.copy()
        # 移除不能序列化的配置对象
        if "config" in serializable_results:
            serializable_results["config"] = str(serializable_results["config"])

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"💾 Results saved to: {results_file}")

    def _generate_report(self):
        """生成可视化报告"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. CV分数分布
            cv_scores = [result["val_metrics"].get(f"val_{self.primary_metric}", 0) for result in self.fold_results]

            axes[0, 0].bar(range(1, self.n_splits + 1), cv_scores)
            axes[0, 0].set_title(f"{self.primary_metric} Score by Fold")
            axes[0, 0].set_xlabel("Fold")
            axes[0, 0].set_ylabel(f"{self.primary_metric}")
            axes[0, 0].grid(True, alpha=0.3)

            # 2. CV分数箱线图
            axes[0, 1].boxplot([cv_scores])
            axes[0, 1].set_title(f"{self.primary_metric} Distribution")
            axes[0, 1].set_ylabel(f"{self.primary_metric}")
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 如果有OOF预测，绘制ROC曲线
            if self.oof_predictions is not None and self.oof_targets is not None:
                from sklearn.metrics import roc_curve, auc

                fpr, tpr, _ = roc_curve(self.oof_targets, self.oof_predictions)
                roc_auc = auc(fpr, tpr)

                axes[1, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
                axes[1, 0].plot([0, 1], [0, 1], "k--")
                axes[1, 0].set_xlabel("False Positive Rate")
                axes[1, 0].set_ylabel("True Positive Rate")
                axes[1, 0].set_title("OOF ROC Curve")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(
                    0.5, 0.5, "OOF predictions not available", ha="center", va="center", transform=axes[1, 0].transAxes
                )

            # 4. 训练时间对比（如果可获得）
            axes[1, 1].text(
                0.5,
                0.5,
                f"Mean CV Score:\n{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
            )
            axes[1, 1].set_title("Summary Statistics")

            plt.tight_layout()

            # 保存图表
            plot_file = self.output_dir / "plots" / "kfold_report.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"📊 Visualization report saved to: {plot_file}")

        except Exception as e:
            logger.warning(f"⚠️  Could not generate visualization report: {e}")


# 便利函数
def run_kfold_training(config: Dict[str, Any], experiment_name: str = None) -> Dict[str, Any]:
    """
    运行K折训练的便利函数

    这个函数为用户提供了一个简单的接口来执行K折训练。
    就像一个遥控器，用户只需要按一个按钮就能启动整个复杂的训练过程。

    Args:
        config: 训练配置
        experiment_name: 实验名称

    Returns:
        训练结果
    """
    trainer = SimpleKFoldTrainer(config, experiment_name)
    return trainer.run_kfold_training()


if __name__ == "__main__":
    print("✅ SimpleKFoldTrainer ready for Kaggle-level cross validation!")
    print("🚀 Use run_kfold_training(config) to start training!")
