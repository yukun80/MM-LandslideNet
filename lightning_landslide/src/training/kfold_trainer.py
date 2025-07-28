# =============================================================================
# lightning_landslide/src/training/kfold_trainer.py - K折交叉验证训练器
# =============================================================================

"""
K折交叉验证训练器 - Kaggle竞赛级别的模型训练管理

这个模块提供了完整的K折交叉验证训练流程，包括：
1. 自动化的K折训练循环
2. OOF(Out-of-Fold)预测生成和管理
3. 测试集预测集成
4. 模型性能分析和报告
5. 模型保存和版本管理
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

warnings.filterwarnings("ignore")

from ..data.kfold_datamodule import KFoldDataModule, create_kfold_datamodule
from ..utils.instantiate import instantiate_from_config

logger = logging.getLogger(__name__)


class KFoldTrainer:
    """
    K折交叉验证训练器

    作为Kaggle竞赛大师的核心武器，这个类提供了：
    1. 全自动的K折训练流程
    2. 严格的性能监控和早停
    3. OOF预测的生成和验证
    4. 测试集预测的集成
    5. 详细的性能分析报告
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        trainer_config: Dict[str, Any],
        # K折配置
        n_splits: int = 5,
        stratified: bool = True,
        # 输出配置
        output_dir: str = "outputs/kfold_experiments",
        experiment_name: str = None,
        # 性能配置
        primary_metric: str = "f1",
        early_stopping_patience: int = 10,
        # 其他配置
        seed: int = 3407,
        save_predictions: bool = True,
        save_models: bool = True,
        generate_oof: bool = True,
        **kwargs,
    ):
        """
        初始化K折训练器

        Args:
            model_config: 模型配置
            data_config: 数据配置
            trainer_config: 训练器配置
            n_splits: K折数量
            experiment_name: 实验名称
            primary_metric: 主要评估指标
        """
        self.model_config = model_config
        self.data_config = data_config
        self.trainer_config = trainer_config

        # K折配置
        self.n_splits = n_splits
        self.stratified = stratified

        # 输出配置
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"kfold_experiment_{int(time.time())}"
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 性能配置
        self.primary_metric = primary_metric
        self.early_stopping_patience = early_stopping_patience

        # 其他配置
        self.seed = seed
        self.save_predictions = save_predictions
        self.save_models = save_models
        self.generate_oof = generate_oof

        # 结果存储
        self.fold_results = []
        self.oof_predictions = None
        self.test_predictions = []
        self.fold_models = []

        # 创建输出目录结构
        self._setup_directories()

        logger.info(f"🚀 KFoldTrainer initialized for {n_splits}-fold CV")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Output directory: {self.experiment_dir}")

    def _setup_directories(self) -> None:
        """设置输出目录结构"""
        dirs_to_create = [
            self.experiment_dir / "models",
            self.experiment_dir / "predictions",
            self.experiment_dir / "logs",
            self.experiment_dir / "plots",
            self.experiment_dir / "reports",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def train_kfold(self) -> Dict[str, Any]:
        """
        执行完整的K折交叉验证训练

        Returns:
            包含所有折结果的字典
        """
        logger.info(f"🎯 Starting {self.n_splits}-fold cross validation training")

        # 准备OOF预测数组
        if self.generate_oof:
            self._prepare_oof_arrays()

        start_time = time.time()

        # 训练每一折
        for fold in range(self.n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Training Fold {fold + 1}/{self.n_splits}")
            logger.info(f"{'='*60}")

            fold_start_time = time.time()

            # 训练单折
            fold_result = self._train_single_fold(fold)
            self.fold_results.append(fold_result)

            fold_time = time.time() - fold_start_time
            logger.info(f"✅ Fold {fold + 1} completed in {fold_time:.2f}s")
            logger.info(
                f"📊 Fold {fold + 1} {self.primary_metric}: {fold_result['val_metrics'][self.primary_metric]:.4f}"
            )

        total_time = time.time() - start_time

        # 生成最终结果
        final_results = self._generate_final_results(total_time)

        # 保存结果
        self._save_results(final_results)

        # 生成报告
        self._generate_reports()

        logger.info(f"🎉 K-Fold training completed in {total_time:.2f}s")
        logger.info(
            f"📈 Mean {self.primary_metric}: {final_results['mean_cv_score']:.4f} ± {final_results['std_cv_score']:.4f}"
        )

        return final_results

    def _prepare_oof_arrays(self) -> None:
        """准备OOF预测数组"""
        # 创建临时数据模块获取数据集大小
        temp_dm = create_kfold_datamodule(self.data_config, 0)
        temp_dm.prepare_data()
        temp_dm.setup("fit")

        total_samples = len(temp_dm.full_dataset)

        # 初始化OOF数组
        self.oof_predictions = np.zeros(total_samples)
        self.oof_targets = np.zeros(total_samples)

        logger.info(f"📋 Prepared OOF arrays for {total_samples} samples")

    def _train_single_fold(self, fold: int) -> Dict[str, Any]:
        """
        训练单个折

        Args:
            fold: 折索引

        Returns:
            单折训练结果
        """
        # 设置种子确保可重现性
        pl.seed_everything(self.seed + fold)

        # 创建数据模块
        datamodule = create_kfold_datamodule(self.data_config, fold)
        datamodule.prepare_data()
        datamodule.setup("fit")

        # 创建模型
        model = instantiate_from_config(self.model_config)

        # 设置回调函数
        callbacks = self._create_callbacks(fold)

        # 设置日志记录器
        logger_config = TensorBoardLogger(save_dir=str(self.experiment_dir / "logs"), name=f"fold_{fold}", version="")

        # 创建训练器
        trainer_params = self.trainer_config.copy()
        trainer_params.update(
            {
                "callbacks": callbacks,
                "logger": logger_config,
                "deterministic": True,
            }
        )

        trainer = pl.Trainer(**trainer_params)

        # 训练模型
        trainer.fit(model, datamodule)

        # 验证模型
        val_results = trainer.validate(model, datamodule, verbose=False)
        val_metrics = val_results[0] if val_results else {}

        # 生成OOF预测
        if self.generate_oof:
            self._generate_oof_predictions(model, datamodule, fold)

        # 生成测试集预测
        test_predictions = self._generate_test_predictions(model, datamodule, fold)
        self.test_predictions.append(test_predictions)

        # 保存模型
        if self.save_models:
            model_path = self.experiment_dir / "models" / f"fold_{fold}_model.ckpt"
            trainer.save_checkpoint(str(model_path))
            self.fold_models.append(str(model_path))

        return {
            "fold": fold,
            "val_metrics": val_metrics,
            "model_path": str(model_path) if self.save_models else None,
            "test_predictions": test_predictions,
        }

    def _create_callbacks(self, fold: int) -> List[pl.Callback]:
        """创建训练回调函数"""
        callbacks = []

        # 模型检查点
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.experiment_dir / "models"),
            filename=f"fold_{fold}_best_{{epoch}}_{{val_{self.primary_metric}:.4f}}",
            monitor=f"val_{self.primary_metric}",
            mode="max" if self.primary_metric in ["f1", "accuracy", "precision", "recall", "auroc"] else "min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # 早停
        early_stopping = EarlyStopping(
            monitor=f"val_{self.primary_metric}",
            mode="max" if self.primary_metric in ["f1", "accuracy", "precision", "recall", "auroc"] else "min",
            patience=self.early_stopping_patience,
            verbose=True,
            strict=True,
        )
        callbacks.append(early_stopping)

        return callbacks

    def _generate_oof_predictions(self, model: pl.LightningModule, datamodule: KFoldDataModule, fold: int) -> None:
        """生成OOF预测"""
        model.eval()

        # 获取验证集索引
        _, val_indices = datamodule.fold_indices[fold]

        # 生成预测
        predictions = []
        targets = []

        val_loader = datamodule.val_dataloader()
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    targets.extend(y.cpu().numpy())
                else:
                    x = batch

                # 模型预测
                logits = model(x.to(model.device))
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs.flatten())

        # 存储OOF预测
        self.oof_predictions[val_indices] = predictions
        self.oof_targets[val_indices] = targets

    def _generate_test_predictions(
        self, model: pl.LightningModule, datamodule: KFoldDataModule, fold: int
    ) -> np.ndarray:
        """生成测试集预测"""
        model.eval()

        # 设置测试数据
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                logits = model(x.to(model.device))
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs.flatten())

        return np.array(predictions)

    def _generate_final_results(self, total_time: float) -> Dict[str, Any]:
        """生成最终结果摘要"""
        # 计算CV分数统计
        cv_scores = [result["val_metrics"][self.primary_metric] for result in self.fold_results]
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        # 生成集成测试预测
        ensemble_predictions = np.mean(self.test_predictions, axis=0)

        # OOF性能评估
        oof_metrics = {}
        if self.generate_oof:
            oof_metrics = self._calculate_oof_metrics()

        # 集成所有信息
        final_results = {
            "experiment_name": self.experiment_name,
            "n_splits": self.n_splits,
            "fold_results": self.fold_results,
            "cv_scores": cv_scores,
            "mean_cv_score": mean_cv_score,
            "std_cv_score": std_cv_score,
            "oof_metrics": oof_metrics,
            "ensemble_predictions": ensemble_predictions,
            "training_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.model_config,
                "data": self.data_config,
                "trainer": self.trainer_config,
            },
        }

        return final_results

    def _calculate_oof_metrics(self) -> Dict[str, float]:
        """计算OOF预测的性能指标"""
        if self.oof_predictions is None:
            return {}

        # 二值化预测
        oof_pred_binary = (self.oof_predictions > 0.5).astype(int)

        # 计算各种指标
        f1 = f1_score(self.oof_targets, oof_pred_binary)
        auc = roc_auc_score(self.oof_targets, self.oof_predictions)

        # 生成分类报告
        report = classification_report(self.oof_targets, oof_pred_binary, output_dict=True)

        oof_metrics = {
            "f1_score": f1,
            "auc_score": auc,
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
        }

        return oof_metrics

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存训练结果"""
        # 保存主要结果
        results_file = self.experiment_dir / "results.json"

        # 准备可序列化的结果
        serializable_results = results.copy()
        serializable_results["ensemble_predictions"] = results["ensemble_predictions"].tolist()

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        # 保存OOF预测
        if self.generate_oof and self.oof_predictions is not None:
            oof_df = pd.DataFrame({"oof_predictions": self.oof_predictions, "targets": self.oof_targets})
            oof_df.to_csv(self.experiment_dir / "oof_predictions.csv", index=False)

        # 保存测试集预测
        if self.save_predictions:
            test_df = pd.DataFrame({"ensemble_prediction": results["ensemble_predictions"]})

            # 添加每折的预测
            for i, pred in enumerate(self.test_predictions):
                test_df[f"fold_{i}_prediction"] = pred

            test_df.to_csv(self.experiment_dir / "test_predictions.csv", index=False)

        logger.info(f"📁 Results saved to {self.experiment_dir}")

    def _generate_reports(self) -> None:
        """生成详细的分析报告"""
        # CV分数分析图
        self._plot_cv_scores()

        # OOF分析图
        if self.generate_oof and self.oof_predictions is not None:
            self._plot_oof_analysis()

        # 生成文字报告
        self._generate_text_report()

    def _plot_cv_scores(self) -> None:
        """绘制CV分数分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CV分数分布
        cv_scores = [result["val_metrics"][self.primary_metric] for result in self.fold_results]
        folds = [f"Fold {i+1}" for i in range(self.n_splits)]

        ax1.bar(folds, cv_scores, color="skyblue", alpha=0.7)
        ax1.axhline(y=np.mean(cv_scores), color="red", linestyle="--", label=f"Mean: {np.mean(cv_scores):.4f}")
        ax1.set_title(f"{self.primary_metric.upper()} Score by Fold")
        ax1.set_ylabel(f"{self.primary_metric.upper()} Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CV分数统计
        ax2.boxplot(cv_scores, labels=[f"{self.primary_metric.upper()}"])
        ax2.set_title(f"CV {self.primary_metric.upper()} Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "cv_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_oof_analysis(self) -> None:
        """绘制OOF分析图"""
        if self.oof_predictions is None:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 预测分布
        ax1.hist(self.oof_predictions[self.oof_targets == 0], alpha=0.5, label="Negative", bins=50)
        ax1.hist(self.oof_predictions[self.oof_targets == 1], alpha=0.5, label="Positive", bins=50)
        ax1.set_title("OOF Prediction Distribution")
        ax1.set_xlabel("Prediction Probability")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ROC曲线
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(self.oof_targets, self.oof_predictions)
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc_score(self.oof_targets, self.oof_predictions):.4f}")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax2.set_title("ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 混淆矩阵
        oof_pred_binary = (self.oof_predictions > 0.5).astype(int)
        cm = confusion_matrix(self.oof_targets, oof_pred_binary)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")

        # 预测校准图
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.oof_targets, self.oof_predictions, n_bins=10
        )
        ax4.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax4.set_title("Calibration Plot")
        ax4.set_xlabel("Mean Predicted Probability")
        ax4.set_ylabel("Fraction of Positives")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "oof_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_text_report(self) -> None:
        """生成文字报告"""
        report_path = self.experiment_dir / "reports" / "experiment_report.md"

        cv_scores = [result["val_metrics"][self.primary_metric] for result in self.fold_results]
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        report_content = f"""# K-Fold Cross Validation Report

## Experiment Information
- **Experiment Name**: {self.experiment_name}
- **Number of Folds**: {self.n_splits}
- **Primary Metric**: {self.primary_metric}
- **Stratified**: {self.stratified}
- **Seed**: {self.seed}

## Performance Summary
- **Mean CV Score**: {mean_cv_score:.4f} ± {std_cv_score:.4f}
- **Best Fold**: Fold {np.argmax(cv_scores) + 1} ({max(cv_scores):.4f})
- **Worst Fold**: Fold {np.argmin(cv_scores) + 1} ({min(cv_scores):.4f})

## Individual Fold Results
"""

        for i, (score, result) in enumerate(zip(cv_scores, self.fold_results)):
            report_content += f"- **Fold {i+1}**: {score:.4f}\n"

        if self.generate_oof and self.oof_predictions is not None:
            oof_metrics = self._calculate_oof_metrics()
            report_content += f"""
## OOF Performance
- **F1 Score**: {oof_metrics.get('f1_score', 0):.4f}
- **AUC Score**: {oof_metrics.get('auc_score', 0):.4f}
- **Accuracy**: {oof_metrics.get('accuracy', 0):.4f}
- **Precision**: {oof_metrics.get('precision', 0):.4f}
- **Recall**: {oof_metrics.get('recall', 0):.4f}
"""

        report_content += f"""
## Configuration
```json
{json.dumps({
    'model': self.model_config,
    'data': {k: v for k, v in self.data_config.items() if k != 'transforms'},
    'trainer': self.trainer_config,
}, indent=2)}
```

## Files Generated
- `results.json`: Complete experiment results
- `oof_predictions.csv`: Out-of-fold predictions
- `test_predictions.csv`: Test set predictions
- `plots/cv_analysis.png`: Cross-validation analysis
- `plots/oof_analysis.png`: OOF prediction analysis
- `models/`: Trained model checkpoints
"""

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"📊 Report generated: {report_path}")

    def load_results(self, results_path: str) -> Dict[str, Any]:
        """加载之前的实验结果"""
        with open(results_path, "r") as f:
            return json.load(f)

    def predict_test(self, test_data_path: str = None) -> np.ndarray:
        """使用训练好的模型进行测试集预测"""
        if not self.test_predictions:
            raise ValueError("No test predictions available. Run train_kfold() first.")

        return np.mean(self.test_predictions, axis=0)


# 便利函数
def run_kfold_experiment(config_path: str, n_splits: int = 5) -> Dict[str, Any]:
    """
    运行完整的K折实验的便利函数

    Args:
        config_path: 配置文件路径
        n_splits: 折数

    Returns:
        实验结果
    """
    from omegaconf import OmegaConf

    # 加载配置
    config = OmegaConf.load(config_path)

    # 创建K折训练器
    trainer = KFoldTrainer(
        model_config=config.model,
        data_config=config.data.params,
        trainer_config=config.trainer.params,
        n_splits=n_splits,
        experiment_name=config.get("experiment_name", "kfold_experiment"),
    )

    # 运行训练
    return trainer.train_kfold()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("✓ KFoldTrainer implementation completed!")
    print("Ready for Kaggle-level cross validation!")
