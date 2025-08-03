import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
from datetime import datetime
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# 导入我们的核心工具
from lightning_landslide.src.utils.instantiate import instantiate_from_config, validate_config_structure
from lightning_landslide.src.utils.logging_utils import setup_logging, get_project_logger
from lightning_landslide.src.training.simple_kfold_trainer import SimpleKFoldTrainer

# 导入主动学习模块
from lightning_landslide.src.active_learning.active_steps import ActiveLearningStepManager

logger = get_project_logger(__name__)


class ExperimentRunner:
    """
    20250728-新增功能：
    1. 支持kfold任务类型
    2. 保持现有架构的简洁性
    3. 委托复杂逻辑给专门的KFoldTrainer

    20250729-新增功能：
    1. active_train: 主动学习+伪标签训练
    2. 完全向后兼容现有功能
    3. 智能配置验证和错误处理

    20250731-新增支持：
    1. uncertainty_estimation: 不确定性估计
    2. sample_selection: 样本选择
    3. retrain: 模型重训练
    4. 保持所有现有功能不变
    """

    def __init__(self, config_path: str, task: str = "train", **kwargs):
        """
        初始化实验运行器

        Args:
            config_path: 配置文件路径
            task: 任务类型
            **kwargs: 额外的任务参数
        """
        setup_logging(level=logging.INFO)
        self.config_path = Path(config_path)
        self.task = task
        self.task_kwargs = kwargs
        self.config = self._load_config()
        self._setup_environment()

    def _load_config(self) -> DictConfig:
        """加载和验证配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading config from: {self.config_path}")
        config = OmegaConf.load(self.config_path)

        # 基础验证
        if not validate_config_structure(config):
            raise ValueError("Invalid configuration structure")

        # 主动学习任务需要验证相关配置
        if self.task in ["active_train", "uncertainty_estimation", "sample_selection", "retrain"]:
            self._validate_active_learning_config(config)

        logger.info("✓ Configuration loaded and validated")
        return config

    def _validate_active_learning_config(self, config: DictConfig):
        """验证主动学习配置"""
        if "active_pseudo_learning" not in config:
            logger.warning("Missing 'active_pseudo_learning' section, using defaults")
            config.active_pseudo_learning = {}

        logger.info("✓ Active learning configuration validated")

    def _setup_environment(self):
        """设置实验环境"""
        # 创建输出目录
        self._create_output_dirs()

        # 设置日志
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        log_file = None

        if "outputs" in self.config and "log_dir" in self.config.outputs:
            log_file = Path(self.config.outputs.log_dir) / f"{self.config.experiment_name}.log"

        setup_logging(level=log_level, log_file=log_file)

        # PyTorch设置
        if "seed" in self.config:
            pl.seed_everything(self.config.seed, workers=True)

        # GPU设置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = self.config.get("cudnn_benchmark", True)

    def _create_output_dirs(self):
        """创建输出目录"""
        if "outputs" not in self.config:
            # 创建默认输出配置
            experiment_name = self.config.get("experiment_name", f"exp_{int(datetime.now().timestamp())}")
            exp_dir = Path("lightning_landslide/exp") / experiment_name

            self.config.outputs = {
                "experiment_dir": str(exp_dir),
                "log_dir": str(exp_dir / "logs"),
                "model_dir": str(exp_dir / "models"),
                "checkpoint_dir": str(exp_dir / "checkpoints"),
            }

        # 创建所有必要的目录
        for dir_key, dir_path in self.config.outputs.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """运行实验"""
        self._print_experiment_info()

        start_time = datetime.now()

        try:
            # 根据任务类型执行不同逻辑
            if self.task == "train":
                results = self._run_training()
            elif self.task == "kfold":
                results = self._run_kfold_training()
            elif self.task == "uncertainty_estimation":
                results = self._run_uncertainty_estimation()
            elif self.task == "sample_selection":
                results = self._run_sample_selection()
            elif self.task == "retrain":
                results = self._run_retraining()
            elif self.task == "predict":
                results = self._run_prediction()
            else:
                raise ValueError(f"Unknown task: {self.task}")

            # 计算运行时间
            end_time = datetime.now()
            results["execution_time"] = str(end_time - start_time)
            results["task"] = self.task

            logger.info(f"✅ {self.task.upper()} completed successfully")
            logger.info(f"⏱️ Total time: {results['execution_time']}")

            return results

        except Exception as e:
            logger.error(f"❌ {self.task.upper()} failed: {str(e)}")
            raise

    def _run_training(self) -> Dict[str, Any]:
        """运行基础训练（步骤1）"""
        logger.info("🚀 Running baseline training...")

        # 实例化组件
        model = instantiate_from_config(self.config.model)
        datamodule = instantiate_from_config(self.config.data)
        trainer = instantiate_from_config(self.config.trainer)

        # 设置回调
        if "callbacks" in self.config:
            callbacks = []
            for cb_name, cb_config in self.config.callbacks.items():
                # 🔧 动态替换ModelCheckpoint中的路径
                if cb_name == "model_checkpoint" and "dirpath" in cb_config.params:
                    # 替换实验名称变量
                    dirpath = cb_config.params.dirpath
                    if "${experiment_name}" in dirpath:
                        experiment_name = self.config.get("experiment_name", "default_exp")
                        cb_config.params.dirpath = dirpath.replace("${experiment_name}", experiment_name)

                callbacks.append(instantiate_from_config(cb_config))
            trainer.callbacks = callbacks

        # 🔧 动态设置Logger路径
        if "logger" in self.config:
            logger_config = self.config.logger.copy()  # 复制配置避免修改原始配置

            # 替换实验名称变量
            if "name" in logger_config.params and "${experiment_name}" in str(logger_config.params.name):
                experiment_name = self.config.get("experiment_name", "default_exp")
                logger_config.params.name = logger_config.params.name.replace("${experiment_name}", experiment_name)

            trainer.logger = instantiate_from_config(logger_config)

        # 训练模型
        trainer.fit(model, datamodule)
        """
            fit 是 pytorch_lightning.Trainer 类的核心方法，这个方法会自动执行整个训练流程：

            数据准备：调用 data_module.prepare_data() 和 data_module.setup()
            创建数据加载器：获取 train/val dataloaders
            训练循环：

            调用 model.training_step() 处理每个batch
            计算loss和梯度
            执行优化器更新

            验证循环：

            调用 model.validation_step()
            计算验证指标

            回调执行：运行checkpointing、early stopping等
            日志记录：自动记录所有指标
            # Lightning内部会调用
            ├── data_module.prepare_data()           # 数据准备
            ├── data_module.setup('fit')             # 数据集设置  
            ├── data_module.train_dataloader()       # 获取训练数据加载器
            ├── data_module.val_dataloader()         # 获取验证数据加载器
            ├── model.configure_optimizers()         # 配置优化器
            └── 训练循环:
                ├── model.training_step(batch, idx)  # 每个训练batch
                ├── optimizer.step()                 # 参数更新
                ├── model.validation_step(batch, idx) # 每个验证batch
                ├── callbacks.on_epoch_end()         # 回调函数
                └── logger.log_metrics()             # 记录指标
        """

        logger.info("🏆 Training completed successfully!")
        logger.info("📊 Model performance should be evaluated based on validation metrics only")
        logger.info("🎯 Use the validation F1 score as the primary performance indicator")

        # 从验证指标中提取最终性能
        best_val_metrics = {}
        if hasattr(trainer, "callback_metrics"):
            for key, value in trainer.callback_metrics.items():
                if "val_" in key:
                    best_val_metrics[key] = float(value) if hasattr(value, "item") else value

        # 保存最终模型
        final_model_path = Path(self.config.outputs.experiment_dir) / "checkpoints" / "final_model.ckpt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_model_path))

        return {
            "best_checkpoint": (
                trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None
            ),
            "final_model": str(final_model_path),
            "validation_metrics": best_val_metrics,  # 🔧 使用验证指标而不是测试指标
            "training_completed": True,
        }

    def _run_kfold_training(self) -> Dict[str, Any]:
        """运行K折交叉验证训练"""
        logger.info("🔄 Running K-fold cross-validation...")

        kfold_trainer = SimpleKFoldTrainer(
            config=dict(self.config),
            experiment_name=self.config.experiment_name,
            output_dir=self.config.outputs.experiment_dir,
        )

        return kfold_trainer.run_kfold_training()

    # =============================================================================
    # 分步主动学习方法（步骤2-5）
    # =============================================================================

    def _run_uncertainty_estimation(self) -> Dict[str, Any]:
        """运行不确定性估计（步骤2）"""
        logger.info("🔍 Running uncertainty estimation step...")

        state_path = self.task_kwargs.get("state_path")
        return ActiveLearningStepManager.run_uncertainty_estimation(config=dict(self.config), state_path=state_path)

    def _run_sample_selection(self) -> Dict[str, Any]:
        """运行样本选择（步骤3）"""
        logger.info("🎯 Running sample selection step...")

        state_path = self.task_kwargs.get("state_path")
        return ActiveLearningStepManager.run_sample_selection(config=dict(self.config), state_path=state_path)

    def _run_retraining(self) -> Dict[str, Any]:
        """运行模型重训练（步骤5）"""
        logger.info("🔄 Running model retraining step...")

        state_path = self.task_kwargs.get("state_path")
        annotation_file = self.task_kwargs.get("annotation_file")

        return ActiveLearningStepManager.run_retraining(
            config=dict(self.config), state_path=state_path, annotation_file=annotation_file
        )

    # =============================================================================
    # 预测
    # =============================================================================

    def _run_prediction(self) -> Dict[str, Any]:
        """运行预测（专门用于Kaggle提交）"""
        logger.info("🔮 Running prediction for Kaggle submission...")

        # 加载最佳模型
        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            # 自动查找最佳检查点
            exp_dir = Path(self.config.outputs.experiment_dir)
            checkpoint_dir = exp_dir / "checkpoints"

            # 查找最佳F1检查点
            best_checkpoints = list(checkpoint_dir.glob("best-epoch=*-val_f1=*.ckpt"))
            if best_checkpoints:
                checkpoint_path = str(sorted(best_checkpoints)[-1])  # 取最新的
            else:
                raise FileNotFoundError("No trained model found. Please run training first.")

        logger.info(f"📥 Loading model from: {checkpoint_path}")

        # 实例化组件
        model = instantiate_from_config(self.config.model)
        datamodule = instantiate_from_config(self.config.data)
        trainer = instantiate_from_config(self.config.trainer)

        # 加载检查点
        model = model.load_from_checkpoint(checkpoint_path)
        model.eval()

        # 设置数据（只需要测试集）
        datamodule.setup("predict")

        # 进行预测（不计算任何指标）
        predictions = trainer.predict(model, datamodule.predict_dataloader())

        # 处理预测结果
        all_probs = []
        all_preds = []

        for batch_preds in predictions:
            probs = batch_preds["probabilities"].cpu().numpy()
            preds = batch_preds["predictions"].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)

        # 获取测试样本ID
        test_dataset = datamodule.test_dataset
        sample_ids = [test_dataset.data_index.iloc[i]["ID"] for i in range(len(test_dataset))]

        # 创建提交文件
        submission_df = pd.DataFrame(
            {"ID": sample_ids, "label": [int(pred) for pred in all_preds]}  # Kaggle通常要求整数标签
        )

        # 保存提交文件
        submission_path = exp_dir / "kaggle_submission.csv"
        submission_df.to_csv(submission_path, index=False)

        # 保存详细预测结果（包含概率）
        detailed_results = pd.DataFrame({"ID": sample_ids, "probability": all_probs, "prediction": all_preds})

        detailed_path = exp_dir / "detailed_predictions.csv"
        detailed_results.to_csv(detailed_path, index=False)

        logger.info(f"✅ Prediction completed!")
        logger.info(f"📄 Kaggle submission saved to: {submission_path}")
        logger.info(f"📊 Detailed results saved to: {detailed_path}")
        logger.info(f"🎯 Predicted {len(sample_ids)} samples")

        # 预测统计
        positive_ratio = sum(all_preds) / len(all_preds)
        logger.info(f"📈 Positive prediction ratio: {positive_ratio:.3f}")

        return {
            "submission_path": str(submission_path),
            "detailed_path": str(detailed_path),
            "num_predictions": len(sample_ids),
            "positive_ratio": positive_ratio,
            "checkpoint_used": checkpoint_path,
        }

    def _print_experiment_info(self):
        """打印实验信息"""
        print("\n" + "=" * 80)
        print(f"🚀 MM-LANDSLIDE NET - {self.task.upper()}")
        print("=" * 80)
        print(f"📅 TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 TASK: {self.task}")
        print(f"📝 CONFIG: {self.config_path}")

        if "experiment_name" in self.config:
            print(f"🔬 EXPERIMENT: {self.config.experiment_name}")

        if "model" in self.config:
            model_name = str(self.config.model.target).split(".")[-1]
            print(f"🤖 MODEL: {model_name}")

        if "data" in self.config:
            data_dir = self.config.data.get("params", {}).get("train_data_dir", "N/A")
            print(f"📊 DATA: {data_dir}")

        # 主动学习特定信息
        if self.task in ["active_train", "uncertainty_estimation", "sample_selection", "retrain"]:
            if "active_pseudo_learning" in self.config:
                apl_config = self.config.active_pseudo_learning
                print(f"🎯 MAX ITERATIONS: {apl_config.get('max_iterations', 5)}")
                print(f"📝 ANNOTATION BUDGET: {apl_config.get('annotation_budget', 50)}")

        print("=" * 80 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="MM-LandslideNet: Stepwise Active Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 步骤1：基础训练
  python main.py train lightning_landslide/configs/optical_baseline.yaml
  
  # K折交叉验证
  python main.py kfold lightning_landslide/configs/optical_baseline_5-fold.yaml
  
  # === 分步主动学习 ===
  # 步骤2：不确定性估计
  python main.py uncertainty_estimation lightning_landslide/configs/optical_baseline_active_steps.yaml
  
  # 步骤3：样本选择
  python main.py sample_selection lightning_landslide/configs/optical_baseline_active_steps.yaml
  
  # 步骤5：模型重训练
  python main.py retrain lightning_landslide/configs/optical_baseline_active_steps.yaml \
--annotation_file lightning_landslide/exp/optical_swin_tiny_0731_active_steps/active_learning/annotation_results_iter_0.json
        """,
    )

    parser.add_argument(
        "task",
        choices=[
            "train",  # 基础训练
            "kfold",  # K折交叉验证
            "uncertainty_estimation",  # 步骤2：不确定性估计
            "sample_selection",  # 步骤3：样本选择
            "retrain",  # 步骤5：模型重训练
        ],
        help="Task to execute",
    )

    parser.add_argument("config", type=str, help="Path to configuration file")

    # 通用参数
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint path for prediction")

    # K折特定参数
    parser.add_argument("--n_splits", type=int, help="Number of folds for K-fold CV")

    # 主动学习特定参数
    parser.add_argument("--max_iterations", type=int, help="Maximum active learning iterations")
    parser.add_argument("--annotation_budget", type=int, help="Annotation budget per iteration")

    # 分步主动学习参数
    parser.add_argument("--state_path", type=str, help="Path to active learning state file")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation results file")

    return parser


def main():
    """
    主函数 - 程序入口

    增强版本支持更多任务类型，同时保持向后兼容性。
    """
    parser = create_parser()
    args = parser.parse_args()

    # 准备任务参数
    task_kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ["task", "config"]}

    try:
        # 创建并运行实验
        runner = ExperimentRunner(config_path=args.config, task=args.task, **task_kwargs)

        results = runner.run()

        # 打印结果摘要
        print("\n" + "=" * 80)
        print(f"✅ {args.task.upper()} COMPLETED SUCCESSFULLY")
        print("=" * 80)

        if "execution_time" in results:
            print(f"⏱️ Execution time: {results['execution_time']}")

        # 打印任务特定结果
        if args.task == "train":
            print(f"💾 Best checkpoint: {results.get('best_checkpoint', 'N/A')}")
            if "test_results" in results:
                test_f1 = results["test_results"].get("test_f1", "N/A")
                print(f"📈 Test F1 Score: {test_f1}")
        elif args.task == "uncertainty_estimation":
            print(f"📊 Estimated uncertainty for {results.get('num_samples', 0)} samples")
            print(f"📁 Results saved to: {results.get('results_path', 'N/A')}")
        elif args.task == "sample_selection":
            print(f"🎯 Selected {results.get('num_selected', 0)} samples for annotation")
            print(f"📝 Annotation request: {results.get('annotation_file', 'N/A')}")
        elif args.task == "retrain":
            print(f"📊 Added {results.get('num_annotations', 0)} human annotations")
            print(f"🏷️ Generated {results.get('num_pseudo_labels', 0)} pseudo labels")
            print(f"💾 New checkpoint: {results.get('new_checkpoint', 'N/A')}")

        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Task failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
