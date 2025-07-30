import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 抑制不必要的警告
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", ".*The dataloader.*")

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# 导入我们的核心工具
from lightning_landslide.src.utils.instantiate import instantiate_from_config, validate_config_structure
from lightning_landslide.src.utils.logging_utils import setup_logging, get_project_logger
from lightning_landslide.src.training.simple_kfold_trainer import SimpleKFoldTrainer
from lightning_landslide.src.active_learning.human_guided_active_learning import create_human_guided_active_learning

logger = get_project_logger(__name__)


class ExperimentRunner:
    """
    20250728-新增功能：
    1. 支持kfold任务类型
    2. 保持现有架构的简洁性
    3. 委托复杂逻辑给专门的KFoldTrainer

    20250729-新增功能：
    1. active_train: 主动学习+伪标签训练
    2. active_kfold: K折+主动学习融合
    3. 完全向后兼容现有功能
    4. 智能配置验证和错误处理
    """

    def __init__(self, config_path: str, task: str = "train", **kwargs):
        """
        初始化实验运行器

        Args:
            config_path: 配置文件路径
            task: 任务类型 (train/predict/kfold/active_train/active_kfold)
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

        # 主动学习特定验证
        if self.task in ["active_train", "active_kfold"]:
            self._validate_active_learning_config(config)

        logger.info("✓ Configuration loaded and validated")
        return config

    def _validate_active_learning_config(self, config: DictConfig):
        """验证主动学习配置"""
        if "active_pseudo_learning" not in config:
            raise ValueError("Missing 'active_pseudo_learning' section for active learning tasks")

        active_config = config.active_pseudo_learning
        required_sections = ["uncertainty_estimation", "pseudo_labeling", "active_learning"]

        for section in required_sections:
            if section not in active_config:
                logger.warning(f"Missing '{section}' in active_pseudo_learning config, using defaults")

        logger.info("✓ Active learning configuration validated")

    def _setup_environment(self):
        """设置实验环境"""
        # 创建输出目录
        self._create_output_dirs()

        # 设置日志，getattr的作用是获取config中的log_level，如果没有则使用INFO
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        log_file = None

        if "outputs" in self.config and "log_dir" in self.config.outputs:
            log_file = Path(self.config.outputs.log_dir) / f"{self.config.experiment_name}.log"

        setup_logging(
            level=log_level,
            log_file=str(log_file) if log_file else None,
            use_colors=True,
        )

        # 设置随机种子，seed_everything的作用是设置随机种子，并设置torch.manual_seed和torch.cuda.manual_seed，
        # workers为True时，会设置torch.utils.data.DataLoader的num_workers为1
        if "seed" in self.config:
            pl.seed_everything(self.config.seed, workers=True)

        # 保存配置文件到实验目录
        self._save_config()

    def _create_output_dirs(self):
        """根据experiment_name动态创建实验输出目录"""
        base_dir = Path(self.config.outputs.base_output_dir)
        experiment_path = base_dir / self.config.experiment_name
        logger.info(f"所有实验输出将保存到: {experiment_path}")

        # 创建所有必要的子目录
        dirs_to_create = ["checkpoints", "logs", "predictions", "models", "visualizations", "data_versions"]

        # 主动学习特定目录
        if self.task in ["active_train", "active_kfold"]:
            dirs_to_create.extend(
                ["active_learning", "pseudo_labels", "uncertainty_analysis", "iteration_results", "annotations"]
            )

        for dir_name in dirs_to_create:
            (experiment_path / dir_name).mkdir(parents=True, exist_ok=True)

        # 更新配置中的路径
        self.config.outputs = OmegaConf.create(
            {
                "base_output_dir": str(base_dir),
                "experiment_dir": str(experiment_path),
                "checkpoint_dir": str(experiment_path / "checkpoints"),
                "log_dir": str(experiment_path / "logs"),
                "prediction_dir": str(experiment_path / "predictions"),
                "model_dir": str(experiment_path / "models"),
                "visualization_dir": str(experiment_path / "visualizations"),
            }
        )

    def _save_config(self):
        """保存配置文件到实验目录"""
        config_save_path = Path(self.config.outputs.experiment_dir) / "config.yaml"

        # 添加运行时信息
        runtime_info = {
            "runtime": {
                "task": self.task,
                "start_time": datetime.now().isoformat(),
                "config_path": str(self.config_path),
                "command_line_args": self.task_kwargs,
                "pytorch_version": str(torch.__version__),
                "pytorch_lightning_version": str(pl.__version__),
            }
        }

        # 合并配置
        enhanced_config = OmegaConf.merge(self.config, runtime_info)

        # 保存
        with open(config_save_path, "w") as f:
            OmegaConf.save(enhanced_config, f)

        logger.info(f"📄 Configuration saved to: {config_save_path}")

    def run(self) -> Dict[str, Any]:
        """运行实验的主入口"""
        logger.info(f"🚀 Starting task: {self.task}")
        self._print_experiment_banner()

        try:
            if self.task == "train":
                return self._run_standard_training()
            elif self.task == "predict":
                return self._run_prediction()
            elif self.task == "kfold":
                return self._run_kfold_training()
            elif self.task == "active_train":
                return self._run_active_training()
            elif self.task == "active_kfold":
                return self._run_active_kfold_training()
            else:
                raise ValueError(f"Unknown task: {self.task}")

        except Exception as e:
            logger.error(f"❌ Task '{self.task}' failed: {str(e)}")
            raise

    def _run_standard_training(self) -> Dict[str, Any]:
        """运行标准训练"""
        logger.info("🎯 Running standard training...")

        # 创建组件
        model = instantiate_from_config(self.config.model)
        datamodule = instantiate_from_config(self.config.data)
        trainer = self._create_standard_trainer()

        # 开始训练
        logger.info("🚀 Starting training...")
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

        # 在测试集上评估
        test_results = trainer.test(model, datamodule, verbose=False)

        # 保存最终模型
        final_model_path = Path(self.config.outputs.model_dir) / "final_model.ckpt"
        trainer.save_checkpoint(str(final_model_path))

        return {
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
            "final_model": str(final_model_path),
            "test_results": test_results[0] if test_results else {},
            "training_completed": True,
        }

    def _run_kfold_training(self) -> Dict[str, Any]:
        """运行K折交叉验证训练"""
        logger.info("🔄 Running K-fold cross-validation...")

        # 使用现有的SimpleKFoldTrainer
        kfold_trainer = SimpleKFoldTrainer(
            config=dict(self.config),
            experiment_name=self.config.experiment_name,
            output_dir=self.config.outputs.experiment_dir,
        )

        return kfold_trainer.run_kfold_training()

    def _run_active_training(self) -> Dict[str, Any]:
        """运行主动学习训练 - 支持人工指导"""
        logger.info("🎯🏷️ Running Active Learning + Pseudo Labeling...")

        # 检查标注模式
        annotation_mode = self.config.get("active_pseudo_learning", {}).get("annotation_mode", "simulated")

        if annotation_mode == "human":
            logger.info("👤 Using HUMAN-GUIDED active learning")
            # 使用人工指导实现
            trainer = create_human_guided_active_learning(
                config=dict(self.config),
                experiment_name=self.config.experiment_name,
                output_dir=self.config.outputs.experiment_dir,
            )
        else:
            # 结束程序
            raise ValueError("Human-guided active learning is not implemented yet")

        # 运行主动学习流程
        results = trainer.run()

        return {
            "active_learning_results": results,
            "annotation_mode": annotation_mode,
            "training_completed": True,
        }

    def _run_active_kfold_training(self) -> Dict[str, Any]:
        """运行主动学习+K折交叉验证融合训练"""
        logger.info("🔄🎯🏷️ Running Active Learning + K-fold Cross-validation...")

        # 这是一个更复杂的组合策略
        # 我们将在每个fold中都应用主动学习
        from lightning_landslide.src.training.active_kfold_trainer import ActiveKFoldTrainer

        active_kfold_trainer = ActiveKFoldTrainer(
            config=dict(self.config),
            experiment_name=self.config.experiment_name,
            output_dir=self.config.outputs.experiment_dir,
        )

        return active_kfold_trainer.run()

    def _run_prediction(self) -> Dict[str, Any]:
        """运行预测任务"""
        logger.info("🔮 Running prediction...")

        # 加载模型
        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for prediction task")

        model = instantiate_from_config(self.config.model)
        model = model.load_from_checkpoint(checkpoint_path)

        # 创建数据模块
        datamodule = instantiate_from_config(self.config.data)

        # 创建预测器
        trainer = self._create_standard_trainer()

        # 进行预测
        predictions = trainer.predict(model, datamodule)

        # 保存预测结果
        prediction_path = Path(self.config.outputs.prediction_dir) / "predictions.csv"
        # 这里需要根据具体的预测格式来保存

        return {
            "prediction_path": str(prediction_path),
            "num_predictions": len(predictions) if predictions else 0,
            "prediction_completed": True,
        }

    def _create_standard_trainer(self) -> pl.Trainer:
        """创建标准PyTorch Lightning训练器"""
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger

        # 基础训练器配置
        trainer_config = dict(self.config.trainer.params)

        # 设置回调
        callbacks = []

        # 模型检查点回调
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.outputs.checkpoint_dir,
            filename="{epoch}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # 早停回调
        early_stopping = EarlyStopping(
            monitor="val_f1",
            patience=15,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stopping)

        # 日志记录器
        tb_logger = TensorBoardLogger(
            save_dir=self.config.outputs.log_dir,
            name="training",
            version="",
        )

        # 创建训练器
        trainer = pl.Trainer(**trainer_config)
        trainer.callbacks = callbacks
        trainer.logger = tb_logger

        return trainer

    def _print_experiment_banner(self):
        """打印实验信息横幅"""
        print("\n" + "=" * 80)
        print(f"🧪 EXPERIMENT: {self.config.experiment_name}")
        print(f"📋 TASK: {self.task.upper()}")
        print(f"⏰ START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 OUTPUT DIR: {self.config.outputs.experiment_dir}")

        if "model" in self.config:
            model_name = self.config.model.get("target", "Unknown").split(".")[-1]
            print(f"🤖 MODEL: {model_name}")

        if "data" in self.config:
            data_dir = self.config.data.get("params", {}).get("train_data_dir", "N/A")
            print(f"📊 DATA: {data_dir}")

        # 主动学习特定信息
        if self.task in ["active_train", "active_kfold"] and "active_pseudo_learning" in self.config:
            apl_config = self.config.active_pseudo_learning
            print(f"🎯 MAX ITERATIONS: {apl_config.get('max_iterations', 5)}")
            print(f"🏷️ PSEUDO THRESHOLD: {apl_config.get('pseudo_labeling', {}).get('confidence_threshold', 0.9)}")
            print(f"📝 ANNOTATION BUDGET: {apl_config.get('annotation_budget', 50)}")

        print("=" * 80 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="MM-LandslideNet: Enhanced Deep Learning Framework with Active Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 标准训练
  python main.py train lightning_landslide/configs/optical_baseline.yaml
  
  # K折交叉验证训练  
  python main.py kfold lightning_landslide/configs/optical_baseline_5-fold.yaml
  
  # 主动学习+伪标签训练
  python main.py active_train lightning_landslide/configs/optical_baseline_active.yaml
  
  # 主动学习+K折交叉验证
  python main.py active_kfold configs/optical_baseline_active_kfold.yaml
  
  # 预测
  python main.py predict configs/predict_config.yaml --checkpoint_path path/to/model.ckpt
        """,
    )

    parser.add_argument(
        "task",
        choices=["train", "predict", "kfold", "active_train", "active_kfold"],
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
    parser.add_argument("--pseudo_threshold", type=float, help="Pseudo label confidence threshold")

    return parser


def main():
    """
    主函数 - 程序入口

    增强版本支持更多任务类型，同时保持向后兼容性。
    """
    parser = create_parser()
    args = parser.parse_args()

    # 准备任务参数
    task_kwargs = {}
    if args.experiment_name:
        task_kwargs["experiment_name"] = args.experiment_name
    if args.checkpoint_path:
        task_kwargs["checkpoint_path"] = args.checkpoint_path
    if args.n_splits:
        task_kwargs["n_splits"] = args.n_splits
    if args.max_iterations:
        task_kwargs["max_iterations"] = args.max_iterations
    if args.annotation_budget:
        task_kwargs["annotation_budget"] = args.annotation_budget
    if args.pseudo_threshold:
        task_kwargs["pseudo_threshold"] = args.pseudo_threshold

    # 创建并运行实验
    try:
        runner = ExperimentRunner(args.config, args.task, **task_kwargs)
        results = runner.run()

        # 报告结果
        print(f"\n🎉 Task '{args.task}' completed successfully!")

        if args.task == "kfold":
            if "mean_cv_score" in results:
                print(f"📈 Mean CV Score: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")

        elif args.task in ["active_train", "active_kfold"]:
            if "best_performance" in results:
                print(f"🏆 Best Performance: {results['best_performance']:.4f}")
            if "total_iterations" in results:
                print(f"🔄 Total Iterations: {results['total_iterations']}")

        elif args.task == "train":
            if results.get("best_checkpoint"):
                print(f"📁 Best model: {results['best_checkpoint']}")

        elif args.task == "predict":
            if results.get("prediction_path"):
                print(f"📄 Predictions saved: {results['prediction_path']}")

    except Exception as e:
        print(f"\n❌ Task failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
