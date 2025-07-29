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

logger = get_project_logger(__name__)


class ExperimentRunner:
    """
    增强的实验运行器 - 支持K折交叉验证

    新增功能：
    1. 支持kfold任务类型
    2. 保持现有架构的简洁性
    3. 委托复杂逻辑给专门的KFoldTrainer
    """

    def __init__(self, config_path: str, task: str = "train", **kwargs):
        """
        初始化实验运行器

        Args:
            config_path: 配置文件路径
            task: 要执行的任务类型（train/predict/kfold）
            **kwargs: 额外的任务参数
        """
        setup_logging(level=logging.INFO)
        self.config_path = Path(config_path)
        self.task = task
        self.task_kwargs = kwargs  # 存储额外的任务参数
        self.config = self._load_config()
        self._setup_environment()

    def _load_config(self) -> DictConfig:
        """加载和验证配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading config from: {self.config_path}")
        config = OmegaConf.load(self.config_path)

        # 验证配置结构
        if not validate_config_structure(config):
            raise ValueError("Invalid configuration structure")

        logger.info("✓ Configuration loaded and validated")
        return config

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

        # 2. 遍历所有子目录配置，创建目录并更新配置
        # .items()是字典的一个方法，它会把字典里的每一对“键 (key)”和“值 (value)”拿出来，组成一个一个的元组 (tuple)。
        for key, subdir_name in list(self.config.outputs.items()):
            if key.endswith("_subdir"):
                # 构建完整的目录路径
                full_path = experiment_path / subdir_name
                full_path.mkdir(parents=True, exist_ok=True)

                # 生成新的配置键 (例如, 'checkpoint_subdir' -> 'checkpoint_dir')
                new_key = key.replace("_subdir", "_dir")

                # 将动态生成的完整路径更新回配置对象
                OmegaConf.update(self.config.outputs, new_key, str(full_path))

                logger.debug(f"创建并配置目录: {new_key} = {full_path}")

        # 3. 清理旧的 subdir 配置（可选，但保持配置整洁）
        for key in list(self.config.outputs.keys()):
            if key.endswith("_subdir"):
                del self.config.outputs[key]

    def _save_config(self):
        """保存配置文件到实验目录（确保可重现性），如果outputs在config中，则保存config.yaml到outputs.log_dir目录下，
        如果outputs.log_dir不存在，则创建log_dir目录"""
        if "outputs" in self.config and "log_dir" in self.config.outputs:
            timestamp = datetime.now().strftime(self.config.outputs.get("timestamp_format", "%Y%m%d_%H%M%S"))
            config_save_path = Path(self.config.outputs.log_dir) / f"config_{timestamp}.yaml"
            config_save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_save_path, "w") as f:
                OmegaConf.save(self.config, f)

            logger.info(f"Config saved to: {config_save_path}")

    def run(self) -> Dict[str, Any]:
        """
        运行实验的主方法 - 扩展支持K折交叉验证

        Returns:
            实验结果字典
        """
        logger.info(f"🚀 Starting {self.task} task")
        self._print_experiment_info()

        # 扩展的任务方法映射
        task_methods = {
            "train": self._run_training,
            "predict": self._run_prediction,
            "kfold": self._run_kfold,  # 新增K折任务
        }

        if self.task not in task_methods:
            raise ValueError(f"Unknown task: {self.task}. Available: {list(task_methods.keys())}")

        return task_methods[self.task]()

    def _run_training(self) -> Dict[str, Any]:
        """执行标准训练任务"""
        logger.info("Initializing training components...")

        # 创建模型和数据模块
        model = instantiate_from_config(self.config.model)
        data_module = instantiate_from_config(self.config.data)

        # 处理trainer配置
        trainer_config = self.config.trainer.copy()
        callbacks = self._create_callbacks()
        loggers = self._create_loggers()

        # 创建trainer
        trainer = instantiate_from_config(trainer_config)

        if callbacks:
            trainer.callbacks = callbacks
        if loggers:
            trainer.logger = loggers[0] if len(loggers) == 1 else loggers

        # 开始训练
        logger.info("🚀 Starting training...")
        trainer.fit(model, data_module)
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

        return {
            "status": "completed",
            "trainer": trainer,
            "model": model,
            "best_checkpoint": self._get_best_checkpoint_path(trainer),
        }

    def _run_kfold(self) -> Dict[str, Any]:
        """
        执行K折交叉验证任务 - 重构版本

        我们不再需要复杂的配置解析和参数传递，
        而是直接将完整配置传递给专门的K折训练器。

        就像把整个菜谱交给专业厨师，而不是逐个解释每个步骤。
        """
        logger.info("🎯 Initializing K-Fold Cross Validation...")

        # 验证配置完整性
        self._validate_kfold_config()

        # 获取实验输出路径 - 与基础训练完全一致的逻辑
        experiment_output_dir = self._get_experiment_output_dir()

        logger.info(f"📁 K-fold experiment will be saved to: {experiment_output_dir}")

        # 创建并运行K折训练器
        kfold_trainer = SimpleKFoldTrainer(
            config=self.config,
            experiment_name=self.config.get("experiment_name", "kfold_experiment"),
            output_dir=str(experiment_output_dir),  # 使用与基础训练一致的路径
        )

        # 执行K折训练
        results = kfold_trainer.run_kfold_training()

        # 打印结果摘要
        self._print_kfold_summary(results)

        return results

    def _get_experiment_output_dir(self) -> Path:
        """
        获取实验输出目录 - 与基础训练保持完全一致

        这个方法复用了基础训练中_create_output_dirs()的核心逻辑，
        确保k-fold训练的输出路径与标准训练完全一致。

        设计思想：
        就像工厂中的标准化生产流程，无论生产什么产品，
        都遵循相同的质量标准和工序规范。

        Returns:
            实验输出目录的完整路径
        """
        # 1. 获取基础输出目录 - 与基础训练相同的逻辑
        base_dir = Path(self.config.outputs.base_output_dir)

        # 2. 使用实验名称构建完整路径 - 与基础训练相同的逻辑
        experiment_path = base_dir / self.config.experiment_name

        # 3. 确保目录存在
        experiment_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"📂 Experiment output directory: {experiment_path}")

        return experiment_path

    def _validate_kfold_config(self):
        """
        验证K折配置的完整性

        这个方法确保用户提供的配置包含所有必需的部分。
        就像检查菜谱是否包含所有必需的材料和步骤。
        """
        required_sections = ["model", "data", "trainer"]
        missing_sections = []

        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)

        if missing_sections:
            raise ValueError(
                f"K-fold training requires the following config sections: {missing_sections}. "
                f"Please ensure your config file includes all required sections."
            )

        # 如果没有kfold配置，我们添加默认配置
        if "kfold" not in self.config:
            logger.info("🔧 No 'kfold' section found in config, using defaults")
            self.config["kfold"] = {
                "n_splits": 5,
                "stratified": True,
                "primary_metric": "f1",
                "save_oof_predictions": True,
                "save_fold_models": True,
            }

        logger.info("✅ K-fold configuration validated")

    def _print_kfold_summary(self, results: Dict[str, Any]):
        """
        打印K折训练结果摘要

        这个方法为用户提供清晰、易读的结果摘要。
        就像为一场音乐会制作节目单，突出最重要的信息。
        """
        print("\n" + "=" * 80)
        print("🎉 K-FOLD CROSS VALIDATION COMPLETED")
        print("=" * 80)

        print(f"📊 Experiment: {results['experiment_name']}")
        print(f"🎯 {results['n_splits']}-Fold Cross Validation")
        print(f"⏱️  Total Time: {results['total_time']:.2f}s")
        print(
            f"📈 Primary Metric ({results['primary_metric']}): {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}"
        )

        print(f"\n📋 Individual Fold Results:")
        for i, score in enumerate(results["cv_scores"]):
            print(f"   Fold {i+1}: {score:.4f}")

        print(f"\n📁 Results saved to: outputs/kfold_experiments/{results['experiment_name']}")
        print("=" * 80 + "\n")

    def _run_prediction(self) -> Dict[str, Any]:
        """
        执行推理任务

        推理任务用于在新数据上生成预测结果。它特别适用于：
        1. 生成竞赛提交文件
        2. 对新的遥感图像进行滑坡检测
        3. 批量处理大量图像

        Returns:
            包含预测结果和输出文件路径的字典
        """
        logger.info("🔮 Initializing prediction task...")

        # 验证必需的配置
        if "checkpoint_path" not in self.config:
            raise ValueError("Prediction requires 'checkpoint_path' in config")

        checkpoint_path = self.config.checkpoint_path
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from: {checkpoint_path}")

        # 创建组件
        model = instantiate_from_config(self.config.model)
        data_module = instantiate_from_config(self.config.data)

        # 为推理任务配置trainer
        trainer_config = self.config.trainer.copy()
        trainer_config.params.update(
            {
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": True,
            }
        )
        trainer = instantiate_from_config(trainer_config)

        # 运行预测
        logger.info("🎯 Generating predictions...")
        predictions = trainer.predict(model, data_module, ckpt_path=checkpoint_path)

        # 处理和保存预测结果
        processed_predictions = self._process_predictions(predictions)
        output_files = self._save_predictions(processed_predictions)

        logger.info("✅ Prediction completed successfully!")
        return {
            "status": "completed",
            "predictions": processed_predictions,
            "output_files": output_files,
            "checkpoint_used": checkpoint_path,
            "num_samples": len(processed_predictions) if processed_predictions else 0,
        }

    def _process_predictions(self, raw_predictions: List) -> List[Dict]:
        """
        处理原始预测结果

        将PyTorch张量转换为可序列化的格式，并添加必要的元数据。

        Args:
            raw_predictions: trainer.predict()的原始返回结果

        Returns:
            处理后的预测结果列表
        """
        import torch
        import numpy as np

        processed = []

        for batch_idx, batch_predictions in enumerate(raw_predictions):
            # 如果预测结果是张量，转换为numpy数组
            if isinstance(batch_predictions, torch.Tensor):
                predictions_np = batch_predictions.cpu().numpy()
            else:
                predictions_np = batch_predictions

            # 处理每个样本的预测
            for sample_idx, prediction in enumerate(predictions_np):
                processed.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "prediction": float(prediction) if np.isscalar(prediction) else prediction.tolist(),
                        "probability": (
                            float(torch.sigmoid(torch.tensor(prediction)).item()) if np.isscalar(prediction) else None
                        ),
                    }
                )

        logger.info(f"Processed {len(processed)} predictions")
        return processed

    def _save_predictions(self, predictions: List[Dict]) -> Dict[str, Path]:
        """
        保存预测结果到多种格式的文件

        Args:
            predictions: 处理后的预测结果

        Returns:
            保存的文件路径字典
        """
        import json
        import pandas as pd
        from datetime import datetime

        # 创建输出目录
        output_dir = Path(self.config.outputs.get("predictions_dir", "outputs/predictions"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"predictions_{self.config.experiment_name}_{timestamp}"

        output_files = {}

        # 保存JSON格式（完整信息）
        json_file = output_dir / f"{base_filename}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "experiment_name": self.config.experiment_name,
                    "timestamp": timestamp,
                    "checkpoint_path": self.config.get("checkpoint_path"),
                    "num_predictions": len(predictions),
                    "predictions": predictions,
                },
                f,
                indent=2,
            )
        output_files["json"] = json_file

        # 保存CSV格式（便于分析）
        if predictions:
            df_data = []
            for pred in predictions:
                df_data.append(
                    {
                        "sample_id": f"sample_{pred['batch_idx']}_{pred['sample_idx']}",
                        "prediction": pred["prediction"],
                        "probability": pred.get("probability", None),
                    }
                )

            df = pd.DataFrame(df_data)
            csv_file = output_dir / f"{base_filename}.csv"
            df.to_csv(csv_file, index=False)
            output_files["csv"] = csv_file

        logger.info(f"Predictions saved to: {list(output_files.values())}")
        return output_files

    def _create_callbacks(self) -> List:
        """
        创建callbacks - 独立的方法，更清晰的职责分离
        callbacks的作用：
        1. 在训练过程中，记录训练日志
        2. 在训练过程中，保存模型
        3. 在训练过程中，保存最佳模型
        4. 在训练过程中，保存验证集上的最佳模型
        5. 在训练过程中，保存测试集上的最佳模型
        6. 在训练过程中，保存训练集上的最佳模型
        """
        callbacks = []

        if "callbacks" not in self.config:
            return callbacks

        # 获取动态生成的路径
        dynamic_checkpoint_dir = self.config.outputs.get("checkpoint_dir")
        dynamic_log_dir = self.config.outputs.get("log_dir")

        logger.info("-" * 100)

        for callback_name, callback_config in self.config.callbacks.items():
            # 深拷贝配置，避免修改原始配置
            effective_config = callback_config.copy()

            # 如果是 ModelCheckpoint 且没有 dirpath，使用动态路径
            if callback_config.target == "pytorch_lightning.callbacks.ModelCheckpoint" and dynamic_checkpoint_dir:
                # 直接设置动态路径，覆盖配置文件中的静态路径
                OmegaConf.update(effective_config, "params.dirpath", dynamic_checkpoint_dir)

            if callback_config.target == "lightning_landslide.src.utils.metrics.MetricsLogger" and dynamic_log_dir:
                OmegaConf.update(effective_config, "params.log_dir", dynamic_log_dir)

            # 创建callback
            callback = instantiate_from_config(effective_config)
            callbacks.append(callback)
            logger.info(f"✓ Added callback: {callback_name} ({type(callback).__name__})")

        logger.info("-" * 100)
        return callbacks

    def _create_loggers(self) -> List:
        """
        创建loggers，并确保它们使用动态生成的路径。
        """
        loggers = []

        if "loggers" not in self.config:
            return loggers

        # 获取我们动态创建的日志目录
        dynamic_log_dir = self.config.outputs.get("log_dir")

        logger.info("-" * 100)

        for logger_name, logger_config in self.config.loggers.items():
            if logger_name == "tensorboard":
                effective_config = logger_config.copy()
                OmegaConf.update(effective_config, "params.save_dir", dynamic_log_dir)
                OmegaConf.update(effective_config, "params.name", "")
                OmegaConf.update(effective_config, "params.version", "")
                logger.info(f"Logger '{logger_name}' 将使用动态路径: {dynamic_log_dir}")

                # 使用更新后的配置来实例化
                lightning_logger = instantiate_from_config(effective_config)
                loggers.append(lightning_logger)
                logger.info(f"✓ Added logger: {logger_name} ({type(lightning_logger).__name__})")

            elif logger_name == "wandb":
                # WandB配置保持不变
                lightning_logger = instantiate_from_config(logger_config)
                loggers.append(lightning_logger)
                logger.info(f"✓ Added logger: {logger_name} ({type(lightning_logger).__name__})")
        logger.info("-" * 100)
        return loggers

    def _get_best_checkpoint_path(self, trainer) -> Optional[str]:
        """获取最佳检查点路径"""
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                return getattr(callback, "best_model_path", None)
        return None

    def _print_experiment_info(self):
        """打印实验信息"""
        print("\n" + "=" * 80)
        print(f"🚀 MM-LandslideNet Experiment: {self.config.get('experiment_name', 'Unnamed')}")
        print("=" * 80)
        print(f"📝 Task: {self.task}")
        print(f"📁 Config: {self.config_path}")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if "model" in self.config:
            print(f"🧠 Model: {self.config.model.target.split('.')[-1]}")
        if "data" in self.config:
            print(f"📊 Data: {self.config.data.get('params', {}).get('train_data_dir', 'N/A')}")

        print("=" * 80 + "\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MM-LandslideNet: Configuration-Driven Deep Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 标准训练
  python main.py train configs/optical_baseline.yaml
  
  # K折交叉验证训练  
  python main.py kfold lightning_landslide/configs/optical_baseline_5-fold.yaml
  
  # K折训练，覆盖折数
  python main.py kfold configs/optical_baseline_kfold.yaml --n_splits 10
  
  # K折训练，自定义实验名称
  python main.py kfold configs/optical_baseline_kfold.yaml --experiment_name my_kfold_experiment
        """,
    )

    parser.add_argument(
        "task",
        choices=["train", "predict", "kfold"],  # 简化任务选择，移除复杂的kfold_predict
        help="Task to execute",
    )

    parser.add_argument("config", type=str, help="Path to configuration file")

    # K折特定参数
    parser.add_argument("--n_splits", type=int, help="Number of folds for K-fold CV (overrides config)")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")

    return parser


def main():
    """
    主函数

    这是整个程序的入口点。与您原来的main.py相比，
    新版本的逻辑极其简洁：
    1. 解析命令行参数
    2. 创建实验运行器
    3. 运行实验
    4. 报告结果

    所有的复杂性都被配置文件和实例化工具吸收了。
    """
    parser = create_parser()  # 创建命令行参数解析器
    args = parser.parse_args()  # 解析命令行参数

    # 创建实验运行器
    runner = ExperimentRunner(args.config, args.task)

    # 运行实验
    results = runner.run()

    # 报告结果
    print(f"\n🎉 Task '{args.task}' completed successfully!")

    if args.task == "kfold":
        if "mean_cv_score" in results:
            print(f"📈 Mean CV Score: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")
    elif results.get("best_checkpoint"):
        print(f"📁 Best model saved to: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
