"""
MM-LandslideNet 统一项目入口点

核心功能：
1. 模型训练 (train)
2. 模型测试 (test)
3. 模型推理 (predict)
4. 性能评估 (evaluate)
5. 数据分析 (analyze)
6. 模型转换 (convert)

使用示例：
1. 训练模型：python main.py train --config-path configs/experiment --config-name optical_baseline
2. 测试模型：python main.py test --checkpoint path/to/model.ckpt
3. 批量推理：python main.py predict --checkpoint path/to/model.ckpt --input-dir test_data/
4. 快速开始：python main.py train --preset quick_test

设计哲学：
"一个入口，多种可能" - 通过统一的接口，让复杂的深度学习工作流变得简单易用。
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 抑制不必要的警告
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The dataloader.*")

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir

# 导入项目模块
from lightning_landslide.src.utils.logging_utils import setup_logging, get_project_logger

logger = get_project_logger(__name__)


class TaskRunner:
    """
    任务执行器基类

    这个类定义了所有任务执行器的通用接口。每种具体的任务
    （训练、测试、推理等）都会继承这个基类，实现统一的
    执行模式。

    这种设计让main.py能够以相同的方式处理不同类型的任务，
    同时为每种任务提供了足够的定制空间。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.setup_environment()

    def setup_environment(self):
        """设置执行环境"""
        # 设置日志
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        setup_logging(level=log_level)

        # 设置随机种子（如果指定）
        if hasattr(self.args, "seed") and self.args.seed is not None:
            pl.seed_everything(self.args.seed, workers=True)
            logger.info(f"Set random seed to {self.args.seed}")

    def run(self) -> Dict[str, Any]:
        """执行任务的主方法，子类必须实现"""
        raise NotImplementedError("Subclasses must implement run method")

    def load_config(self, config_path: str = None, config_name: str = None) -> DictConfig:
        """
        加载配置文件

        这个方法提供了灵活的配置加载机制，支持多种配置来源：
        1. 命令行指定的配置文件
        2. 预设的配置模板
        3. 检查点中保存的配置
        """
        if config_path and config_name:
            # 从指定路径加载配置
            with initialize_config_dir(config_dir=str(Path(config_path).absolute())):
                cfg = compose(config_name=config_name)
        elif hasattr(self.args, "preset") and self.args.preset:
            # 使用预设配置
            cfg = self._load_preset_config(self.args.preset)
        else:
            # 使用默认配置
            cfg = self._load_default_config()

        # 应用命令行覆盖
        if hasattr(self.args, "overrides") and self.args.overrides:
            for override in self.args.overrides:
                self._apply_override(cfg, override)

        return cfg

    def _load_preset_config(self, preset_name: str) -> DictConfig:
        """加载预设配置"""
        preset_configs = {
            "quick_test": {
                "experiment": {"name": "quick_test", "description": "Quick test run"},
                "model": {"type": "optical_swin", "backbone_name": "swin_tiny_patch4_window7_224"},
                "data": {"batch_size": 16, "val_split": 0.3},
                "training": {"max_epochs": 10, "optimizer": {"lr": 2e-4}},
                "compute": {"precision": "16-mixed"},
            },
            "full_multimodal": {
                "experiment": {"name": "full_multimodal", "description": "Full 13-channel training"},
                "model": {"type": "optical_swin", "input_channels": 13},
                "data": {"usage_mode": "full_multimodal", "batch_size": 32},
                "training": {"max_epochs": 50},
            },
            "high_performance": {
                "experiment": {"name": "high_performance", "description": "High-performance training"},
                "model": {"type": "optical_swin", "backbone_name": "swin_base_patch4_window7_224"},
                "data": {"batch_size": 64, "use_weighted_sampling": True},
                "training": {"max_epochs": 100, "optimizer": {"layer_wise_lr": True}},
                "compute": {"precision": "16-mixed", "accumulate_grad_batches": 2},
            },
        }

        if preset_name not in preset_configs:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(preset_configs.keys())}")

        return OmegaConf.create(preset_configs[preset_name])

    def _load_default_config(self) -> DictConfig:
        """加载默认配置"""
        default_config_path = project_root / "configs" / "experiment" / "optical_baseline.yaml"
        if default_config_path.exists():
            return OmegaConf.load(default_config_path)
        else:
            # 如果配置文件不存在，返回最小配置
            logger.warning("Default config not found, using minimal configuration")
            return self._load_preset_config("quick_test")

    def _apply_override(self, cfg: DictConfig, override: str):
        """应用配置覆盖"""
        try:
            key, value = override.split("=", 1)
            # 尝试将值转换为适当的类型
            try:
                value = eval(value)  # 尝试解析为Python对象
            except:
                pass  # 保持为字符串

            OmegaConf.set(cfg, key, value)
            logger.info(f"Applied override: {key}={value}")
        except Exception as e:
            logger.warning(f"Failed to apply override '{override}': {e}")


class TrainTaskRunner(TaskRunner):
    """
    训练任务执行器

    这个类专门处理模型训练任务。它整合了我们之前构建的
    训练框架，同时提供了更灵活的配置和执行选项。
    """

    def run(self) -> Dict[str, Any]:
        """执行训练任务"""
        logger.info("🚀 Starting training task")

        # 加载配置
        cfg = self.load_config(
            config_path=getattr(self.args, "config_path", None), config_name=getattr(self.args, "config_name", None)
        )

        # 导入训练模块（延迟导入避免不必要的依赖）
        from lightning_landslide.src.models import LandslideClassificationModule
        from lightning_landslide.src.data import MultiModalDataModule
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger

        # 创建数据模块
        logger.info("Creating data module...")
        data_module = MultiModalDataModule(cfg)

        # 创建模型
        logger.info("Creating model...")
        model = LandslideClassificationModule(cfg)

        # 创建回调函数
        callbacks = self._create_callbacks(cfg)

        # 创建日志记录器
        loggers = self._create_loggers(cfg)

        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=cfg.compute.accelerator,
            devices=cfg.compute.devices,
            precision=cfg.compute.precision,
            callbacks=callbacks,
            logger=loggers,
            deterministic=cfg.reproducibility.deterministic,
            log_every_n_steps=cfg.logging.log_every_n_steps,
        )

        # 开始训练
        trainer.fit(model, data_module)

        # 返回训练结果
        return {
            "status": "success",
            "best_model_path": trainer.checkpoint_callback.best_model_path,
            "best_model_score": trainer.checkpoint_callback.best_model_score.item(),
            "final_epoch": trainer.current_epoch,
        }

    def _create_callbacks(self, cfg: DictConfig) -> List[pl.Callback]:
        """创建训练回调函数"""
        callbacks = []

        # 模型检查点
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.callbacks.model_checkpoint.monitor,
            mode=cfg.callbacks.model_checkpoint.mode,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            filename=cfg.callbacks.model_checkpoint.filename,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # 早停
        if cfg.callbacks.early_stopping.enable:
            early_stopping = EarlyStopping(
                monitor=cfg.callbacks.early_stopping.monitor,
                patience=cfg.callbacks.early_stopping.patience,
                mode=cfg.callbacks.early_stopping.mode,
            )
            callbacks.append(early_stopping)

        # 学习率监控
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        return callbacks

    def _create_loggers(self, cfg: DictConfig) -> List[pl.LightningLoggerBase]:
        """创建日志记录器"""
        loggers = []

        # TensorBoard日志
        if cfg.logging.tensorboard.enable:
            tb_logger = TensorBoardLogger(
                save_dir=cfg.logging.save_dir, name=cfg.logging.name, version=cfg.experiment.name
            )
            loggers.append(tb_logger)

        return loggers


class TestTaskRunner(TaskRunner):
    """
    测试任务执行器

    负责对训练好的模型进行测试评估。
    """

    def run(self) -> Dict[str, Any]:
        """执行测试任务"""
        logger.info("🧪 Starting test task")

        if not hasattr(self.args, "checkpoint") or not self.args.checkpoint:
            raise ValueError("Test task requires --checkpoint argument")

        checkpoint_path = Path(self.args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 从检查点加载模型和配置
        from src.models import LandslideClassificationModule
        from src.data import MultiModalDataModule

        logger.info(f"Loading model from {checkpoint_path}")
        model = LandslideClassificationModule.load_from_checkpoint(checkpoint_path)

        # 创建数据模块（使用保存的配置或新配置）
        if hasattr(self.args, "config_path") and self.args.config_path:
            cfg = self.load_config(self.args.config_path, self.args.config_name)
        else:
            # 尝试从检查点中恢复配置
            cfg = self._extract_config_from_checkpoint(checkpoint_path)

        data_module = MultiModalDataModule(cfg)

        # 创建测试器
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=False  # 测试时不需要日志
        )

        # 执行测试
        test_results = trainer.test(model, data_module)

        logger.info("✅ Test completed")
        return {"status": "success", "test_results": test_results, "checkpoint_path": str(checkpoint_path)}

    def _extract_config_from_checkpoint(self, checkpoint_path: Path) -> DictConfig:
        """从检查点中提取配置信息"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "hyper_parameters" in checkpoint:
            # Lightning自动保存的超参数
            return OmegaConf.create(checkpoint["hyper_parameters"])
        else:
            # 使用默认配置
            logger.warning("No configuration found in checkpoint, using default")
            return self._load_default_config()


class PredictTaskRunner(TaskRunner):
    """
    推理任务执行器

    处理批量推理任务，生成预测结果。
    """

    def run(self) -> Dict[str, Any]:
        """执行推理任务"""
        logger.info("🔮 Starting prediction task")

        if not hasattr(self.args, "checkpoint") or not self.args.checkpoint:
            raise ValueError("Predict task requires --checkpoint argument")

        if not hasattr(self.args, "input_dir") or not self.args.input_dir:
            raise ValueError("Predict task requires --input-dir argument")

        # 导入必要的模块
        from src.models import LandslideClassificationModule

        # 加载模型
        logger.info(f"Loading model from {self.args.checkpoint}")
        model = LandslideClassificationModule.load_from_checkpoint(self.args.checkpoint)
        model.eval()

        # 设置输出目录
        output_dir = Path(getattr(self.args, "output_dir", "predictions"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 执行批量推理
        predictions = self._run_batch_inference(model, self.args.input_dir, output_dir)

        logger.info(f"✅ Prediction completed. Results saved to {output_dir}")
        return {"status": "success", "predictions_count": len(predictions), "output_dir": str(output_dir)}

    def _run_batch_inference(self, model, input_dir: str, output_dir: Path) -> List[Dict]:
        """执行批量推理"""
        input_path = Path(input_dir)
        predictions = []

        # 获取所有输入文件
        data_files = list(input_path.glob("*.npy"))
        logger.info(f"Found {len(data_files)} files for prediction")

        # 逐文件进行推理
        for data_file in data_files:
            try:
                # 加载数据
                data = torch.from_numpy(np.load(data_file)).float().unsqueeze(0)

                # 执行推理
                with torch.no_grad():
                    logits = model(data)
                    prob = torch.sigmoid(logits).item()
                    pred = int(prob > 0.5)

                # 记录结果
                result = {"file_id": data_file.stem, "probability": prob, "prediction": pred}
                predictions.append(result)

            except Exception as e:
                logger.error(f"Failed to process {data_file}: {e}")

        # 保存结果
        self._save_predictions(predictions, output_dir)
        return predictions

    def _save_predictions(self, predictions: List[Dict], output_dir: Path):
        """保存预测结果"""
        import pandas as pd

        # 创建DataFrame
        df = pd.DataFrame(predictions)

        # 保存为CSV
        csv_path = output_dir / "predictions.csv"
        df.to_csv(csv_path, index=False)

        # 创建提交格式文件
        submission_df = df[["file_id", "prediction"]].copy()
        submission_df.columns = ["ID", "label"]
        submission_path = output_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False)

        logger.info(f"Saved predictions to {csv_path}")
        logger.info(f"Saved submission format to {submission_path}")


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    这个函数定义了统一入口点的完整命令行接口。
    设计上参考了git等成功工具的子命令模式。
    """
    parser = argparse.ArgumentParser(
        description="MM-LandslideNet: Multi-modal Landslide Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基础训练
  python main.py train --config-path configs/experiment --config-name optical_baseline
  
  # 使用预设快速训练
  python main.py train --preset quick_test
  
  # 训练时覆盖参数
  python main.py train --preset quick_test --override training.max_epochs=20 --override data.batch_size=32
  
  # 测试模型
  python main.py test --checkpoint experiments/optical_baseline/checkpoints/best.ckpt
  
  # 批量推理
  python main.py predict --checkpoint best.ckpt --input-dir test_data/ --output-dir results/
  
  # 快速帮助
  python main.py --help
  python main.py train --help
        """,
    )

    # 全局参数
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # 创建子命令
    subparsers = parser.add_subparsers(dest="task", help="Task to execute")

    # 训练子命令
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config-path", type=str, help="Path to config directory")
    train_parser.add_argument("--config-name", type=str, help="Config file name (without .yaml)")
    train_parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "full_multimodal", "high_performance"],
        help="Use predefined configuration preset",
    )
    train_parser.add_argument(
        "--override", action="append", dest="overrides", help="Override config parameters (e.g., training.lr=0.01)"
    )

    # 测试子命令
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    test_parser.add_argument("--config-path", type=str, help="Override config path")
    test_parser.add_argument("--config-name", type=str, help="Override config name")

    # 推理子命令
    predict_parser = subparsers.add_parser("predict", help="Run batch prediction")
    predict_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    predict_parser.add_argument("--input-dir", required=True, help="Directory containing input data")
    predict_parser.add_argument("--output-dir", default="predictions", help="Output directory for results")

    return parser


def main():
    """
    主函数：项目的统一入口点

    这个函数实现了整个框架的核心调度逻辑。它解析用户的命令，
    创建相应的任务执行器，然后运行任务并处理结果。
    """
    parser = create_parser()
    args = parser.parse_args()

    # 如果没有指定任务，显示帮助信息
    if args.task is None:
        parser.print_help()
        sys.exit(1)

    # 创建任务执行器
    task_runners = {"train": TrainTaskRunner, "test": TestTaskRunner, "predict": PredictTaskRunner}

    if args.task not in task_runners:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)

    # 执行任务
    try:
        runner = task_runners[args.task](args)
        results = runner.run()

        # 显示执行结果
        print("\n" + "=" * 60)
        print(f"🎉 Task '{args.task}' completed successfully!")
        print("=" * 60)

        for key, value in results.items():
            if key != "status":
                print(f"{key}: {value}")

        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Task '{args.task}' failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
