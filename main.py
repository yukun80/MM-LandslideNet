#!/usr/bin/env python3
"""
MM-LandslideNet 统一项目入口点 (重构版)

这是参考latent-diffusion设计的新版本入口点。核心思想是"配置驱动一切"：
- 所有组件都通过配置文件创建
- 支持多种任务（训练、测试、推理等）
- 极简的代码逻辑，最大的灵活性

设计哲学：
"让配置文件成为唯一的变化点" - 添加新模型、新数据集或新训练策略时，
只需要编写配置文件，无需修改任何Python代码。

教学要点：
对比您原来的main.py，新版本的核心改进是用"配置驱动"替代了"代码驱动"。
这种设计让框架具备了类似latent-diffusion的强大灵活性。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 抑制不必要的警告
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The dataloader.*")

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# 导入我们的核心工具
from lightning_landslide.src.utils.instantiate import instantiate_from_config, validate_config_structure
from lightning_landslide.src.utils.logging_utils import setup_logging, get_project_logger

logger = get_project_logger(__name__)


class ExperimentRunner:
    """
    实验运行器

    这个类负责协调整个实验的执行流程。它就像一个指挥家，
    根据配置文件的"乐谱"来指挥各个组件协同工作。

    与您原来的TaskRunner相比，这个版本更加专注和简化：
    - 只有一个核心职责：运行实验
    - 所有的复杂性都封装在配置文件中
    - 代码逻辑变得极其简洁
    """

    def __init__(self, config_path: str, task: str = "train"):
        """
        初始化实验运行器

        Args:
            config_path: 配置文件路径
            task: 要执行的任务类型（train/test/predict等）
        """
        self.config_path = Path(config_path)
        self.task = task
        self.config = self._load_config()
        self._setup_environment()

    def _load_config(self) -> DictConfig:
        """
        加载和验证配置文件

        这里我们做两件事：
        1. 加载YAML配置文件
        2. 验证配置的基本结构
        """
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
        """
        设置实验环境

        包括日志、随机种子、输出目录等基础设施。
        """
        # 设置日志
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        setup_logging(level=log_level)

        # 设置随机种子
        if "seed" in self.config:
            pl.seed_everything(self.config.seed, workers=True)
            logger.info(f"Set random seed to {self.config.seed}")

        # 创建输出目录
        self._create_output_dirs()

        # 保存配置文件到实验目录
        self._save_config()

    def _create_output_dirs(self):
        """创建实验需要的输出目录"""
        if "outputs" in self.config:
            for dir_name, dir_path in self.config.outputs.items():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")

    def _save_config(self):
        """保存配置文件到实验目录（确保可重现性）"""
        if "outputs" in self.config and "log_dir" in self.config.outputs:
            config_save_path = Path(self.config.outputs.log_dir) / "config.yaml"
            config_save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_save_path, "w") as f:
                OmegaConf.save(self.config, f)

            logger.info(f"Config saved to: {config_save_path}")

    def run(self) -> Dict[str, Any]:
        """
        运行实验的主方法

        这是整个框架的核心。它根据任务类型调用相应的执行方法。
        注意这里的代码有多么简洁 - 所有的复杂性都被配置文件吸收了。

        Returns:
            实验结果字典
        """
        logger.info(f"🚀 Starting {self.task} task")
        self._print_experiment_info()

        # 根据任务类型分发到不同的执行方法
        task_methods = {
            "train": self._run_training,
            "test": self._run_testing,
            "predict": self._run_prediction,
            "validate": self._run_validation,
        }

        if self.task not in task_methods:
            raise ValueError(f"Unknown task: {self.task}. Available: {list(task_methods.keys())}")

        return task_methods[self.task]()

    def _run_training(self) -> Dict[str, Any]:
        """
        执行训练任务

        这是整个重构的核心成果展示。看看这个方法有多么简洁：
        - 从配置创建模型：一行代码
        - 从配置创建数据：一行代码
        - 从配置创建训练器：一行代码
        - 开始训练：一行代码

        这就是"配置驱动"设计的威力！
        """
        logger.info("Initializing training components...")

        # 🎯 核心改进：用配置创建所有组件
        # 不再需要复杂的工厂类或if-else判断
        model = instantiate_from_config(self.config.model)
        data_module = instantiate_from_config(self.config.data)

        # 处理trainer配置（可能包含callbacks和loggers）
        trainer_config = self.config.trainer.copy()

        # 创建callbacks（如果配置中有的话）
        if "callbacks" in self.config:
            callbacks = []
            for callback_name, callback_config in self.config.callbacks.items():
                callback = instantiate_from_config(callback_config)
                callbacks.append(callback)
                logger.info(f"Added callback: {callback_name}")
            trainer_config.params.callbacks = callbacks

        # 创建loggers（如果配置中有的话）
        if "loggers" in self.config:
            loggers = []
            for logger_name, logger_config in self.config.loggers.items():
                log_obj = instantiate_from_config(logger_config)
                loggers.append(log_obj)
                logger.info(f"Added logger: {logger_name}")
            trainer_config.params.logger = loggers

        # 创建训练器
        trainer = instantiate_from_config(trainer_config)

        logger.info("🎓 Starting training...")

        # 开始训练 - 就是这么简单！
        trainer.fit(model, data_module)

        # 返回训练结果
        return {
            "status": "completed",
            "best_model_path": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None,
            "trainer": trainer,
        }

    def _run_testing(self) -> Dict[str, Any]:
        """
        执行测试任务

        测试任务的核心目标是评估已训练模型的性能。它加载保存的
        检查点，在测试集上运行模型，并生成详细的性能报告。

        Returns:
            包含测试结果和相关文件路径的字典
        """
        logger.info("🧪 Initializing testing task...")

        # 验证必需的配置
        if "checkpoint_path" not in self.config:
            raise ValueError("Testing requires 'checkpoint_path' in config")

        checkpoint_path = self.config.checkpoint_path
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # 创建组件
        logger.info("Creating model and data module...")
        model = instantiate_from_config(self.config.model)
        data_module = instantiate_from_config(self.config.data)

        # 为测试任务调整trainer配置
        trainer_config = self.config.trainer.copy()
        trainer_config.params.update(
            {
                "logger": False,  # 测试时不需要日志记录
                "enable_checkpointing": False,  # 测试时不保存检查点
                "enable_progress_bar": True,  # 显示测试进度
            }
        )

        trainer = instantiate_from_config(trainer_config)

        # 运行测试
        logger.info("🎯 Running model testing...")
        test_results = trainer.test(model, data_module, ckpt_path=checkpoint_path)

        # 保存测试结果
        results_file = self._save_test_results(test_results)

        logger.info("✅ Testing completed successfully!")
        return {
            "status": "completed",
            "test_results": test_results,
            "results_file": results_file,
            "checkpoint_used": checkpoint_path,
        }

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

    def _run_validation(self) -> Dict[str, Any]:
        """
        执行验证任务

        验证任务用于在验证集上评估模型性能，通常用于：
        1. 模型开发过程中的快速性能检查
        2. 超参数调优
        3. 模型选择和比较

        Returns:
            包含验证结果的字典
        """
        logger.info("🔍 Initializing validation task...")

        # 创建组件
        model = instantiate_from_config(self.config.model)
        data_module = instantiate_from_config(self.config.data)

        # 配置trainer（验证任务通常比较轻量）
        trainer_config = self.config.trainer.copy()
        trainer_config.params.update(
            {
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": True,
            }
        )
        trainer = instantiate_from_config(trainer_config)

        # 检查是否指定了检查点
        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path:
            logger.info(f"Using checkpoint: {checkpoint_path}")
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 运行验证
        logger.info("🎯 Running validation...")
        val_results = trainer.validate(model, data_module, ckpt_path=checkpoint_path)

        # 保存验证结果
        results_file = self._save_validation_results(val_results)

        logger.info("✅ Validation completed successfully!")
        return {
            "status": "completed",
            "validation_results": val_results,
            "results_file": results_file,
            "checkpoint_used": checkpoint_path,
        }

    def _save_test_results(self, test_results: List[Dict]) -> Path:
        """
        保存测试结果到文件

        Args:
            test_results: Lightning trainer.test()的返回结果

        Returns:
            保存的结果文件路径
        """
        import json
        from datetime import datetime

        # 创建输出目录
        output_dir = Path(self.config.outputs.get("predictions_dir", "outputs/test_results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{self.config.experiment_name}_{timestamp}.json"
        results_file = output_dir / filename

        # 保存结果
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_name": self.config.experiment_name,
                    "timestamp": timestamp,
                    "checkpoint_path": self.config.get("checkpoint_path"),
                    "test_results": test_results,
                    "config_summary": {
                        "model_type": self.config.model.target.split(".")[-1],
                        "data_config": self.config.data.target.split(".")[-1],
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"Test results saved to: {results_file}")
        return results_file

    def _save_validation_results(self, val_results: List[Dict]) -> Path:
        """
        保存验证结果到文件

        Args:
            val_results: Lightning trainer.validate()的返回结果

        Returns:
            保存的结果文件路径
        """
        import json
        from datetime import datetime

        # 创建输出目录
        output_dir = Path(self.config.outputs.get("predictions_dir", "outputs/validation_results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{self.config.experiment_name}_{timestamp}.json"
        results_file = output_dir / filename

        # 保存结果
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_name": self.config.experiment_name,
                    "timestamp": timestamp,
                    "checkpoint_path": self.config.get("checkpoint_path"),
                    "validation_results": val_results,
                    "config_summary": {
                        "model_type": self.config.model.target.split(".")[-1],
                        "data_config": self.config.data.target.split(".")[-1],
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"Validation results saved to: {results_file}")
        return results_file

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
    """
    创建命令行参数解析器

    新版本的命令行接口更加简洁，重点突出配置文件的作用。
    """
    parser = argparse.ArgumentParser(
        description="MM-LandslideNet: Configuration-Driven Deep Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 训练模型
  python main.py train configs/experiment/optical_baseline.yaml
  
  # 测试模型  
  python main.py test configs/experiment/optical_baseline.yaml
  
  # 运行推理
  python main.py predict configs/experiment/optical_baseline.yaml
  
  # 验证模型
  python main.py validate configs/experiment/optical_baseline.yaml

Configuration-First Design:
  This framework follows the "configuration-first" principle inspired by 
  latent-diffusion. All model architectures, training strategies, and data 
  processing pipelines are defined in YAML configuration files, making the
  framework extremely flexible and maintainable.
        """,
    )

    # 主要参数
    parser.add_argument("task", choices=["train", "test", "predict", "validate"], help="Task to execute")

    parser.add_argument("config", type=str, help="Path to configuration file")

    # 可选参数
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (overrides config)")

    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (e.g., --override training.max_epochs=100 data.batch_size=32)",
    )

    return parser


def apply_overrides(config: DictConfig, overrides: list) -> DictConfig:
    """
    应用命令行覆盖

    允许用户在命令行中覆盖配置文件中的特定值。
    这在调试和快速实验时非常有用。

    Args:
        config: 原始配置
        overrides: 覆盖列表，格式为 ["key=value", "another.key=value"]

    Returns:
        修改后的配置
    """
    if not overrides:
        return config

    logger.info(f"Applying {len(overrides)} config overrides...")

    for override in overrides:
        try:
            key, value = override.split("=", 1)

            # 尝试自动类型转换
            try:
                # 处理数字
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit() and value.count(".") == 1:
                    value = float(value)
                # 处理布尔值
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                # 处理列表（简单情况）
                elif value.startswith("[") and value.endswith("]"):
                    value = eval(value)  # 注意：生产环境中应该用更安全的解析方法
            except:
                pass  # 保持为字符串

            OmegaConf.set(config, key, value)
            logger.info(f"  {key} = {value}")

        except Exception as e:
            logger.warning(f"Failed to apply override '{override}': {e}")

    return config


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
    parser = create_parser()
    args = parser.parse_args()

    try:
        # 创建实验运行器
        runner = ExperimentRunner(args.config, args.task)

        # 应用命令行覆盖（如果有的话）
        if args.override:
            runner.config = apply_overrides(runner.config, args.override)

        # 覆盖checkpoint路径（如果在命令行中指定）
        if args.checkpoint:
            runner.config.checkpoint_path = args.checkpoint
            logger.info(f"Using checkpoint: {args.checkpoint}")

        # 运行实验
        results = runner.run()

        # 报告结果
        print(f"\n🎉 Task '{args.task}' completed successfully!")
        if results.get("best_model_path"):
            print(f"📁 Best model saved to: {results['best_model_path']}")

        return 0

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
