#!/usr/bin/env python3
"""
PyTorch Lightning训练脚本

这是整个框架的入口点。它将所有组件（模型、数据、训练逻辑）
整合在一起，提供统一的训练接口。

核心功能：
1. 配置管理：使用Hydra进行配置管理
2. 实验追踪：集成多种日志系统
3. 错误处理：优雅的错误处理和恢复
4. 自动化流程：从训练到测试的完整自动化

使用示例：
1. 基础训练：python train.py
2. 修改参数：python train.py training.max_epochs=100 data.batch_size=64
3. 使用不同配置：python train.py --config-name=optical_baseline
4. 多实验运行：python train.py --multirun training.lr=1e-4,5e-5,2e-4

设计哲学：
"让复杂的事情变简单，让简单的事情变自动化"
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# 导入我们的模块
from src.models import LandslideClassificationModule
from src.data import OpticalDataModule
from src.utils.metrics import MetricsLogger
from src.utils.logging_utils import setup_logging

# 抑制一些不重要的警告
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The dataloader.*")

logger = logging.getLogger(__name__)


def create_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """
    创建训练回调函数

    回调函数是Lightning的核心功能之一，它们在训练的不同阶段
    自动执行特定的操作，如保存检查点、早停、学习率监控等。

    Args:
        cfg: 配置对象

    Returns:
        回调函数列表
    """
    callbacks = []

    # 早停回调
    if cfg.callbacks.early_stopping.enable:
        early_stop = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            mode=cfg.callbacks.early_stopping.mode,
            patience=cfg.callbacks.early_stopping.patience,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            verbose=cfg.callbacks.early_stopping.verbose,
        )
        callbacks.append(early_stop)
        logger.info(f"Added EarlyStopping: monitor={cfg.callbacks.early_stopping.monitor}")

    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        save_last=cfg.callbacks.model_checkpoint.save_last,
        filename=cfg.callbacks.model_checkpoint.filename,
        auto_insert_metric_name=cfg.callbacks.model_checkpoint.auto_insert_metric_name,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"Added ModelCheckpoint: monitor={cfg.callbacks.model_checkpoint.monitor}")

    # 学习率监控回调
    lr_monitor = LearningRateMonitor(logging_interval=cfg.callbacks.lr_monitor.logging_interval)
    callbacks.append(lr_monitor)

    # 自定义指标日志回调
    metrics_logger = MetricsLogger()
    callbacks.append(metrics_logger)

    logger.info(f"Created {len(callbacks)} callbacks")
    return callbacks


def create_loggers(cfg: DictConfig) -> List[pl.LightningLoggerBase]:
    """
    创建日志记录器

    支持多种日志后端，包括TensorBoard和Weights & Biases。
    这些日志系统帮助我们追踪实验进展，可视化训练过程。

    Args:
        cfg: 配置对象

    Returns:
        日志记录器列表
    """
    loggers = []

    # TensorBoard日志记录器
    if cfg.logging.tensorboard.enable:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.name,
            version=cfg.experiment.name,
            log_graph=cfg.logging.tensorboard.log_graph,
        )
        loggers.append(tb_logger)
        logger.info(f"Added TensorBoard logger: {tb_logger.log_dir}")

    # Weights & Biases日志记录器
    if cfg.logging.wandb.enable:
        try:
            wandb_logger = WandbLogger(
                project=cfg.logging.wandb.project,
                name=cfg.experiment.name,
                tags=cfg.logging.wandb.tags,
                notes=cfg.logging.wandb.notes,
                save_dir=cfg.logging.save_dir,
            )
            loggers.append(wandb_logger)
            logger.info("Added Weights & Biases logger")
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logger")

    if not loggers:
        logger.warning("No loggers configured, using default Lightning logger")

    return loggers


def setup_environment(cfg: DictConfig) -> None:
    """
    设置训练环境

    配置各种环境变量和PyTorch设置，确保训练的稳定性和可重现性。

    Args:
        cfg: 配置对象
    """
    # 设置随机种子
    if cfg.reproducibility.seed is not None:
        pl.seed_everything(cfg.reproducibility.seed, workers=True)
        logger.info(f"Set random seed to {cfg.reproducibility.seed}")

    # 设置PyTorch性能选项
    if cfg.reproducibility.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Enabled deterministic mode")
    elif cfg.reproducibility.benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode")

    # 设置线程数（避免过度并行化）
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(4)
        os.environ["OMP_NUM_THREADS"] = "4"

    # 创建输出目录
    for dir_path in [
        cfg.outputs.checkpoint_dir,
        cfg.outputs.log_dir,
        cfg.outputs.results_dir,
        cfg.outputs.predictions_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("Environment setup completed")


def validate_config(cfg: DictConfig) -> None:
    """
    验证配置的有效性

    在开始训练前检查配置的合理性，避免浪费时间在无效的实验上。

    Args:
        cfg: 配置对象

    Raises:
        ValueError: 如果配置无效
    """
    # 验证必需的配置项
    required_keys = [
        "model.type",
        "model.num_classes",
        "data.train_data_dir",
        "data.train_csv",
        "training.max_epochs",
        "training.optimizer.type",
    ]

    missing_keys = []
    for key in required_keys:
        try:
            OmegaConf.select(cfg, key)
        except:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # 验证数据路径
    data_paths = [cfg.data.train_data_dir, cfg.data.train_csv]
    if cfg.data.get("exclude_ids_file"):
        data_paths.append(cfg.data.exclude_ids_file)

    for path in data_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Data path not found: {path}")

    # 验证训练参数
    if cfg.training.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    if cfg.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if cfg.training.optimizer.lr <= 0:
        raise ValueError("learning rate must be positive")

    # 验证GPU配置
    if cfg.compute.accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU requested but not available, falling back to CPU")
        cfg.compute.accelerator = "cpu"
        cfg.compute.devices = 1

    logger.info("✓ Configuration validation passed")


def print_experiment_info(cfg: DictConfig) -> None:
    """
    打印实验信息

    在训练开始前显示关键的实验信息，便于追踪和调试。

    Args:
        cfg: 配置对象
    """
    print("\n" + "=" * 80)
    print(f"🚀 Starting Experiment: {cfg.experiment.name}")
    print("=" * 80)
    print(f"📝 Description: {cfg.experiment.description}")
    print(f"🏷️  Tags: {cfg.experiment.tags}")
    print(f"📅 Version: {cfg.experiment.version}")
    print()
    print(f"🧠 Model: {cfg.model.type}")
    print(f"📊 Data: {cfg.data.train_data_dir}")
    print(f"⚙️  Batch Size: {cfg.data.batch_size}")
    print(f"🔄 Max Epochs: {cfg.training.max_epochs}")
    print(f"📈 Learning Rate: {cfg.training.optimizer.lr}")
    print(f"💾 Precision: {cfg.compute.precision}")
    print(f"🎯 Monitor Metric: {cfg.callbacks.model_checkpoint.monitor}")
    print()
    print(f"💾 Checkpoints: {cfg.outputs.checkpoint_dir}")
    print(f"📋 Logs: {cfg.outputs.log_dir}")
    print("=" * 80 + "\n")


def save_config(cfg: DictConfig, save_dir: str) -> None:
    """
    保存配置文件

    将完整的配置保存到实验目录，确保实验的可重现性。

    Args:
        cfg: 配置对象
        save_dir: 保存目录
    """
    config_save_path = Path(save_dir) / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)

    logger.info(f"Configuration saved to {config_save_path}")


def train_model(cfg: DictConfig) -> Dict[str, Any]:
    """
    训练模型的主要函数

    这是整个训练流程的核心函数，组织和执行完整的训练过程。

    Args:
        cfg: 配置对象

    Returns:
        训练结果字典
    """
    logger.info("Starting model training...")

    # 创建数据模块
    logger.info("Creating data module...")
    data_module = OpticalDataModule(cfg)

    # 创建模型
    logger.info("Creating model...")
    model = LandslideClassificationModule(cfg)

    # 创建回调函数和日志记录器
    callbacks = create_callbacks(cfg)
    loggers = create_loggers(cfg)

    # 创建训练器
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        # 基础配置
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.devices,
        precision=cfg.compute.precision,
        # 回调和日志
        callbacks=callbacks,
        logger=loggers,
        # 验证配置
        val_check_interval=cfg.evaluation.val_check_interval,
        # 性能优化
        gradient_clip_val=cfg.compute.gradient_clip_val,
        accumulate_grad_batches=cfg.compute.accumulate_grad_batches,
        # 可重现性
        deterministic=cfg.reproducibility.deterministic,
        # 日志配置
        log_every_n_steps=cfg.logging.log_every_n_steps,
        # 其他设置
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # 打印模型和数据信息
    logger.info("Model and data information:")
    data_module.setup("fit")
    model_info = model.model.get_model_info()
    data_info = data_module.get_data_info()

    for key, value in model_info.items():
        logger.info(f"  Model {key}: {value}")

    for key, value in data_info.items():
        logger.info(f"  Data {key}: {value}")

    # 开始训练
    logger.info("🚀 Starting training...")
    try:
        trainer.fit(model, data_module)
        logger.info("✅ Training completed successfully")

        # 自动测试（如果配置启用）
        if cfg.evaluation.test_after_training and data_module.test_dataloader() is not None:
            logger.info("🧪 Starting automatic testing...")
            test_results = trainer.test(model, data_module, ckpt_path="best")
            logger.info("✅ Testing completed")
        else:
            test_results = None
            logger.info("⏭️  Skipping automatic testing")

        # 收集训练结果
        training_results = {
            "status": "success",
            "best_model_path": trainer.checkpoint_callback.best_model_path,
            "best_model_score": trainer.checkpoint_callback.best_model_score.item(),
            "logged_metrics": trainer.logged_metrics,
            "test_results": test_results,
        }

        # 检查性能阈值
        best_score = trainer.checkpoint_callback.best_model_score.item()
        min_threshold = cfg.evaluation.performance_thresholds.min_val_f1
        target_threshold = cfg.evaluation.performance_thresholds.target_val_f1

        if best_score >= target_threshold:
            logger.info(f"🎉 Excellent! Achieved target performance: {best_score:.4f} >= {target_threshold}")
        elif best_score >= min_threshold:
            logger.info(f"✅ Good! Achieved minimum performance: {best_score:.4f} >= {min_threshold}")
        else:
            logger.warning(f"⚠️  Performance below minimum threshold: {best_score:.4f} < {min_threshold}")

        return training_results

    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise


def main(cfg: DictConfig) -> None:
    """
    主函数

    整个训练流程的入口点。处理配置、环境设置、训练执行和结果保存。

    Args:
        cfg: Hydra配置对象
    """
    try:
        # 设置日志
        setup_logging(level=logging.INFO)

        # 验证配置
        validate_config(cfg)

        # 设置环境
        setup_environment(cfg)

        # 打印实验信息
        print_experiment_info(cfg)

        # 保存配置
        save_config(cfg, cfg.outputs.log_dir)

        # 训练模型
        results = train_model(cfg)

        # 保存结果
        results_path = Path(cfg.outputs.results_dir) / "training_results.yaml"
        with open(results_path, "w") as f:
            OmegaConf.save(OmegaConf.create(results), f)

        logger.info(f"Results saved to {results_path}")

        # 成功完成
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Best model: {results['best_model_path']}")
        print(f"📊 Best score: {results['best_model_score']:.4f}")
        print(f"💾 Results: {results_path}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("Please check the logs for more details.")
        print("=" * 80 + "\n")
        raise


@hydra.main(version_base=None, config_path="configs/experiment", config_name="optical_baseline")
def hydra_main(cfg: DictConfig) -> None:
    """
    Hydra装饰的主函数

    Hydra提供了强大的配置管理功能，包括：
    - 配置文件组合
    - 命令行参数覆盖
    - 多实验运行
    - 自动化实验目录管理

    Args:
        cfg: Hydra处理后的配置对象
    """
    main(cfg)


if __name__ == "__main__":
    # 启动训练
    hydra_main()
