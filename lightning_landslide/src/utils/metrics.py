# =============================================================================
# src/utils/metrics.py - 自定义指标和回调
# =============================================================================

"""
自定义指标和回调函数

这个模块提供了额外的指标计算和回调功能，补充Lightning内置的功能。
特别针对滑坡检测任务的特殊需求进行了优化。
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsLogger(Callback):
    """
    自定义指标日志回调

    这个回调扩展了Lightning的日志功能，提供：
    1. 更详细的指标记录
    2. 实时性能监控
    3. 自动的模型性能分析
    4. 训练过程的统计信息
    """

    def __init__(self):
        super().__init__()
        self.metrics_history = []
        self.epoch_times = []
        self.best_metrics = {}

    def on_train_epoch_start(self, trainer, pl_module):
        """训练epoch开始时记录时间"""
        self.epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.epoch_start_time:
            self.epoch_start_time.record()

    def on_train_epoch_end(self, trainer, pl_module):
        """训练epoch结束时的处理"""
        # 记录epoch耗时
        if self.epoch_start_time and torch.cuda.is_available():
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = self.epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # 转换为秒
            self.epoch_times.append(epoch_time)

            # 记录到日志
            pl_module.log("epoch_time_sec", epoch_time, on_epoch=True, logger=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时的处理"""
        # 收集当前epoch的指标
        current_metrics = {
            "epoch": trainer.current_epoch,
            "train_loss": trainer.logged_metrics.get("train_loss_epoch", 0),
            "val_loss": trainer.logged_metrics.get("val_loss", 0),
            "val_f1": trainer.logged_metrics.get("val_f1", 0),
            "val_acc": trainer.logged_metrics.get("val_acc", 0),
            "val_auroc": trainer.logged_metrics.get("val_auroc", 0),
            "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
        }

        # 将trainer.logged_metrics转换为字典
        for key, value in trainer.logged_metrics.items():
            if isinstance(value, torch.Tensor):
                current_metrics[key] = value.item()  # 转换tensor为float
            elif hasattr(value, "item"):  # 处理numpy等其他数值类型
                current_metrics[key] = value.item()
            else:
                current_metrics[key] = value

        self.metrics_history.append(current_metrics)

        # 更新最佳指标
        val_f1 = current_metrics["val_f1"]
        if isinstance(val_f1, torch.Tensor):
            val_f1 = val_f1.item()

        if "best_val_f1" not in self.best_metrics or val_f1 > self.best_metrics["best_val_f1"]:
            self.best_metrics.update(
                {
                    "best_val_f1": val_f1,
                    "best_val_f1_epoch": trainer.current_epoch,
                    "best_val_acc": current_metrics["val_acc"],
                    "best_val_auroc": current_metrics["val_auroc"],
                }
            )

        # 每10个epoch打印一次详细信息
        if trainer.current_epoch % 10 == 0:
            self._log_detailed_metrics(trainer, current_metrics)

    def _log_detailed_metrics(self, trainer, current_metrics):
        """记录详细的指标信息"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {trainer.current_epoch} Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"Train Loss: {current_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss:   {current_metrics['val_loss']:.4f}")
        logger.info(f"Val F1:     {current_metrics['val_f1']:.4f}")
        logger.info(f"Val Acc:    {current_metrics['val_acc']:.4f}")
        logger.info(f"Val AUROC:  {current_metrics['val_auroc']:.4f}")
        logger.info(f"LR:         {current_metrics['learning_rate']:.6f}")

        if self.best_metrics:
            logger.info(f"\nBest Metrics So Far:")
            logger.info(
                f"Best Val F1: {self.best_metrics['best_val_f1']:.4f} (Epoch {self.best_metrics['best_val_f1_epoch']})"
            )

        if self.epoch_times:
            avg_time = np.mean(self.epoch_times[-10:])  # 最近10个epoch的平均时间
            logger.info(f"Avg Epoch Time: {avg_time:.2f}s")

        logger.info(f"{'='*60}\n")

    def on_train_end(self, trainer, pl_module):
        """训练结束时保存完整的指标历史"""
        if hasattr(trainer, "log_dir") and trainer.log_dir:
            metrics_file = Path(trainer.log_dir) / "metrics_history.json"

            # 转换tensor为float
            history_serializable = []
            for epoch_metrics in self.metrics_history:
                epoch_data = {}
                for key, value in epoch_metrics.items():
                    if isinstance(value, torch.Tensor):
                        epoch_data[key] = value.item()
                    else:
                        epoch_data[key] = value
                history_serializable.append(epoch_data)

            # 保存到文件
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "metrics_history": history_serializable,
                        "best_metrics": self.best_metrics,
                        "total_epochs": trainer.current_epoch + 1,
                        "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Metrics history saved to {metrics_file}")
