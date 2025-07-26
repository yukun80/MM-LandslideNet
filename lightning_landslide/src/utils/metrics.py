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
import time
from datetime import datetime

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

    # def __init__(self, log_dir: Optional[str] = None):
    def __init__(
        self,
        log_dir: Optional[str] = None,
        summary_interval: int = 10,
        save_history: bool = True,
    ):
        """
        初始化MetricsLogger

        Args:
            log_dir: 自定义日志保存目录。如果为None，会尝试使用trainer.log_dir
        """
        super().__init__()
        self.custom_log_dir = Path(log_dir) if log_dir else None
        self.summary_interval = summary_interval
        self.save_history = save_history

        self.best_metrics = {}
        self.metrics_history = [] if self.save_history else None
        self.epoch_times = []
        self.training_start_time = None

        # 当前epoch数据
        self.current_epoch_start_time = None

    def on_train_start(self, trainer, pl_module):
        """训练epoch开始时记录时间"""
        self.training_start_time = time.time()
        logger.info("🚀 MetricsLogger: Training started")
        logger.info(f"📊 Will save metrics to: {self.custom_log_dir}")

    def on_train_epoch_start(self, trainer, pl_module):
        """训练epoch开始时记录时间"""
        self.current_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """训练epoch结束时的处理"""
        # 记录epoch耗时
        epoch_time = time.time() - self.current_epoch_start_time
        self.epoch_times.append(epoch_time)

        # 记录到日志
        pl_module.log("epoch_time_sec", epoch_time, on_epoch=True, logger=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时的处理"""

        current_metrics = self._extract_metrics_safely(trainer, pl_module)
        if current_metrics is None:
            logger.warning(f"Epoch {trainer.current_epoch}: Failed to extract metrics")
            return

        self._update_best_metrics(current_metrics, trainer.current_epoch)

        # 保存历史记录
        if self.save_history and self.metrics_history is not None:
            self.metrics_history.append(current_metrics)

        # 定期打印详细总结
        if trainer.current_epoch % self.summary_interval == 0:
            self._print_detailed_summary(trainer, current_metrics)

    def on_train_end(self, trainer, pl_module):
        """训练结束时的总结和保存"""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0

        # 打印最终总结
        self._print_final_summary(trainer, training_time)

        # 保存完整报告
        if self.save_history:
            self._save_experiment_report(trainer, training_time)

    def _extract_metrics_safely(self, trainer, pl_module) -> Optional[Dict[str, Any]]:
        """
        安全地从Lightning系统中提取指标

        Returns:
            提取的指标字典，如果失败返回None
        """
        try:
            logged_metrics = trainer.logged_metrics

            # 调试信息（仅在需要时开启）
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Available metric keys: {list(logged_metrics.keys())}")

            # 构建当前指标字典
            current_metrics = {
                "epoch": trainer.current_epoch,
                "timestamp": datetime.now().isoformat(),
            }

            # 关键指标映射 - 定义所有可能的键名
            metric_mappings = {
                "train_loss": ["train_loss_epoch", "train_loss", "loss_epoch", "loss"],
                "val_loss": ["val_loss"],
                "val_f1": ["val_f1"],
                "val_acc": ["val_acc", "val_accuracy"],
                "val_precision": ["val_precision"],
                "val_recall": ["val_recall"],
                "val_auroc": ["val_auroc"],
                "learning_rate": ["lr", "learning_rate"],
            }

            # 安全提取每个指标
            for metric_name, possible_keys in metric_mappings.items():
                value = self._safe_get_metric(logged_metrics, possible_keys)
                current_metrics[metric_name] = value

            # 添加epoch时间
            if self.epoch_times:
                current_metrics["epoch_time"] = self.epoch_times[-1]

            return current_metrics

        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return None

    def _safe_get_metric(self, metrics_dict: Dict, possible_keys: List[str]) -> Optional[float]:
        """
        安全获取指标值，处理多种可能的键名和数据类型

        Args:
            metrics_dict: Lightning的logged_metrics字典
            possible_keys: 可能的键名列表

        Returns:
            指标值（float）或None
        """
        for key in possible_keys:
            if key in metrics_dict:
                value = metrics_dict[key]

                # 处理tensor类型
                if isinstance(value, torch.Tensor):
                    return value.item()

                # 处理numpy类型
                elif hasattr(value, "item"):
                    return value.item()

                # 处理普通数值
                elif isinstance(value, (int, float)):
                    return float(value)

                # 其他类型
                else:
                    logger.warning(f"Unknown metric type for {key}: {type(value)}")
                    return None

        return None

    def _update_best_metrics(self, current_metrics: Dict[str, Any], current_epoch: int):
        """更新最佳指标跟踪"""
        val_f1 = current_metrics.get("val_f1")

        if val_f1 is not None:
            # 更新最佳F1
            if "best_val_f1" not in self.best_metrics or val_f1 > self.best_metrics["best_val_f1"]:
                self.best_metrics.update(
                    {
                        "best_val_f1": val_f1,
                        "best_val_f1_epoch": current_epoch,
                        "best_val_acc": current_metrics.get("val_acc"),
                        "best_val_auroc": current_metrics.get("val_auroc"),
                        "best_val_loss": current_metrics.get("val_loss"),
                        "best_timestamp": current_metrics.get("timestamp"),
                    }
                )

                logger.info(f"🎉 New best F1 score: {val_f1:.4f} at epoch {current_epoch}")

        # 更新最佳准确率（独立跟踪）
        val_acc = current_metrics.get("val_acc")
        if val_acc is not None:
            if "best_val_acc" not in self.best_metrics or val_acc > self.best_metrics.get("best_val_acc_standalone", 0):
                self.best_metrics["best_val_acc_standalone"] = val_acc
                self.best_metrics["best_val_acc_epoch"] = current_epoch

    def _print_detailed_summary(self, trainer, current_metrics: Dict[str, Any]):
        """打印详细的训练总结"""
        logger.info(f"\n{'='*100}")
        logger.info(f"📊 Epoch {trainer.current_epoch} Detailed Summary")
        logger.info(f"{'='*100}")

        # 当前指标
        train_loss = current_metrics.get("train_loss")
        val_loss = current_metrics.get("val_loss")
        val_f1 = current_metrics.get("val_f1")
        val_acc = current_metrics.get("val_acc")
        val_auroc = current_metrics.get("val_auroc")
        lr = current_metrics.get("learning_rate")

        logger.info(f"Current Metrics:")
        logger.info(f"  Train Loss: {train_loss:.4f}" if train_loss is not None else "  Train Loss: N/A")
        logger.info(f"  Val Loss:   {val_loss:.4f}" if val_loss is not None else "  Val Loss: N/A")
        logger.info(f"  Val F1:     {val_f1:.4f}" if val_f1 is not None else "  Val F1: N/A")
        logger.info(f"  Val Acc:    {val_acc:.4f}" if val_acc is not None else "  Val Acc: N/A")
        logger.info(f"  Val AUROC:  {val_auroc:.4f}" if val_auroc is not None else "  Val AUROC: N/A")
        logger.info(f"  Learning Rate: {lr:.6f}" if lr is not None else "  Learning Rate: N/A")

        # 最佳指标
        if self.best_metrics:
            logger.info(f"\nBest Metrics So Far:")
            logger.info(
                f"  Best Val F1: {self.best_metrics.get('best_val_f1', 'N/A'):.4f} "
                f"(Epoch {self.best_metrics.get('best_val_f1_epoch', 'N/A')})"
            )

            best_acc = self.best_metrics.get("best_val_acc_standalone")
            if best_acc is not None:
                logger.info(
                    f"  Best Val Acc: {best_acc:.4f} " f"(Epoch {self.best_metrics.get('best_val_acc_epoch', 'N/A')})"
                )

        # 训练效率统计
        if self.epoch_times:
            recent_times = self.epoch_times[-min(10, len(self.epoch_times)) :]
            avg_time = np.mean(recent_times)
            logger.info(f"\nTraining Efficiency:")
            logger.info(f"  Avg Epoch Time (last {len(recent_times)}): {avg_time:.2f}s")

            if len(self.epoch_times) > 1:
                total_time = sum(self.epoch_times)
                remaining_epochs = trainer.max_epochs - trainer.current_epoch - 1
                estimated_remaining = remaining_epochs * avg_time
                logger.info(f"  Estimated Remaining Time: {estimated_remaining/3600:.2f}h")

        logger.info(f"{'='*100}\n")

    def _print_final_summary(self, trainer, training_time: float):
        """打印最终训练总结"""
        logger.info(f"\n{'🎯'*50}")
        logger.info(f"🎯 TRAINING COMPLETED - FINAL SUMMARY")
        logger.info(f"{'🎯'*50}")

        logger.info(f"📈 Training Statistics:")
        logger.info(f"  Total Epochs: {trainer.current_epoch + 1}")
        logger.info(f"  Total Training Time: {training_time/3600:.2f}h")

        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            logger.info(f"  Average Epoch Time: {avg_epoch_time:.2f}s")

        # 最佳性能
        if self.best_metrics:
            logger.info(f"\n🏆 Best Performance Achieved:")
            best_f1 = self.best_metrics.get("best_val_f1")
            best_epoch = self.best_metrics.get("best_val_f1_epoch")

            if best_f1 is not None:
                logger.info(f"  🥇 Best F1 Score: {best_f1:.4f} at Epoch {best_epoch}")
                logger.info(f"     ├─ Validation Accuracy: {self.best_metrics.get('best_val_acc', 'N/A'):.4f}")
                logger.info(f"     ├─ Validation AUROC: {self.best_metrics.get('best_val_auroc', 'N/A'):.4f}")
                logger.info(f"     └─ Validation Loss: {self.best_metrics.get('best_val_loss', 'N/A'):.4f}")

        logger.info(f"{'🎯'*50}\n")

    def _save_experiment_report(self, trainer, training_time: float):
        """保存完整的实验报告"""
        try:
            # 确定保存目录
            save_dir = self._get_save_directory()
            report_file = save_dir / "experiment_report.json"

            # 构建完整报告
            report = {
                "experiment_info": {
                    "completed_at": datetime.now().isoformat(),
                    "total_epochs": trainer.current_epoch + 1,
                    "total_training_time_hours": training_time / 3600,
                    "average_epoch_time_seconds": np.mean(self.epoch_times) if self.epoch_times else 0,
                    "max_epochs_configured": trainer.max_epochs,
                },
                "best_metrics": self.best_metrics,
                "training_efficiency": {
                    "epoch_times": self.epoch_times,
                    "total_batches_processed": trainer.global_step,
                    "gpu_used": torch.cuda.is_available(),
                    "device_count": trainer.num_devices if hasattr(trainer, "num_devices") else 1,
                },
                "metrics_history": self.metrics_history if self.metrics_history else [],
                "model_info": {
                    "model_class": trainer.model.__class__.__name__,
                    "total_parameters": sum(p.numel() for p in trainer.model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                },
            }

            # 保存报告
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"📁 Experiment report saved: {report_file}")

        except Exception as e:
            logger.error(f"Failed to save experiment report: {e}")

    def _get_save_directory(self) -> Path:
        """获取保存目录"""
        if self.custom_log_dir:
            target_dir = self.custom_log_dir
        else:
            # 使用当前工作目录作为fallback
            target_dir = Path.cwd() / "logs"

        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir
