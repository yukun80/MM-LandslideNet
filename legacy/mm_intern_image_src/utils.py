"""
Utility Functions for MM-InternImage-TNF

This module contains reusable utility functions including loss functions,
evaluation metrics, checkpoint management, and reproducibility utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import logging
import torch.nn.functional as F

"""python -m mm_intern_image_src.utils"""

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.

    Reference: Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
            pos_weight: A weight of positive examples. If provided, this will be used as the 'pos_weight' argument in BCEWithLogitsLoss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.

        Args:
            inputs: Predicted logits [N, 1] or [N]
            targets: Ground truth labels [N, 1] or [N] (0 or 1)

        Returns:
            Focal loss value
        """
        # FIXED: Ensure both inputs and targets are properly shaped and consistent
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (B, 1) -> (B,)

        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # (B, 1) -> (B,)

        # Ensure targets are float type
        targets = targets.float()

        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=self.pos_weight)

        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation/classification.

    Dice loss is particularly effective for imbalanced datasets and
    optimizes for overlap between predicted and true regions.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Dice Loss.

        Args:
            inputs: Predicted logits [N, 1] or [N]
            targets: Ground truth labels [N] (0 or 1)

        Returns:
            Dice loss value
        """
        # Ensure inputs are properly shaped
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # Dice loss is 1 - Dice coefficient
        dice_loss = 1 - dice_coeff

        if self.reduction == "mean":
            return dice_loss
        elif self.reduction == "sum":
            return dice_loss * len(probs)
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that sums Focal Loss and Dice Loss.

    This combination leverages the benefits of both loss functions:
    - Focal Loss: Handles class imbalance and focuses on hard examples
    - Dice Loss: Optimizes for overlap and handles imbalanced data well
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Combined Loss.

        Args:
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            dice_smooth: Smoothing parameter for dice loss
            pos_weight: A weight of positive examples for Focal Loss.
        """
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Combined Loss.

        Args:
            inputs: Predicted logits
            targets: Ground truth labels

        Returns:
            Dictionary containing individual and total losses
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        total = self.focal_weight * focal + self.dice_weight * dice

        return {"focal_loss": focal, "dice_loss": dice, "total_loss": total}


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate classification metrics with enhanced error handling.

    Args:
        predictions: Probability predictions (continuous values 0-1)
        labels: True binary labels (0 or 1)
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary containing classification metrics
    """
    try:
        # 数据验证
        print(f"Debug calculate_metrics:")
        print(f"  - predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"  - labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"  - predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  - labels unique values: {np.unique(labels)}")

        # 确保输入是1D数组
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if labels.ndim > 1:
            labels = labels.flatten()

        # 确保labels是整数类型
        labels = labels.astype(int)

        # 检查标签值是否有效
        unique_labels = np.unique(labels)
        if not all(label in [0, 1] for label in unique_labels):
            print(f"⚠️ Warning: Invalid label values found: {unique_labels}")
            # 将标签规范化为0和1
            labels = (labels > 0.5).astype(int)

        # 确保predictions是浮点数类型且在[0,1]范围内
        predictions = np.clip(predictions.astype(float), 0.0, 1.0)

        # Convert probabilities to binary predictions
        binary_preds = (predictions >= threshold).astype(int)

        # Calculate metrics with proper error handling
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        metrics = {}

        # Basic accuracy
        metrics["accuracy"] = accuracy_score(labels, binary_preds)

        # Classification metrics with zero_division handling
        metrics["f1_score"] = f1_score(labels, binary_preds, zero_division=0)
        metrics["precision"] = precision_score(labels, binary_preds, zero_division=0)
        metrics["recall"] = recall_score(labels, binary_preds, zero_division=0)

        # AUC calculation
        if len(np.unique(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, predictions)
        else:
            metrics["auc"] = 0.0
            print("⚠️ Warning: Only one class present in labels, AUC set to 0.0")

        print(f"  - Calculated metrics: {metrics}")
        return metrics

    except Exception as e:
        print(f"❌ Error in calculate_metrics: {e}")
        import traceback

        traceback.print_exc()

        # Return default metrics
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, Any],
    config_dict: Dict[str, Any],
    filepath: Path,
    is_best: bool = False,
    best_model_path: Optional[Path] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config_dict,
        "best_metric": metrics.get("val_f1_score", 0.0),
    }

    # Save checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

    # Save best model separately
    if is_best and best_model_path:
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model saved: {best_model_path}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Checkpoint loaded from: {filepath}")
    return checkpoint


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for handling imbalanced data."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)

    # Calculate weights inversely proportional to class frequencies
    weights = total_samples / (len(unique_labels) * counts)

    # Create weight tensor
    class_weights = torch.zeros(len(unique_labels), dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        class_weights[int(label)] = weights[i]

    logger.info(f"Class weights: {dict(zip(unique_labels, weights))}")
    return class_weights


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics for logging."""
    formatted_lines = []

    # Training metrics
    train_metrics = {
        k: v for k, v in metrics.items() if k.startswith("train") or k in ["loss", "focal_loss", "dice_loss"]
    }
    if train_metrics:
        train_str = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in train_metrics.items()]
        )
        formatted_lines.append(f"Train: {train_str}")

    # Validation metrics
    val_metrics = {k: v for k, v in metrics.items() if k.startswith("val_")}
    if val_metrics:
        val_str = " | ".join(
            [f"{k[4:]}: {v:.4f}" if isinstance(v, float) else f"{k[4:]}: {v}" for k, v in val_metrics.items()]
        )
        formatted_lines.append(f"Val:   {val_str}")

    # Learning rate
    if "lr" in metrics:
        formatted_lines.append(f"LR: {metrics['lr']:.2e}")

    return "\n".join(formatted_lines)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, monitor: str = "val_f1_score", mode: str = "max", min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.wait = 0
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def __call__(self, current_metric: float):
        """Check if training should stop."""
        if self.mode == "max":
            improved = current_metric > self.best_metric + self.min_delta
        else:
            improved = current_metric < self.best_metric - self.min_delta

        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered. Best {self.monitor}: {self.best_metric:.4f}")


class MetricTracker:
    """Track and compute moving averages of metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

            # Keep only recent values
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size :]

    def get_average(self, key: str) -> float:
        """Get moving average for a metric."""
        if key in self.metrics and self.metrics[key]:
            return np.mean(self.metrics[key])
        return 0.0

    def get_all_averages(self) -> Dict[str, float]:
        """Get moving averages for all metrics."""
        return {key: self.get_average(key) for key in self.metrics}


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration."""
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_submission(predictions: Dict[str, float], threshold: float = 0.5) -> Dict[str, Any]:
    """Create submission from predictions."""
    submission_data = []

    for image_id, prob in predictions.items():
        prediction = 1 if prob >= threshold else 0
        submission_data.append({"ID": image_id, "label": prediction, "probability": prob})

    return submission_data


def validate_data_integrity(data_dir: Path, csv_path: Path) -> Tuple[bool, list]:
    """Validate that all samples in CSV have corresponding data files."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing_files = []

    for _, row in df.iterrows():
        sample_id = row["ID"]
        data_path = data_dir / f"{sample_id}.npy"

        if not data_path.exists():
            missing_files.append(sample_id)

    is_valid = len(missing_files) == 0

    if not is_valid:
        logger.warning(f"Found {len(missing_files)} missing data files")
    else:
        logger.info("Data integrity check passed")

    return is_valid, missing_files


def log_system_info():
    """Log system and environment information."""
    import torch
    import platform

    logger.info("=" * 50)
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    logger.info("=" * 50)


def compute_class_distribution(df):
    """Compute and log class distribution."""
    distribution = df["label"].value_counts().sort_index()
    total = len(df)

    logger.info("Class Distribution:")
    for label, count in distribution.items():
        percentage = (count / total) * 100
        logger.info(f"  Class {label}: {count} samples ({percentage:.1f}%)")

    return distribution


def check_gpu_memory():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3

            logger.info(f"GPU {i} Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference."""
    model.eval()

    # Try to compile model if using PyTorch 2.0+
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            logger.info("Model compiled for faster inference")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")

    return model
