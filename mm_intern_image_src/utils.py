"""
Utility Functions for MM-InternImage-TNF

This module contains reusable utility functions including loss functions,
evaluation metrics, checkpoint management, and reproducibility utilities.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from pathlib import Path

try:
    # Try relative import first (when imported as part of package)
    from .config import config
except ImportError:
    # Fall back to absolute import (when run directly)
    from config import config


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.

    Reference: Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.

        Args:
            inputs: Predicted logits [N, 1] or [N]
            targets: Ground truth labels [N] (0 or 1)

        Returns:
            Focal loss value
        """
        # Ensure inputs are properly shaped
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)

        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

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
    ):
        """
        Initialize Combined Loss.

        Args:
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            dice_smooth: Smoothing parameter for dice loss
        """
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
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


def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for binary classification.

    Args:
        y_pred: Predicted probabilities or logits [N] or [N, 1]
        y_true: Ground truth labels [N]
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary containing various metrics
    """
    # Ensure proper shapes and convert to numpy
    if isinstance(y_pred, torch.Tensor):
        if y_pred.dim() == 2 and y_pred.size(1) == 1:
            y_pred = y_pred.squeeze(1)
        # Apply sigmoid if logits
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = y_pred

    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = y_true

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_np >= threshold).astype(int)

    # Calculate metrics
    try:
        metrics = {
            "accuracy": accuracy_score(y_true_np, y_pred_binary),
            "precision": precision_score(y_true_np, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_np, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true_np, y_pred_binary, zero_division=0),
        }

        # Add AUC if there are both classes present
        if len(np.unique(y_true_np)) > 1:
            metrics["auc"] = roc_auc_score(y_true_np, y_pred_np)
        else:
            metrics["auc"] = 0.0

    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.0}

    return metrics


def save_checkpoint(state: Dict[str, Any], filepath: Path, is_best: bool = False) -> None:
    """
    Save model checkpoint to file.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        torch.save(state, filepath)
        print(f"Checkpoint saved: {filepath}")

        # Save as best model if applicable
        if is_best:
            best_path = config.get_best_model_path()
            torch.save(state, best_path)
            print(f"Best model saved: {best_path}")

    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(
    filepath: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint from file.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on

    Returns:
        Dictionary containing loaded checkpoint information
    """
    if device is None:
        device = config.DEVICE

    try:
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model state loaded from {filepath}")

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "best_metric": checkpoint.get("best_metric", 0.0),
            "metrics": checkpoint.get("metrics", {}),
        }

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {"epoch": 0, "best_metric": 0.0, "metrics": {}}


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: Array of class labels

    Returns:
        Tensor of class weights
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)

    # Calculate weights inversely proportional to class frequency
    weights = total_samples / (len(unique_classes) * counts)

    # Create weight tensor
    weight_dict = dict(zip(unique_classes, weights))
    class_weights = torch.tensor([weight_dict[i] for i in range(len(unique_classes))], dtype=torch.float32)

    return class_weights


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary for pretty printing.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to add to metric names

    Returns:
        Formatted string
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")

    return " | ".join(formatted)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should be stopped.

        Args:
            metric_value: Current metric value

        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = metric_value
        elif self._is_improvement(metric_value):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, metric_value: float) -> bool:
        """Check if the current metric value is an improvement."""
        if self.mode == "max":
            return metric_value > self.best_score + self.min_delta
        else:
            return metric_value < self.best_score - self.min_delta
