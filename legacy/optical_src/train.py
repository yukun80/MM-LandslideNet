#!/usr/bin/env python3
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import optical baseline modules
from optical_src.config import OpticalBaselineConfig
from optical_src.dataset import LandslideDataset
from optical_src.model import BaselineOpticalModel
from optical_src.utils import setup_logging, set_seed, get_device, format_time, ensure_dir

# Setup logger for this module
logger = logging.getLogger("optical_baseline.train")


class Trainer:
    """
    Comprehensive trainer class for the optical baseline model.
    """

    def __init__(self, config: Optional[OpticalBaselineConfig] = None):
        """
        Initialize the trainer with configuration.

        Args:
            config: OpticalBaselineConfig object (uses OpticalBaselineConfig() if None)
        """
        # Load configuration
        self.config = config if config is not None else OpticalBaselineConfig()

        # Set random seed for reproducibility
        set_seed(self.config.SEED)

        # Ensure directories exist
        self.config.create_dirs()

        # Set device
        self.device = get_device()

        # Initialize components
        self._load_data()
        self._initialize_model()
        self._setup_training()
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        logger.info("Trainer initialized successfully")

    def _load_data(self) -> None:
        """Load and prepare datasets and dataloaders."""
        logger.info("Loading datasets...")

        try:
            # Get dataloaders from the dataset module
            (self.train_loader, self.val_loader, self.train_dataset, self.val_dataset) = (
                LandslideDataset.get_train_val_dataloaders(self.config)
            )

            # Store dataset information
            self.train_size = len(self.train_dataset)
            self.val_size = len(self.val_dataset)
            self.steps_per_epoch = len(self.train_loader)

            logger.info(f"Training samples: {self.train_size}")
            logger.info(f"Validation samples: {self.val_size}")
            logger.info(f"Steps per epoch: {self.steps_per_epoch}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _initialize_model(self) -> None:
        """Initialize the model and move to device."""
        logger.info("Initializing model...")

        try:
            # Create model from configuration
            self.model = BaselineOpticalModel.from_config(self.config, variant="swin_tiny")
            self.model = self.model.to(self.device)

            logger.info("Model initialization completed successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _setup_training(self) -> None:
        """Setup loss function, optimizer, and scheduler."""
        logger.info("Setting up training components...")

        try:
            # Loss function with class weighting for imbalance
            if self.config.USE_CLASS_WEIGHTING:
                pos_weight = self._calculate_pos_weight()
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                logger.info(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight.item():.3f}")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                logger.info("Using BCEWithLogitsLoss without class weighting")

            # Optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY
            )

            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.SCHEDULER_T_MAX, eta_min=self.config.SCHEDULER_ETA_MIN
            )

            logger.info(f"Optimizer: AdamW (lr={self.config.LEARNING_RATE}, wd={self.config.WEIGHT_DECAY})")
            logger.info(
                f"Scheduler: CosineAnnealingLR (T_max={self.config.SCHEDULER_T_MAX}, eta_min={self.config.SCHEDULER_ETA_MIN})"
            )

        except Exception as e:
            logger.error(f"Error setting up training components: {str(e)}")
            raise

    def _calculate_pos_weight(self) -> torch.Tensor:
        """Calculate positive class weight for handling class imbalance."""
        try:
            labels = self.train_dataset.labels
            neg_count = sum(labels == 0)
            pos_count = sum(labels == 1)
            pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32).to(self.device)
            logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
            return pos_weight
        except Exception as e:
            logger.warning(f"Could not calculate pos_weight: {str(e)}. Using default.")
            return torch.tensor(1.0).to(self.device)

    def _setup_logging(self) -> None:
        """Setup TensorBoard logging."""
        log_dir = self.config.OPTICAL_LOG_DIR / f"run_{int(time.time())}"
        ensure_dir(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging to: {log_dir}")

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data).squeeze()
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()

            # Convert outputs to predictions
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Log progress
            if batch_idx % self.config.LOG_FREQUENCY == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, "
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions)
        metrics["loss"] = avg_loss

        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(data).squeeze()
                loss = self.criterion(outputs, labels)

                # Accumulate metrics
                total_loss += loss.item()

                # Convert outputs to predictions and probabilities
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probs)
        metrics["loss"] = avg_loss

        return metrics

    def _calculate_metrics(
        self, labels: List[float], predictions: List[float], probabilities: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            labels: True labels
            predictions: Predicted labels
            probabilities: Predicted probabilities (optional, for AUC)

        Returns:
            Dictionary of metrics
        """
        try:
            # Calculate basic metrics with safe handling of edge cases
            accuracy = accuracy_score(labels, predictions)

            # For precision, recall, and f1, handle edge cases manually
            try:
                precision = precision_score(labels, predictions)
            except (ValueError, ZeroDivisionError):
                precision = 0.0

            try:
                recall = recall_score(labels, predictions)
            except (ValueError, ZeroDivisionError):
                recall = 0.0

            try:
                f1 = f1_score(labels, predictions)
            except (ValueError, ZeroDivisionError):
                f1 = 0.0

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            # Add AUC if probabilities provided
            if probabilities is not None:
                try:
                    metrics["auc"] = roc_auc_score(labels, probabilities)
                except ValueError:
                    metrics["auc"] = 0.0  # In case of edge cases

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating metrics: {str(e)}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.0}

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard and console."""
        # Console logging
        logger.info(f"Epoch {self.current_epoch} Results:")
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"F1: {train_metrics['f1_score']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"F1: {val_metrics['f1_score']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        # TensorBoard logging
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f"Train/{metric_name}", value, self.current_epoch)

        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f"Val/{metric_name}", value, self.current_epoch)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning_Rate", current_lr, self.current_epoch)

    def _save_checkpoint(self, val_metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_f1": self.best_f1,
            "config": self.config.__dict__,
        }

        # Ensure checkpoint directory exists
        ensure_dir(self.config.OPTICAL_CHECKPOINT_DIR)

        # Save regular checkpoint
        checkpoint_path = self.config.OPTICAL_CHECKPOINT_DIR / f"epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.config.OPTICAL_CHECKPOINT_DIR / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with F1: {val_metrics['f1_score']:.4f}")

    def _check_early_stopping(self, val_f1: float) -> bool:
        """
        Check if early stopping criteria are met.

        Returns:
            True if training should stop
        """
        if val_f1 > self.best_f1 + self.config.MIN_DELTA:
            self.best_f1 = val_f1
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping triggered. Best F1: {self.best_f1:.4f} " f"at epoch {self.best_epoch}")
                return True
            return False

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.NUM_EPOCHS} epochs")

        start_time = time.time()

        try:
            for epoch in range(self.config.NUM_EPOCHS):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training
                train_metrics = self._train_epoch()

                # Validation
                val_metrics = self._validate_epoch()

                # Learning rate scheduling
                self.scheduler.step()

                # Logging
                self._log_metrics(train_metrics, val_metrics)

                # Model saving and early stopping
                is_best = val_metrics["f1_score"] > self.best_f1

                if epoch % self.config.SAVE_FREQUENCY == 0 or is_best:
                    self._save_checkpoint(val_metrics, is_best)

                # Early stopping check
                if self.config.EARLY_STOPPING and self._check_early_stopping(val_metrics["f1_score"]):
                    break

                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch} completed in {format_time(epoch_time)}")

            total_time = time.time() - start_time
            logger.info(f"Training completed in {format_time(total_time)}")
            logger.info(f"Best validation F1: {self.best_f1:.4f} at epoch {self.best_epoch}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
        finally:
            self.writer.close()


def main() -> None:
    """Main function to run training."""
    # Load configuration
    config = OpticalBaselineConfig()

    # Setup logging
    log_file = config.OPTICAL_LOG_DIR / "training.log"
    setup_logging(log_level="INFO", log_file=log_file)

    logger.info("Starting MM-LandslideNet Optical Baseline Training")
    logger.info(f"Configuration: {config}")

    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
