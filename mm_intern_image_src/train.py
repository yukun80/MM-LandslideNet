"""
Training Script for MM-InternImage-TNF

This module orchestrates the complete training and evaluation process for the
multi-modal landslide detection model, including data preparation, model training,
validation, and checkpoint management.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

try:
    # Try relative imports first (when imported as part of package)
    from .config import config
    from .dataset import create_datasets
    from .models import create_model
    from .utils import (
        CombinedLoss,
        calculate_metrics,
        save_checkpoint,
        load_checkpoint,
        seed_everything,
        get_class_weights,
        format_metrics,
        EarlyStopping,
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import config
    from dataset import create_datasets
    from models import create_model
    from utils import (
        CombinedLoss,
        calculate_metrics,
        save_checkpoint,
        load_checkpoint,
        seed_everything,
        get_class_weights,
        format_metrics,
        EarlyStopping,
    )


def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a weighted random sampler to handle class imbalance.

    Args:
        dataset: Training dataset

    Returns:
        WeightedRandomSampler instance
    """
    # Extract labels from dataset
    labels = []
    for i in range(len(dataset)):
        sample = dataset.df.iloc[i]
        labels.append(sample["label"])

    labels = np.array(labels)

    # Calculate class weights
    class_weights = get_class_weights(labels)

    # Create sample weights
    sample_weights = []
    for label in labels:
        sample_weights.append(class_weights[int(label)].item())

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # Use logger from the calling function
    logger = logging.getLogger('MM-InternImage-TNF')
    logger.info(f"Class distribution: {np.bincount(labels)}")
    logger.info(f"Class weights: {class_weights}")

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary containing training metrics
    """
    model.train()

    running_loss = 0.0
    running_focal_loss = 0.0
    running_dice_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        for key in ["optical", "sar", "sar_diff", "label"]:
            if key in batch:
                batch[key] = batch[key].to(device)

        labels = batch["label"]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch)

        # Calculate loss
        loss_dict = criterion(logits.squeeze(), labels)
        total_loss = loss_dict["total_loss"]

        # Backward pass
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        running_loss += total_loss.item()
        running_focal_loss += loss_dict["focal_loss"].item()
        running_dice_loss += loss_dict["dice_loss"].item()

        # Store predictions and labels for metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        if batch_idx % config.LOG_EVERY_N_STEPS == 0:
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{avg_loss:.4f}",
                    "Focal": f"{running_focal_loss / (batch_idx + 1):.4f}",
                    "Dice": f"{running_dice_loss / (batch_idx + 1):.4f}",
                }
            )

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_focal_loss = running_focal_loss / len(dataloader)
    epoch_dice_loss = running_dice_loss / len(dataloader)

    # Calculate classification metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))

    train_metrics = {
        "train_loss": epoch_loss,
        "train_focal_loss": epoch_focal_loss,
        "train_dice_loss": epoch_dice_loss,
        "train_accuracy": metrics["accuracy"],
        "train_f1_score": metrics["f1_score"],
        "train_precision": metrics["precision"],
        "train_recall": metrics["recall"],
        "train_auc": metrics["auc"],
    }

    return train_metrics


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: CombinedLoss, device: str, epoch: int
) -> Dict[str, float]:
    """
    Evaluate the model on validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        epoch: Current epoch number

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()

    running_loss = 0.0
    running_focal_loss = 0.0
    running_dice_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            for key in ["optical", "sar", "sar_diff", "label"]:
                if key in batch:
                    batch[key] = batch[key].to(device)

            labels = batch["label"]

            # Forward pass
            logits = model(batch)

            # Calculate loss
            loss_dict = criterion(logits.squeeze(), labels)
            total_loss = loss_dict["total_loss"]

            # Accumulate metrics
            running_loss += total_loss.item()
            running_focal_loss += loss_dict["focal_loss"].item()
            running_dice_loss += loss_dict["dice_loss"].item()

            # Store predictions and labels
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_focal_loss = running_focal_loss / len(dataloader)
    epoch_dice_loss = running_dice_loss / len(dataloader)

    # Calculate classification metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))

    val_metrics = {
        "val_loss": epoch_loss,
        "val_focal_loss": epoch_focal_loss,
        "val_dice_loss": epoch_dice_loss,
        "val_accuracy": metrics["accuracy"],
        "val_f1_score": metrics["f1_score"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_auc": metrics["auc"],
    }

    return val_metrics


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration for the training process.
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(config.OUTPUT_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mm_intern_image_tnf_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger('MM-InternImage-TNF')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def run_training() -> None:
    """
    Main training function that orchestrates the entire training process.
    """
    # Set up logging
    logger = setup_logging()
    logger.info("=== MM-InternImage-TNF Training ===")

    # Set random seed for reproducibility
    seed_everything(config.RANDOM_SEED)

    # Print configuration
    config.print_config()

    # Load and prepare data
    logger.info("📊 Loading and preparing data...")
    df = pd.read_csv(config.TRAIN_CSV_PATH)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")

    # Load exclude IDs first
    from mm_intern_image_src.dataset import load_exclude_ids
    exclude_ids = load_exclude_ids()
    logger.info(f"Excluding {len(exclude_ids)} low-quality samples")

    # Filter out excluded samples BEFORE splitting
    clean_df = df[~df["ID"].isin(exclude_ids)].reset_index(drop=True)
    logger.info(f"Clean samples after filtering: {len(clean_df)}")
    logger.info(f"Clean class distribution:\n{clean_df['label'].value_counts()}")
    
    # Calculate retention rate
    retention_rate = len(clean_df) / len(df)
    logger.info(f"Retention rate: {retention_rate:.1%}")

    # Train-validation split (8:2) with stratification on clean data
    train_df, val_df = train_test_split(
        clean_df, test_size=0.2, random_state=config.RANDOM_SEED, stratify=clean_df["label"]
    )

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Training class distribution:\n{train_df['label'].value_counts()}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Validation class distribution:\n{val_df['label'].value_counts()}")

    # Create datasets (exclude_ids already filtered, so pass empty list)
    train_dataset, val_dataset = create_datasets(train_df=train_df, val_df=val_df, data_dir=config.TRAIN_DATA_DIR)

    # Create weighted sampler for training
    logger.info("⚖️ Creating weighted sampler for class imbalance...")
    weighted_sampler = create_weighted_sampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=weighted_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create model
    logger.info("🏗️ Creating model...")
    model = create_model(num_classes=config.NUM_CLASSES, pretrained_optical=True, device=config.DEVICE)

    # Create loss function
    criterion = CombinedLoss(**config.LOSS_CONFIG)

    # Create optimizer
    optimizer = AdamW(model.parameters(), **config.OPTIMIZER_CONFIG)

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, **config.SCHEDULER_CONFIG)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="max")  # Maximize F1 score

    # Training tracking
    best_f1_score = 0.0
    train_history = []

    print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs...")
    print(f"Device: {config.DEVICE}")
    print(f"Monitoring metric: {config.MONITOR_METRIC}")

    # Main training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        # Training phase
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.DEVICE,
            epoch=epoch,
        )

        # Validation phase
        if (epoch + 1) % config.VALIDATE_EVERY_N_EPOCHS == 0:
            val_metrics = evaluate(
                model=model, dataloader=val_loader, criterion=criterion, device=config.DEVICE, epoch=epoch
            )
        else:
            val_metrics = {}

        # Learning rate scheduler step
        scheduler.step()

        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics["epoch"] = epoch + 1
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        train_history.append(epoch_metrics)

        # Print metrics
        print(f"Train: {format_metrics(train_metrics, 'train_')}")
        if val_metrics:
            print(f"Val:   {format_metrics(val_metrics, 'val_')}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint and check for best model
        if val_metrics and config.MONITOR_METRIC in val_metrics:
            current_f1 = val_metrics[config.MONITOR_METRIC]

            # Check if this is the best model
            is_best = current_f1 > best_f1_score
            if is_best:
                best_f1_score = current_f1
                print(f"🏆 New best {config.MONITOR_METRIC}: {best_f1_score:.4f}")

            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_f1_score,
                "metrics": epoch_metrics,
                "config": config.__dict__,
            }

            # Save regular checkpoint
            checkpoint_path = config.get_model_save_path(epoch + 1, current_f1)
            save_checkpoint(checkpoint_state, checkpoint_path, is_best)

            # Save latest checkpoint
            latest_path = config.get_latest_model_path()
            save_checkpoint(checkpoint_state, latest_path, False)

            # Early stopping check
            if early_stopping(current_f1):
                print(f"⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

    print("\n✅ Training completed!")
    print(f"Best {config.MONITOR_METRIC}: {best_f1_score:.4f}")
    print(f"Model checkpoints saved to: {config.CHECKPOINT_DIR}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train MM-InternImage-TNF model")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")

    args = parser.parse_args()

    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        # Implementation for resuming would go here
        # For now, we'll just start fresh

    # Run training
    run_training()


if __name__ == "__main__":
    main()
