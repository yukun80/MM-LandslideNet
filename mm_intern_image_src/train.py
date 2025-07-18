"""
Complete training script for MM-InternImage-TNF model.
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging
import time
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

"""
python -m mm_intern_image_src.train
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import project modules
try:
    from .config import config
    from .dataset import MultiModalLandslideDataset, get_augmentations, load_exclude_ids
    from .models import create_optical_dominated_model
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
    # Fallback for direct execution
    from config import config
    from dataset import MultiModalLandslideDataset, get_augmentations, load_exclude_ids
    from models import create_optical_dominated_model
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


def create_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Create weighted sampler for handling class imbalance."""
    labels = df["label"].values
    class_weights = get_class_weights(labels)
    sample_weights = torch.tensor([class_weights[int(label)].item() for label in labels], dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch["label"]

        optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        if config.MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss_dict = criterion(logits.squeeze(), labels)
                total_loss = loss_dict["total_loss"]

            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss_dict = criterion(logits.squeeze(), labels)
            total_loss = loss_dict["total_loss"]

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
            optimizer.step()

        # Accumulate losses
        running_loss += total_loss.item()
        total_focal_loss += loss_dict["focal_loss"].item()
        total_dice_loss += loss_dict["dice_loss"].item()

        # Collect predictions and labels
        predictions = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{total_loss.item():.4f}",
                "Focal": f'{loss_dict["focal_loss"].item():.4f}',
                "Dice": f'{loss_dict["dice_loss"].item():.4f}',
            }
        )

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_focal_loss = total_focal_loss / len(dataloader)
    epoch_dice_loss = total_dice_loss / len(dataloader)

    # Calculate classification metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_predictions, all_labels)

    return {"loss": epoch_loss, "focal_loss": epoch_focal_loss, "dice_loss": epoch_dice_loss, **metrics}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    running_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)

    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch["label"]

        # Forward pass
        if config.MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss_dict = criterion(logits.squeeze(), labels)
                total_loss = loss_dict["total_loss"]
        else:
            logits = model(batch)
            loss_dict = criterion(logits.squeeze(), labels)
            total_loss = loss_dict["total_loss"]

        # Accumulate losses
        running_loss += total_loss.item()
        total_focal_loss += loss_dict["focal_loss"].item()
        total_dice_loss += loss_dict["dice_loss"].item()

        # Collect predictions and labels
        predictions = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Val Loss": f"{total_loss.item():.4f}",
                "Val Focal": f'{loss_dict["focal_loss"].item():.4f}',
                "Val Dice": f'{loss_dict["dice_loss"].item():.4f}',
            }
        )

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_focal_loss = total_focal_loss / len(dataloader)
    epoch_dice_loss = total_dice_loss / len(dataloader)

    # Calculate classification metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_predictions, all_labels)

    return {
        "val_loss": epoch_loss,
        "val_focal_loss": epoch_focal_loss,
        "val_dice_loss": epoch_dice_loss,
        **{f"val_{k}": v for k, v in metrics.items()},
    }


def setup_training(resume_path: Optional[str] = None) -> Dict[str, Any]:
    """Setup training components."""
    logger.info("🚀 Setting up training components...")

    # Set random seed
    seed_everything(config.RANDOM_SEED)

    # Setup paths
    config.setup_run_paths()

    # Setup device
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    # Load and prepare data
    logger.info("📊 Loading data...")
    df = pd.read_csv(config.TRAIN_CSV_PATH)
    logger.info(f"Loaded {len(df)} samples from CSV")

    # Load exclude IDs
    exclude_ids = load_exclude_ids()
    if exclude_ids:
        logger.info(f"Excluding {len(exclude_ids)} problematic samples")
        df = df[~df["ID"].isin(exclude_ids)]
        logger.info(f"After filtering: {len(df)} samples")

    # Check class distribution
    class_counts = df["label"].value_counts().sort_index()
    logger.info(f"Class distribution: {dict(class_counts)}")

    # Split data
    train_df, val_df = train_test_split(
        df, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED, stratify=df["label"]
    )
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create datasets
    train_dataset = MultiModalLandslideDataset(
        df=train_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("train"), mode="train"
    )

    val_dataset = MultiModalLandslideDataset(
        df=val_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("val"), mode="val"
    )

    # Create data loaders
    if config.USE_WEIGHTED_SAMPLER:
        sampler = create_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Create model
    logger.info("🏗️ Creating model...")
    model = create_optical_dominated_model(num_classes=config.NUM_CLASSES, pretrained=config.MODEL_CONFIG["pretrained"])
    model = model.to(device)

    # Setup loss function
    criterion = CombinedLoss(**config.LOSS_CONFIG)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), **config.OPTIMIZER_CONFIG)

    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, **config.SCHEDULER_CONFIG)

    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        monitor=config.MONITOR_METRIC,
        mode="max",  # F1 score should be maximized
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_metric = 0.0
    if resume_path:
        logger.info(f"📂 Loading checkpoint from {resume_path}")
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", 0.0)
        logger.info(f"Resuming from epoch {start_epoch}, best metric: {best_metric:.4f}")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "early_stopping": early_stopping,
        "device": device,
        "start_epoch": start_epoch,
        "best_metric": best_metric,
    }


def run_training(args: Optional[argparse.Namespace] = None):
    """Main training function."""
    logger.info("=" * 80)
    logger.info("🚀 Starting MM-InternImage-TNF Training")
    logger.info("=" * 80)

    # Setup training
    resume_path = args.resume if args and hasattr(args, "resume") else None
    training_components = setup_training(resume_path)

    model = training_components["model"]
    train_loader = training_components["train_loader"]
    val_loader = training_components["val_loader"]
    criterion = training_components["criterion"]
    optimizer = training_components["optimizer"]
    scheduler = training_components["scheduler"]
    scaler = training_components["scaler"]
    early_stopping = training_components["early_stopping"]
    device = training_components["device"]
    start_epoch = training_components["start_epoch"]
    best_metric = training_components["best_metric"]

    # Print configuration
    config.print_config()

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "lr": [],
    }

    logger.info(f"🎯 Starting training for {config.NUM_EPOCHS} epochs...")
    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            epoch_start_time = time.time()

            # Training phase
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)

            # Validation phase
            val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch)

            # Step scheduler
            scheduler.step()

            # Combine metrics
            combined_metrics = {**train_metrics, **val_metrics}
            combined_metrics["lr"] = optimizer.param_groups[0]["lr"]
            combined_metrics["epoch"] = epoch

            # Update history
            for key in history:
                if key in combined_metrics:
                    history[key].append(combined_metrics[key])
                elif f"train_{key}" in combined_metrics:
                    history[key].append(combined_metrics[f"train_{key}"])

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} ({epoch_time:.1f}s)")
            logger.info(format_metrics(combined_metrics))

            # Check if this is the best model
            current_metric = combined_metrics[config.MONITOR_METRIC]
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                logger.info(f"🎉 New best {config.MONITOR_METRIC}: {best_metric:.4f}")

            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=combined_metrics,
                config_dict=config.to_dict(),
                filepath=config.get_latest_model_path(),
                is_best=is_best,
                best_model_path=config.get_best_model_path(),
            )

            # Save periodic checkpoint
            if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=combined_metrics,
                    config_dict=config.to_dict(),
                    filepath=config.get_epoch_model_path(epoch),
                )

            # Early stopping check
            early_stopping(current_metric)
            if early_stopping.should_stop:
                logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break

    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise

    # Training completed
    total_time = time.time() - training_start_time
    logger.info("=" * 80)
    logger.info(f"🏁 Training completed in {total_time/3600:.2f} hours")
    logger.info(f"🏆 Best {config.MONITOR_METRIC}: {best_metric:.4f}")
    logger.info(f"📂 Model saved to: {config.get_best_model_path()}")
    logger.info(f"📊 Logs saved to: {config.LOG_DIR}")
    logger.info("=" * 80)

    # Save training history
    history_path = config.LOG_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"📈 Training history saved to: {history_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MM-InternImage-TNF model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config file (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_training(args)
