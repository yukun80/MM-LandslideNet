"""
Modified Training Script for MM-TNF Model

Key Changes:
1. Adapted for dual-branch TNF model input format
2. Updated forward pass: model(optical_data, sar_data)
3. Multi-branch loss computation (optical + sar + fusion)
4. Updated metrics calculation for ensemble predictions
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
from .config import config, init_training
from .dataset import MultiModalLandslideDataset, get_augmentations, load_exclude_ids
from .models import create_tnf_model
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
    """Train TNF model for one epoch with fixed data collection."""
    model.train()

    # Metrics tracking
    total_loss = 0.0
    branch_losses = {"optical": 0.0, "sar": 0.0, "fusion": 0.0}
    all_predictions = []
    all_targets = []

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Extract data and move to device
        optical_data = batch["optical"].to(device)  # (B, 5, 64, 64)
        sar_data = batch["sar"].to(device)  # (B, 8, 64, 64)
        targets = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if config.MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(optical_data, sar_data)
                loss_dict = model.compute_loss(outputs, targets, criterion)
                loss = loss_dict["total_loss"]
        else:
            outputs = model(optical_data, sar_data)
            loss_dict = model.compute_loss(outputs, targets, criterion)
            loss = loss_dict["total_loss"]

        # Backward pass
        if config.MIXED_PRECISION and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
            optimizer.step()

        # Update metrics
        total_loss += loss.item()
        for branch, branch_loss in loss_dict.items():
            if branch != "total_loss" and branch in branch_losses:
                branch_losses[branch] += branch_loss.item()

        # 🔧 FIXED: Collect predictions for metrics (use ensemble prediction)
        with torch.no_grad():
            final_probs = torch.sigmoid(outputs["final_logits"])

            # 确保数据是1D的，并转换为标量列表
            # final_probs形状: (B, 1) -> flatten成 (B,) -> 转为list
            probs_flat = final_probs.flatten().cpu().numpy()  # (B,)
            targets_flat = targets.flatten().cpu().numpy()  # (B,)

            # 添加到列表中 - 现在是标量值而不是数组
            all_predictions.extend(probs_flat.tolist())
            all_targets.extend(targets_flat.tolist())

        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg": f"{total_loss/(batch_idx+1):.4f}"})

    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_branch_losses = {k: v / len(dataloader) for k, v in branch_losses.items()}

    # 🔧 FIXED: Calculate training metrics with proper data types
    try:
        # 验证数据类型和形状
        predictions_array = np.array(all_predictions)  # 应该是 (N,) 形状
        targets_array = np.array(all_targets)  # 应该是 (N,) 形状

        print(f"Debug - Predictions shape: {predictions_array.shape}, dtype: {predictions_array.dtype}")
        print(f"Debug - Targets shape: {targets_array.shape}, dtype: {targets_array.dtype}")

        # 确保目标值是整数类型 (0或1)
        targets_array = targets_array.astype(int)

        train_metrics = calculate_metrics(predictions_array, targets_array, threshold=0.5)

    except Exception as e:
        print(f"⚠️ Warning: Error calculating training metrics: {e}")
        # 返回默认指标值，避免训练中断
        train_metrics = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
        }

    return {
        "loss": avg_loss,
        "optical_loss": avg_branch_losses["optical"],
        "sar_loss": avg_branch_losses["sar"],
        "fusion_loss": avg_branch_losses["fusion"],
        **train_metrics,
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate TNF model for one epoch with fixed data collection."""
    model.eval()

    # Metrics tracking
    total_loss = 0.0
    branch_losses = {"optical": 0.0, "sar": 0.0, "fusion": 0.0}
    all_predictions = []
    all_targets = []

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Extract data and move to device
            optical_data = batch["optical"].to(device)
            sar_data = batch["sar"].to(device)
            targets = batch["label"].to(device).unsqueeze(1)

            # Forward pass
            if config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(optical_data, sar_data)
                    loss_dict = model.compute_loss(outputs, targets, criterion)
                    loss = loss_dict["total_loss"]
            else:
                outputs = model(optical_data, sar_data)
                loss_dict = model.compute_loss(outputs, targets, criterion)
                loss = loss_dict["total_loss"]

            # Update metrics
            total_loss += loss.item()
            for branch, branch_loss in loss_dict.items():
                if branch != "total_loss" and branch in branch_losses:
                    branch_losses[branch] += branch_loss.item()

            # 🔧 FIXED: Collect predictions (use ensemble prediction)
            final_probs = torch.sigmoid(outputs["final_logits"])

            # 确保数据是1D的，并转换为标量列表
            probs_flat = final_probs.flatten().cpu().numpy()  # (B,)
            targets_flat = targets.flatten().cpu().numpy()  # (B,)

            # 添加到列表中 - 现在是标量值而不是数组
            all_predictions.extend(probs_flat.tolist())
            all_targets.extend(targets_flat.tolist())

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg": f"{total_loss/(batch_idx+1):.4f}"})

    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_branch_losses = {k: v / len(dataloader) for k, v in branch_losses.items()}

    # 🔧 FIXED: Calculate validation metrics with proper data types
    try:
        # 验证数据类型和形状
        predictions_array = np.array(all_predictions)  # 应该是 (N,) 形状
        targets_array = np.array(all_targets)  # 应该是 (N,) 形状

        print(f"Debug - Val Predictions shape: {predictions_array.shape}, dtype: {predictions_array.dtype}")
        print(f"Debug - Val Targets shape: {targets_array.shape}, dtype: {targets_array.dtype}")

        # 确保目标值是整数类型 (0或1)
        targets_array = targets_array.astype(int)

        val_metrics = calculate_metrics(predictions_array, targets_array, threshold=0.5)

    except Exception as e:
        print(f"⚠️ Warning: Error calculating validation metrics: {e}")
        # 返回默认指标值，避免训练中断
        val_metrics = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
        }

    return {
        "loss": avg_loss,
        "optical_loss": avg_branch_losses["optical"],
        "sar_loss": avg_branch_losses["sar"],
        "fusion_loss": avg_branch_losses["fusion"],
        **val_metrics,
    }


def setup_training(resume_path: Optional[str] = None) -> Dict[str, Any]:
    """Setup all training components for TNF model."""
    logger.info("🚀 Setting up TNF training components...")

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

    # Create TNF model
    logger.info("🏗️ Creating TNF model...")
    model = create_tnf_model(
        pretrained=config.MODEL_CONFIG.get("pretrained", True),
        optical_channels=5,
        sar_channels=8,
        optical_feature_dim=512,
        sar_feature_dim=512,
        fusion_dim=512,
    )
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
        patience=config.EARLY_STOPPING_CONFIG["patience"],
        monitor=config.EARLY_STOPPING_CONFIG["monitor"],
        mode=config.EARLY_STOPPING_CONFIG["mode"],
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
    """Main training function for TNF model."""
    init_training()  # 只在训练时调用
    logger.info("=" * 80)
    logger.info("🚀 Starting MM-TNF Training")
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
        "train_optical_loss": [],
        "train_sar_loss": [],
        "train_fusion_loss": [],
        "val_optical_loss": [],
        "val_sar_loss": [],
        "val_fusion_loss": [],
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

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Log metrics
            logger.info(f"\nEpoch {epoch}/{config.NUM_EPOCHS-1}")
            logger.info(f"Train - {format_metrics(train_metrics)}")
            logger.info(f"Val   - {format_metrics(val_metrics)}")
            logger.info(
                f"Branch Losses - Train: O:{train_metrics['optical_loss']:.4f} S:{train_metrics['sar_loss']:.4f} F:{train_metrics['fusion_loss']:.4f}"
            )
            logger.info(
                f"Branch Losses - Val:   O:{val_metrics['optical_loss']:.4f} S:{val_metrics['sar_loss']:.4f} F:{val_metrics['fusion_loss']:.4f}"
            )
            logger.info(f"LR: {current_lr:.2e}, Time: {time.time() - epoch_start_time:.1f}s")

            # Update history
            for key in history.keys():
                if key == "lr":
                    history[key].append(current_lr)
                elif key.startswith("train_"):
                    metric_name = key[6:]  # Remove 'train_' prefix
                    history[key].append(train_metrics.get(metric_name, 0.0))
                elif key.startswith("val_"):
                    metric_name = key[4:]  # Remove 'val_' prefix
                    history[key].append(val_metrics.get(metric_name, 0.0))

            # Check if this is the best model
            current_metric = val_metrics[config.EARLY_STOPPING_CONFIG["monitor"]]
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                logger.info(f"🎉 New best model! {config.EARLY_STOPPING_CONFIG['monitor']}: {best_metric:.4f}")
                best_model_path = config.CHECKPOINT_DIR / "best_model.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    config_dict=config.to_dict(),
                    filepath=best_model_path,
                    is_best=is_best,
                    best_model_path=best_model_path,
                )
                # Save regular checkpoint
                logger.info(f"💾 Checkpoint saved: {best_model_path}")
            elif (epoch + 1) % config.CHECKPOINT_CONFIG["save_interval"] == 0:
                # Save checkpoint
                checkpoint_path = config.CHECKPOINT_DIR / f"epoch_{epoch}.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    config_dict=config.to_dict(),
                    filepath=checkpoint_path,
                    is_best=is_best,
                    best_model_path=best_model_path,
                )
                # Save regular checkpoint
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

            # Early stopping check
            early_stopping(current_metric)
            if early_stopping.should_stop:
                logger.info("⏰ Early stopping triggered")
                break

    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        total_time = time.time() - training_start_time
        logger.info(f"🏁 Training completed in {total_time:.1f}s")
        logger.info(f"📊 Best {config.EARLY_STOPPING_CONFIG['monitor']}: {best_metric:.4f}")

        # Save final history
        history_path = config.LOG_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"📈 Training history saved: {history_path}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train MM-TNF model for landslide detection")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    run_training(args)


if __name__ == "__main__":
    main()
