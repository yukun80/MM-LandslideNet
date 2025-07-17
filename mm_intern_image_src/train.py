import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging

from .config import config
from .dataset import MultiModalLandslideDataset, get_augmentations, load_exclude_ids
from .models import create_optical_dominated_model
from .utils import (
    CombinedLoss,
    calculate_metrics,
    save_checkpoint,
    seed_everything,
    get_class_weights,
    format_metrics,
    EarlyStopping,
)

def create_weighted_sampler(df: pd.DataFrame):
    labels = df["label"].values
    class_weights = get_class_weights(labels)
    sample_weights = torch.tensor([class_weights[int(label)].item() for label in labels], dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    all_predictions, all_labels = [], []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(batch)
        loss_dict = criterion(logits.squeeze(), labels)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_VAL)
        optimizer.step()

        running_loss += total_loss.item()
        all_predictions.extend(torch.sigmoid(logits.squeeze()).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_predictions, all_labels)
    metrics["train_loss"] = running_loss / len(dataloader)
    return metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch["label"]
            logits = model(batch)
            loss_dict = criterion(logits.squeeze(), labels)
            running_loss += loss_dict["total_loss"].item()
            all_predictions.extend(torch.sigmoid(logits.squeeze()).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_predictions, all_labels)
    metrics["val_loss"] = running_loss / len(dataloader)
    return metrics

def setup_logging():
    config.setup_run_paths() # This creates the unique directories
    log_file = config.LOG_DIR / "training.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger('LandslideDetection')

def run_training(args):
    logger = setup_logging()
    seed_everything(config.RANDOM_SEED)
    config.print_config()

    logger.info("📊 Loading data...")
    df = pd.read_csv(config.TRAIN_CSV_PATH)
    exclude_ids = load_exclude_ids()
    clean_df = df[~df["ID"].isin(exclude_ids)].reset_index(drop=True)
    train_df, val_df = train_test_split(clean_df, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED, stratify=clean_df["label"])

    train_dataset = MultiModalLandslideDataset(df=train_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("train"))
    val_dataset = MultiModalLandslideDataset(df=val_df, data_dir=config.TRAIN_DATA_DIR, augmentations=get_augmentations("val"))

    sampler = create_weighted_sampler(train_df) if config.USE_WEIGHTED_SAMPLER else None
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=sampler is None)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    logger.info("🏗️ Creating model...")
    model = create_optical_dominated_model(num_classes=config.NUM_CLASSES, pretrained=True)
    model.to(config.DEVICE)

    criterion = CombinedLoss(**config.LOSS_CONFIG)
    optimizer = AdamW(model.parameters(), **config.OPTIMIZER_CONFIG)
    scheduler = CosineAnnealingLR(optimizer, **config.SCHEDULER_CONFIG)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="max", monitor_metric=config.MONITOR_METRIC)

    best_metric_value = 0.0
    logger.info(f"🚀 Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE, epoch)
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()

        logger.info(f"Epoch {epoch+1} | Train: {format_metrics(train_metrics, 'train')} | Val: {format_metrics(val_metrics, 'val')} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        current_metric_value = val_metrics[config.MONITOR_METRIC]
        
        # Save best model based on F1 score
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            logger.info(f"🏆 New best {config.MONITOR_METRIC}: {best_metric_value:.4f}. Saving best_model.pth")
            save_checkpoint(model, config.get_best_model_path())

        # Save latest model checkpoint
        save_checkpoint(model, config.get_latest_model_path())

        # Save checkpoint every N epochs
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            logger.info(f"💾 Saving periodic checkpoint at epoch {epoch+1}.")
            save_checkpoint(model, config.get_epoch_model_path(epoch + 1))

        if early_stopping(current_metric_value):
            logger.info("⏹️ Early stopping triggered.")
            break

    logger.info(f"✅ Training completed! Best {config.MONITOR_METRIC}: {best_metric_value:.4f}")
    logger.info(f"Checkpoints and logs saved in: {config.CHECKPOINT_DIR.parent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Optical-Dominated Cooperative Model")
    # Add any arguments if needed in the future
    args = parser.parse_args()
    run_training(args)
