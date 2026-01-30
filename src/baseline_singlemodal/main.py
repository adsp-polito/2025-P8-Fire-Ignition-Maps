
import os
import time
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PiedmontDataset
from model import MultiModalFPN
from train import train, val


# ------------------------
# Global configuration
# ------------------------

ROOT_DIR = "../data/"  # root directory containing fire_* folders
GEOJSON_PATH = "../geojson/piedmont_2012_2024_fa.geojson"  # TODO: set this correctly

CHECKPOINT_DIR = "checkpoints"
LOG_DIR_BASE = "runs"

model_name = "best_model_2.pth"



BATCH_SIZE = 4
NUM_WORKERS = 4

NUM_EPOCHS = 40 
LEARNING_RATE_MAIN = 1e-4
WEIGHT_DECAY = 5e-4

W_MASK = 1.0        # weight for burned-area loss
W_LANDCOVER = 1.0   # weight for landcover loss (unused baseline, kept for API)

TRAIN_SPLIT = 0.8   # 80% train / 20% val
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-3

TARGET_SIZE: Tuple[int, int] = (256, 256)
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Utilities
# ------------------------

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class EarlyStopping:
    """
    Early stopping on a scalar metric (here: validation IoU for burned area).

    Stop when the metric does not improve by at least `min_delta` for `patience` epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.best_epoch = None
        self.counter = 0

    def step(self, metric: float, epoch: int) -> bool:
        """
        Update state with new metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_metric is None or metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def compute_pos_weight(train_dataset) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss based on class imbalance
    in the train_dataset GT masks.

    pos_weight ~ (#negative_pixels / #positive_pixels)
    """
    import numpy as np

    total_pos = 0
    total_neg = 0

    for idx in range(len(train_dataset)):
        # dataset __getitem__ returns 7 elements; last one is gt_mask_tensor
        *_, gt_mask_tensor = train_dataset[idx]  # (sentinel,..., landcover, gt_mask)

        gt_np = gt_mask_tensor.numpy()
        if gt_np.ndim == 3:
            gt_np = gt_np[0]  # squeeze channel [1, H, W] -> [H, W]

        pos = np.sum(gt_np > 0.5)
        neg = np.sum(gt_np <= 0.5)

        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        print("Warning: no positive pixels found when computing pos_weight. Using 1.0.")
        return torch.tensor([1.0], dtype=torch.float32)

    ratio = total_neg / max(1, total_pos)
    print(f"Computed pos_weight = {ratio:.2f} (neg/pos ratio)")
    return torch.tensor([ratio], dtype=torch.float32)


def build_datasets_and_loaders():
    """
    Build train/val datasets and dataloaders.

    Strategy:
    - Instantiate a base dataset to compute global stats and list all fire_dirs.
    - Split indices into train/val.
    - Instantiate two new datasets with:
        - initial_fire_dirs = split-specific fire_dirs
        - apply_augmentations=True for train, False for val
        - global_stats reused from the base dataset
    """
    print("Initializing base dataset to compute global stats and discover fire directories...")
    base_dataset = PiedmontDataset(
        root_dir=ROOT_DIR,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=True,
        apply_augmentations=True,
        global_stats=None,
        initial_fire_dirs=None,
    )

    num_samples = len(base_dataset)
    train_size = int(TRAIN_SPLIT * num_samples)
    val_size = num_samples - train_size

    print(f"Total samples: {num_samples} -> Train: {train_size}, Val: {val_size}")

    # Use simple shuffled split
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Map indices to fire_dirs so we can create new datasets with different augmentation settings
    train_fire_dirs = [base_dataset.fire_dirs[i] for i in train_indices]
    val_fire_dirs = [base_dataset.fire_dirs[i] for i in val_indices]

    # Train dataset: augmentations ON, reuse global_stats
    train_dataset = PiedmontDataset(
        root_dir=ROOT_DIR,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False,
        apply_augmentations=True,
        global_stats=base_dataset.global_stats,
        initial_fire_dirs=train_fire_dirs,
    )

    # Validation dataset: augmentations OFF, reuse global_stats
    val_dataset = PiedmontDataset(
        root_dir=ROOT_DIR,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False,
        apply_augmentations=False,
        global_stats=base_dataset.global_stats,
        initial_fire_dirs=val_fire_dirs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model():
    """
    Instantiate the MultiModalFPN model for the Sentinel-only baseline.

    The model will accept Sentinel + placeholder multimodal inputs but internally
    only use Sentinel features in this baseline.
    """
    model = MultiModalFPN(
        in_channels_sentinel=12,          # Sentinel-2 bands
        in_channels_landsat=16,           # Landsat + flag (even if zero in baseline)
        in_channels_other_data=3,         # DEM + streets (concatenated)
        in_channels_era5_raster=2,        # as defined in model.py
        in_channels_era5_tabular=1,
        in_channels_ignition_map=1,
        num_classes=1,                    # burned area (binary)
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        landcover_classes=12,
    )
    return model


def main():
    set_seed(42)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR_BASE, exist_ok=True)

    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR_BASE, f"run_{current_time}")
    writer = SummaryWriter(log_dir)

    print(f"Logging to: {log_dir}")

    # Build data
    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders()

    # Compute pos_weight for imbalanced BCE
    pos_weight = compute_pos_weight(train_dataset).to(DEVICE)

    # Build model and move to device
    model = build_model().to(DEVICE)

    # Optimizer & scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE_MAIN,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=5e-5
    )

    # Loss functions
    burned_area_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    landcover_loss_fn = nn.CrossEntropyLoss()  # unused baseline, kept for API

    # Early stopping on validation IoU (burned area)
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    best_iou = -1.0

    # Log experiment config
    experiment_notes = f"""
Experiment time: {current_time}
- Dataset root: {ROOT_DIR}
- GeoJSON: {GEOJSON_PATH}
- Model: MultiModalFPN (encoder={ENCODER_NAME}, weights={ENCODER_WEIGHTS})
- Input:
    Sentinel: 12 bands
    Landsat: placeholder (not used in baseline)
    DEM + Streets: placeholder (not used in baseline)
    ERA5 raster + tabular: placeholder (not used in baseline)
- Split: {int(TRAIN_SPLIT * 100)}% train / {int((1 - TRAIN_SPLIT) * 100)}% val
- Loss: BCEWithLogits (burned area) with pos_weight={pos_weight.item():.2f}, CrossEntropy (landcover)
- Optimizer: Adam (lr={LEARNING_RATE_MAIN}, weight_decay={WEIGHT_DECAY})
- Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS}, eta_min=5e-5)
- Epochs: {NUM_EPOCHS} with EarlyStopping(patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA})
- Batch size: {BATCH_SIZE}
"""
    writer.add_text("Experiment/Notes", experiment_notes, global_step=0)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_total_loss, train_ba_loss, train_lc_loss, train_iou_ba = train(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            burned_area_loss_fn=burned_area_loss_fn,
            landcover_loss_fn=landcover_loss_fn,
            device=DEVICE,
            writer=writer,
            epoch=epoch,
            w_mask=W_MASK,
            w_landcover=W_LANDCOVER,
        )

        print(
            f"  Train - total loss: {train_total_loss:.4f} | "
            f"mask loss: {train_ba_loss:.4f} | landcover loss: {train_lc_loss:.4f} | "
            f"IoU BA: {train_iou_ba:.4f}"
        )

        val_total_loss, val_ba_loss, val_lc_loss, val_iou_ba = val(
            model=model,
            dataloader=val_loader,
            burned_area_loss_fn=burned_area_loss_fn,
            landcover_loss_fn=landcover_loss_fn,
            device=DEVICE,
            writer=writer,
            epoch=epoch,
        )

        print(
            f"  Val   - total loss: {val_total_loss:.4f} | "
            f"mask loss: {val_ba_loss:.4f} | landcover loss: {val_lc_loss:.4f} | "
            f"IoU BA: {val_iou_ba:.4f}"
        )

        # Update learning rate
        scheduler.step()

        # Checkpoint best model
        if val_iou_ba > best_iou:
            best_iou = val_iou_ba
            checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name)   
            torch.save(model.state_dict(), checkpoint_path)
            writer.add_text(
                "Checkpoint",
                f"New best model at epoch {epoch + 1} with IoU BA: {best_iou:.4f}",
                global_step=epoch + 1,
            )
            print(f"  >> New best model saved at {checkpoint_path} (IoU={best_iou:.4f})")

        # Early stopping
        stop_training = early_stopping.step(val_iou_ba, epoch)
        if stop_training:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(
                f"Best validation IoU: {early_stopping.best_metric:.4f} "
                f"(epoch {early_stopping.best_epoch + 1})"
            )
            break

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()