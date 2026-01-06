import os
import numpy as np
import torch
import torch.nn as nn

from main import (
    build_datasets_and_loaders,
    build_model,
    DEVICE,
    CHECKPOINT_DIR,
    model_name,
    set_seed,
)

def fast_confusion_binary(y_true: np.ndarray, y_pred: np.ndarray):
    # y_true,y_pred are {0,1} arrays
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def metrics_from_confusion(tp, tn, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return iou, precision, recall, f1

@torch.no_grad()
def evaluate_thresholds(model, dataloader, thresholds):
    # Accumulate pixel-level confusion over the whole val set for each threshold
    # To keep memory reasonable, we compute per-batch and accumulate.
    results = {thr: {"tp":0, "tn":0, "fp":0, "fn":0} for thr in thresholds}

    for batch in dataloader:
        (
            image_sentinel,
            image_landsat,
            other_data,
            era5_raster,
            era5_tabular,
            landcover_input,
            gt_mask,
        ) = batch

        image_sentinel = image_sentinel.to(DEVICE, non_blocking=True)
        image_landsat  = image_landsat.to(DEVICE, non_blocking=True)
        other_data     = other_data.to(DEVICE, non_blocking=True)
        era5_raster    = era5_raster.to(DEVICE, non_blocking=True)
        era5_tabular   = era5_tabular.to(DEVICE, non_blocking=True)
        gt_mask        = gt_mask.to(DEVICE, non_blocking=True)

        if gt_mask.dim() == 3:
            gt_mask_ch = gt_mask.unsqueeze(1)
        else:
            gt_mask_ch = gt_mask

        ignition_pt = torch.zeros_like(gt_mask_ch, device=DEVICE)

        logits_ba, _ = model(
            image_sentinel,
            image_landsat,
            other_data,
            ignition_pt,
            era5_raster,
            era5_tabular,
        )

        probs = torch.sigmoid(logits_ba).detach().cpu().numpy()
        y_true = (gt_mask_ch.detach().cpu().numpy() > 0.5).astype(np.uint8)

        # Flatten once
        probs_f = probs.reshape(-1)
        y_true_f = y_true.reshape(-1)

        for thr in thresholds:
            y_pred_f = (probs_f > thr).astype(np.uint8)
            tp, tn, fp, fn = fast_confusion_binary(y_true_f, y_pred_f)
            results[thr]["tp"] += int(tp)
            results[thr]["tn"] += int(tn)
            results[thr]["fp"] += int(fp)
            results[thr]["fn"] += int(fn)

    # Convert to metrics table
    table = []
    for thr in thresholds:
        tp = results[thr]["tp"]; tn = results[thr]["tn"]
        fp = results[thr]["fp"]; fn = results[thr]["fn"]
        iou, p, r, f1 = metrics_from_confusion(tp, tn, fp, fn)
        table.append((thr, iou, p, r, f1, tp, fp, fn, tn))
    return table

def main():
    set_seed(42)

    # Data
    _, _, _, val_loader = build_datasets_and_loaders()

    # Model + checkpoint
    model = build_model().to(DEVICE)
    ckpt = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    table = evaluate_thresholds(model, val_loader, thresholds)

    # Pick best IoU
    table_sorted = sorted(table, key=lambda x: x[1], reverse=True)
    best = table_sorted[0]

    # Find metrics at threshold 0.50
    thr05 = [row for row in table if abs(row[0] - 0.50) < 1e-9][0]

    print("=== PAPER-COMPARISON METRICS (VAL) ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Best IoU threshold sweep:")
    print(f"  thr={best[0]:.2f} | IoU={best[1]:.4f} | P={best[2]:.4f} | R={best[3]:.4f} | F1={best[4]:.4f}")
    print(f"Default threshold 0.50:")
    print(f"  thr={thr05[0]:.2f} | IoU={thr05[1]:.4f} | P={thr05[2]:.4f} | R={thr05[3]:.4f} | F1={thr05[4]:.4f}")

    # Optional: dump full table
    print("\nthr, IoU, Precision, Recall, F1")
    for thr, iou, p, r, f1, *_ in table_sorted:
        print(f"{thr:.2f}, {iou:.4f}, {p:.4f}, {r:.4f}, {f1:.4f}")

if __name__ == "__main__":
    main()