#!/usr/bin/env python3
"""
modality_ablation_quick.py

RQ3: Which input modalities contribute most?

Method (fast): inference-time ablation
- Evaluate the trained multimodal model normally ("FULL")
- Then zero-out one modality at a time and re-evaluate on same val subset
- Compare drops in IoU/F1

CPU-friendly: small subset & limited thresholds.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

OUT_DIR = "Results/modality_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Fast CPU settings (MacBook friendly)
DEVICE = torch.device("cpu")
BATCH_SIZE = 2
NUM_WORKERS = 0
MAX_VAL_SAMPLES = 120  # keep small for speed; increase later if you want

# thresholds to test; best threshold chosen among these
THRESHOLDS = [0.50, 0.75, 0.90, 0.95]

# ---- Update this if your checkpoint path differs
# Try common paths:
CHECKPOINT_CANDIDATES = [
    "checkpoint_2/best_model_3.pth",
]

# ----------------------------
# Metrics helpers
# ----------------------------
def fast_confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(iou), float(precision), float(recall), float(f1)


def find_checkpoint() -> str:
    for p in CHECKPOINT_CANDIDATES:
        if os.path.exists(p):
            return p
    # last resort: search for best_model*.pth
    for root, _, files in os.walk("."):
        for f in files:
            if f.startswith("best_model") and f.endswith(".pth"):
                return os.path.join(root, f)
    raise FileNotFoundError("Could not find a checkpoint. Update CHECKPOINT_CANDIDATES.")


def plot_drop_bars(df: pd.DataFrame, out_path: str):
    """
    Plot drop in IoU/F1 relative to FULL (positive bars mean worse).
    """
    full_iou = float(df.loc[df["Setting"] == "FULL", "IoU_best"].iloc[0])
    full_f1 = float(df.loc[df["Setting"] == "FULL", "F1_best"].iloc[0])

    rows = []
    for _, r in df.iterrows():
        if r["Setting"] == "FULL":
            continue
        rows.append({
            "Setting": r["Setting"],
            "IoU_drop": full_iou - float(r["IoU_best"]),
            "F1_drop": full_f1 - float(r["F1_best"]),
        })

    dd = pd.DataFrame(rows).sort_values("IoU_drop", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(dd))
    w = 0.4
    ax.bar(x - w/2, dd["IoU_drop"], w, label="IoU drop")
    ax.bar(x + w/2, dd["F1_drop"], w, label="F1 drop")
    ax.set_xticks(x)
    ax.set_xticklabels(dd["Setting"], rotation=25, ha="right")
    ax.set_ylabel("Drop vs FULL (higher = more important modality)")
    ax.set_title("Inference-time Modality Ablation (Val subset)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Core eval
# ----------------------------
@torch.no_grad()
def eval_model_with_ablation(model, val_loader, thresholds: List[float], ablation: str, max_samples: int):
    """
    ablation options:
      - "FULL"
      - "NO_LANDSAT"
      - "NO_OTHER"     (e.g., DEM+roads)
      - "NO_ERA5"
      - "NO_IGNITION"
      - "SENTINEL_ONLY" (all extras zeroed)
    """
    model.eval()

    accum = {thr: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for thr in thresholds}

    seen = 0
    for batch in val_loader:
        # Expected batch structure (your dataset):
        # (sentinel, landsat, other, era5_raster, era5_tabular, landcover_input, gt_mask)
        sentinel, landsat, other_data, era5_raster, era5_tabular, landcover_input, gt_mask = batch

        bs = sentinel.shape[0]
        if seen >= max_samples:
            break

        sentinel = sentinel.to(DEVICE)
        landsat = landsat.to(DEVICE)
        other_data = other_data.to(DEVICE)
        era5_raster = era5_raster.to(DEVICE)
        era5_tabular = era5_tabular.to(DEVICE)
        gt_mask = gt_mask.to(DEVICE)

        # ensure gt channel
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)

        # ignition map is not in dataset batch; we create zeros by default
        ignition = torch.zeros_like(gt_mask, device=DEVICE)

        # Apply ablation by zeroing modality tensors
        if ablation == "NO_LANDSAT":
            landsat = torch.zeros_like(landsat)
        elif ablation == "NO_OTHER":
            other_data = torch.zeros_like(other_data)
        elif ablation == "NO_ERA5":
            era5_raster = torch.zeros_like(era5_raster)
            era5_tabular = torch.zeros_like(era5_tabular)
        elif ablation == "NO_IGNITION":
            ignition = torch.zeros_like(ignition)
        elif ablation == "SENTINEL_ONLY":
            landsat = torch.zeros_like(landsat)
            other_data = torch.zeros_like(other_data)
            era5_raster = torch.zeros_like(era5_raster)
            era5_tabular = torch.zeros_like(era5_tabular)
            ignition = torch.zeros_like(ignition)
        elif ablation == "FULL":
            pass
        else:
            raise ValueError(f"Unknown ablation setting: {ablation}")

        # Forward
        out = model(sentinel, landsat, other_data, ignition, era5_raster, era5_tabular)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = (gt_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)

        probs_f = probs.reshape(-1)
        y_true_f = y_true.reshape(-1)

        for thr in thresholds:
            y_pred_f = (probs_f > thr).astype(np.uint8)
            tp, tn, fp, fn = fast_confusion_binary(y_true_f, y_pred_f)
            accum[thr]["tp"] += tp
            accum[thr]["tn"] += tn
            accum[thr]["fp"] += fp
            accum[thr]["fn"] += fn

        seen += bs

    # pick best threshold by IoU
    rows = []
    for thr in thresholds:
        tp, tn, fp, fn = accum[thr]["tp"], accum[thr]["tn"], accum[thr]["fp"], accum[thr]["fn"]
        iou, p, r, f1 = metrics_from_confusion(tp, tn, fp, fn)
        rows.append({"thr": thr, "IoU": iou, "Precision": p, "Recall": r, "F1": f1})

    df = pd.DataFrame(rows).sort_values("IoU", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return best, df


def main():
    print(f"[info] device={DEVICE}")

    # Import your project utilities
    from torch.utils.data import DataLoader
    from main import build_datasets_and_loaders, set_seed
    from model import MultiModalFPN  # your repo model

    set_seed(42)

    # Build datasets and a CPU-friendly val_loader
    _, val_dataset, _, _ = build_datasets_and_loaders()
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Build model (must match your training config)
    # If your main.py has build_model(), use that instead (more reliable).
    # We'll try to import build_model from main first.
    model = None
    try:
        from main import build_model
        model = build_model().to(DEVICE)
        print("[info] model built using main.build_model()")
    except Exception:
        # fallback: instantiate directly (only if needed)
        model = MultiModalFPN(
            in_channels_sentinel=12,
            in_channels_landsat=16,
            in_channels_other_data=3,
            in_channels_era5_raster=2,
            in_channels_era5_tabular=1,
            in_channels_ignition_map=1,
            num_classes=1,
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            landcover_classes=12,
        ).to(DEVICE)
        print("[warn] model built using fallback MultiModalFPN(...) — ensure it matches training.")

    # Load checkpoint
    ckpt_path = find_checkpoint()
    print(f"[info] loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    # support both state_dict-only and dict format
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    # Ablation settings
    settings = ["FULL", "NO_LANDSAT", "NO_OTHER", "NO_ERA5", "NO_IGNITION", "SENTINEL_ONLY"]

    results = []
    for s in settings:
        print(f"\n[run] {s} ...")
        best, per_thr = eval_model_with_ablation(
            model=model,
            val_loader=val_loader,
            thresholds=THRESHOLDS,
            ablation=s,
            max_samples=MAX_VAL_SAMPLES,
        )
        results.append({
            "Setting": s,
            "Best_thr": best["thr"],
            "IoU_best": best["IoU"],
            "F1_best": best["F1"],
            "Precision_best": best["Precision"],
            "Recall_best": best["Recall"],
        })
        # save per-threshold details for each setting (optional)
        per_thr.to_csv(os.path.join(OUT_DIR, f"per_threshold_{s}.csv"), index=False)

        print(f"  best_thr={best['thr']:.2f} | IoU={best['IoU']:.4f} | F1={best['F1']:.4f} | P={best['Precision']:.4f} | R={best['Recall']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False)

    # Plot drop chart
    plot_drop_bars(df, os.path.join(OUT_DIR, "ablation_drop_bars.png"))

    # Rank modalities by IoU drop
    full_iou = float(df.loc[df["Setting"] == "FULL", "IoU_best"].iloc[0])
    rank = []
    for _, r in df.iterrows():
        if r["Setting"] == "FULL":
            continue
        rank.append((r["Setting"], full_iou - float(r["IoU_best"])))
    rank.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Importance ranking (by IoU drop vs FULL) ===")
    for name, drop in rank:
        print(f"{name:15s}  IoU_drop={drop:.4f}")

    print("\n[OK] Outputs saved to:", OUT_DIR)
    print(" - ablation_results.csv")
    print(" - ablation_drop_bars.png")


if __name__ == "__main__":
    main()