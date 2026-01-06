#!/usr/bin/env python3
"""
CPU-fast Baseline (U-Net Sentinel-only) vs Multimodal comparison.

Fixes your error by:
- Loading baseline checkpoint into the correct architecture (smp.Unet),
  not MultiModalFPN.

Outputs:
  Results/model_comparison/comparison_table.png
  Results/model_comparison/comparison_metrics_bars.png
  Results/model_comparison/baseline_quick_thresholds.csv
  Results/model_comparison/comparison_metrics.csv
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ----------------------------
# Hardcoded paths
# ----------------------------
OUT_DIR = "Results/model_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_CKPT = "Baseline_model/unet_sentinel_best.pth"

METRICS_CANDIDATE_PATHS = [
    "slide_figures/tables/metrics_summary.csv",
    "slide_figures/tables/metrics_summary.csv",
    "metrics_summary.csv",
]

# CPU quick settings (MacBook friendly)
CPU_BATCH_SIZE = 2
CPU_NUM_WORKERS = 0
MAX_VAL_SAMPLES = 120          # keep small for speed; increase later if needed
THRESHOLDS = [0.50, 0.75, 0.90, 0.95]

# Baseline model assumptions (from your config)
BASELINE_ENCODER = "resnet34"
BASELINE_ENCODER_WEIGHTS = "imagenet"
BASELINE_IN_CHANNELS = 12
BASELINE_CLASSES = 1

# ----------------------------
# Setup table inputs (from your configs)
# ----------------------------
BASELINE_CONFIG = {
    "Architecture": "U-Net (Sentinel-only)",
    "Encoder": "resnet34 (imagenet)",
    "Input modalities": "Sentinel-2 only (12 bands)",
    "Batch size": 4,
    "Epochs": 40,
    "Learning rate": 1e-4,
    "Weight decay": 5e-4,
    "Early stopping patience": 20,
    "Early stopping min_delta": 1e-3,
    "Input size": "256×256",
}

MULTIMODAL_CONFIG = {
    "Architecture": "Multimodal model",
    "Encoder": "efficientnet-b4 (imagenet)",
    "Input modalities": "Sentinel-2 + multimodal",
    "Batch size": 4,
    "Epochs": 120,
    "Learning rate": 3e-4,
    "Weight decay": 1e-4,
    "Early stopping patience": 12,
    "Early stopping min_delta": 5e-4,
    "Input size": "256×256",
}

# ----------------------------
# Helper functions
# ----------------------------
def find_metrics_csv() -> str:
    for p in METRICS_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("."):
        if "metrics_summary.csv" in files:
            return os.path.join(root, "metrics_summary.csv")
    raise FileNotFoundError("Could not find metrics_summary.csv anywhere in this repo.")


def normalize_setting(s: str) -> str:
    return str(s).strip().lower()


def load_multimodal_metrics_csv(path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    required = {"Setting", "Threshold", "IoU", "Precision", "Recall", "F1"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"metrics_summary.csv must have columns {required}, got {df.columns.tolist()}")

    out = {}
    for _, r in df.iterrows():
        key = normalize_setting(r["Setting"])
        out[key] = {
            "Threshold": float(r["Threshold"]),
            "IoU": float(r["IoU"]),
            "Precision": float(r["Precision"]),
            "Recall": float(r["Recall"]),
            "F1": float(r["F1"]),
        }
    return out


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


def save_table_png(df: pd.DataFrame, path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 2.4 + 0.45 * len(df)))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.0, 1.4)
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_metric_bar_chart(metric_rows: List[Dict[str, float]], path: str, title: str):
    df = pd.DataFrame(metric_rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["Baseline"], width, label="Baseline")
    ax.bar(x + width/2, df["Multimodal"], width, label="Multimodal")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Strip a leading prefix from state_dict keys if present."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_baseline_unet_quick(
    model: torch.nn.Module,
    val_loader,
    thresholds: List[float],
    device: torch.device,
    max_samples: int,
) -> pd.DataFrame:
    """
    Expects val_loader batches from your project dataset structure:
      (sentinel, landsat, other, era5_raster, era5_tabular, landcover, gt_mask)

    For baseline U-Net: we only use sentinel + gt_mask.
    """
    accum = {thr: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for thr in thresholds}
    model.eval()

    seen = 0
    for batch in val_loader:
        sentinel = batch[0]
        gt_mask = batch[-1]

        bs = sentinel.shape[0]
        if seen >= max_samples:
            break

        sentinel = sentinel.to(device)
        gt_mask = gt_mask.to(device)

        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)

        logits = model(sentinel)
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

    rows = []
    for thr in thresholds:
        tp, tn, fp, fn = accum[thr]["tp"], accum[thr]["tn"], accum[thr]["fp"], accum[thr]["fn"]
        iou, p, r, f1 = metrics_from_confusion(tp, tn, fp, fn)
        rows.append({"thr": thr, "IoU": iou, "Precision": p, "Recall": r, "F1": f1})

    return pd.DataFrame(rows)


def main():
    # Always CPU for your MacBook Air
    device = torch.device("cpu")
    print(f"[info] device = {device}")

    # Use your project pipeline to get val_dataset
    from main import build_datasets_and_loaders, set_seed
    from torch.utils.data import DataLoader

    set_seed(42)
    _, val_dataset, _, _ = build_datasets_and_loaders()

    val_loader = DataLoader(
        val_dataset,
        batch_size=CPU_BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_NUM_WORKERS,
        pin_memory=False,
    )

    # Load multimodal metrics
    metrics_path = find_metrics_csv()
    mm = load_multimodal_metrics_csv(metrics_path)
    print(f"[info] loaded multimodal metrics from: {metrics_path}")

    mm_best = mm.get("best threshold (val)")
    mm_def = mm.get("default threshold")
    if mm_best is None or mm_def is None:
        raise ValueError("metrics_summary.csv must include 'Best threshold (val)' and 'Default threshold' rows.")

    # Build baseline U-Net
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise ImportError(
            "segmentation_models_pytorch is required for baseline U-Net evaluation.\n"
            "Install: pip install segmentation-models-pytorch"
        ) from e

    baseline_model = smp.Unet(
        encoder_name=BASELINE_ENCODER,
        encoder_weights=BASELINE_ENCODER_WEIGHTS,
        in_channels=BASELINE_IN_CHANNELS,
        classes=BASELINE_CLASSES,
    ).to(device)

    if not os.path.exists(BASELINE_CKPT):
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CKPT}")

    # Load baseline weights (state_dict only). Handle 'unet.' prefix in keys.
    state_dict = torch.load(BASELINE_CKPT, map_location=device)  # your file, safe enough
    state_dict = strip_prefix(state_dict, "unet.")
    missing, unexpected = baseline_model.load_state_dict(state_dict, strict=False)

    # If too many missing keys, something is still off
    print(f"[info] baseline load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 50:
        print("[warn] Many missing keys. Baseline checkpoint might not match smp.Unet exactly.")
        print("[warn] Missing example:", missing[:10])
        print("[warn] Unexpected example:", unexpected[:10])

    # Evaluate baseline quickly
    print(f"[info] evaluating baseline quickly: max_samples={MAX_VAL_SAMPLES}, thresholds={THRESHOLDS}")
    base_df = evaluate_baseline_unet_quick(
        model=baseline_model,
        val_loader=val_loader,
        thresholds=THRESHOLDS,
        device=device,
        max_samples=MAX_VAL_SAMPLES,
    )
    base_df.to_csv(os.path.join(OUT_DIR, "baseline_quick_thresholds.csv"), index=False)

    # Pick baseline best among tested thresholds + thr=0.50 row
    base_best = base_df.sort_values("IoU", ascending=False).iloc[0].to_dict()
    base_thr05_df = base_df[base_df["thr"] == 0.50]
    base_thr05 = base_thr05_df.iloc[0].to_dict() if len(base_thr05_df) else None

    # Setup table (slide-friendly)
    setup_rows = []
    for k in BASELINE_CONFIG:
        setup_rows.append([k, BASELINE_CONFIG[k], MULTIMODAL_CONFIG[k]])
    setup_table = pd.DataFrame(setup_rows, columns=["Field", "Baseline", "Multimodal"])
    setup_table.to_csv(os.path.join(OUT_DIR, "comparison_setup.csv"), index=False)
    save_table_png(setup_table, os.path.join(OUT_DIR, "comparison_table.png"),
                   "Baseline vs Multimodal — Training Setup")

    # Metrics comparison (headline)
    metric_rows = [
        {"Metric": "IoU (best thr)", "Baseline": float(base_best["IoU"]), "Multimodal": float(mm_best["IoU"])},
        {"Metric": "F1 (best thr)", "Baseline": float(base_best["F1"]), "Multimodal": float(mm_best["F1"])},
        {"Metric": "Precision (best thr)", "Baseline": float(base_best["Precision"]), "Multimodal": float(mm_best["Precision"])},
        {"Metric": "Recall (best thr)", "Baseline": float(base_best["Recall"]), "Multimodal": float(mm_best["Recall"])},
    ]
    if base_thr05 is not None:
        metric_rows += [
            {"Metric": "IoU (thr=0.50)", "Baseline": float(base_thr05["IoU"]), "Multimodal": float(mm_def["IoU"])},
            {"Metric": "F1 (thr=0.50)", "Baseline": float(base_thr05["F1"]), "Multimodal": float(mm_def["F1"])},
        ]

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "comparison_metrics.csv"), index=False)
    save_metric_bar_chart(metric_rows, os.path.join(OUT_DIR, "comparison_metrics_bars.png"),
                          "Baseline vs Multimodal — Validation Metrics")

    # Terminal summary
    print("\n=== Multimodal (from metrics_summary.csv) ===")
    print(f"Best thr(val) = {mm_best['Threshold']:.2f} | IoU={mm_best['IoU']:.4f} | P={mm_best['Precision']:.4f} | R={mm_best['Recall']:.4f} | F1={mm_best['F1']:.4f}")
    print(f"Default thr   = {mm_def['Threshold']:.2f} | IoU={mm_def['IoU']:.4f} | P={mm_def['Precision']:.4f} | R={mm_def['Recall']:.4f} | F1={mm_def['F1']:.4f}")

    print("\n=== Baseline (quick eval on subset, U-Net) ===")
    print(f"Tested thresholds: {THRESHOLDS}")
    print(f"Best(thr among tested) = {base_best['thr']:.2f} | IoU={base_best['IoU']:.4f} | P={base_best['Precision']:.4f} | R={base_best['Recall']:.4f} | F1={base_best['F1']:.4f}")
    if base_thr05 is not None:
        print(f"Thr=0.50 = IoU={base_thr05['IoU']:.4f} | P={base_thr05['Precision']:.4f} | R={base_thr05['Recall']:.4f} | F1={base_thr05['F1']:.4f}")

    print("\n[OK] Slide-ready outputs saved to:", OUT_DIR)
    print(" - comparison_table.png")
    print(" - comparison_metrics_bars.png")
    print(" - baseline_quick_thresholds.csv")
    print(" - comparison_metrics.csv")


if __name__ == "__main__":
    main()