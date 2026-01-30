#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sentinel_to_rgb(sentinel_chw: np.ndarray) -> np.ndarray:
    """
    Make a pseudo-RGB from the first 3 channels.
    """
    c, h, w = sentinel_chw.shape
    if c >= 3:
        rgb = sentinel_chw[:3]
    else:
        rgb = np.repeat(sentinel_chw[:1], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)

    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = (rgb - lo) / (hi - lo + 1e-6)
    return np.clip(rgb, 0, 1)


def unwrap_state_dict(obj) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    cleaned = {}
    for k, v in obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def torch_load_any(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def landcover_to_labels(lc: np.ndarray) -> np.ndarray:
    """
    Accepts landcover as:
      - (H,W) integer labels
      - (C,H,W) one-hot / logits
      - (1,H,W) binary label map (treated as labels 0/1)
    Returns (H,W) int labels.
    """
    if lc.ndim == 2:
        return lc.astype(np.int64)
    if lc.ndim == 3:
        if lc.shape[0] == 1:
            return (lc[0] > 0.5).astype(np.int64)
        return np.argmax(lc, axis=0).astype(np.int64)
    raise ValueError(f"Unsupported landcover shape: {lc.shape}")


def compute_confusion(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix for labels in [0..num_classes-1]
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mask = (yt >= 0) & (yt < num_classes)
    yt = yt[mask]
    yp = yp[mask]
    for t, p in zip(yt, yp):
        if 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_iou(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    IoU per class from confusion matrix.
    IoU_k = TP / (TP + FP + FN)
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    iou = tp / (tp + fp + fn + 1e-8)
    miou = float(np.nanmean(iou))
    return iou, miou


# ---------------------------
# Model forward
# ---------------------------
@torch.no_grad()
def forward_landcover(
    model: torch.nn.Module,
    sentinel: torch.Tensor,
    landsat: torch.Tensor,
    other: torch.Tensor,
    era5_r: torch.Tensor,
    era5_t: torch.Tensor,
) -> torch.Tensor:
    """
    Returns landcover logits (B, C, H, W).
    Supports:
      - multimodal: forward(sentinel, landsat, other, ignition, era5_r, era5_t) -> (burned, landcover)
      - sentinel-only: forward(sentinel) -> ??? (no landcover)
    """
    # Determine if model expects multimodal by trying call signatures safely
    # We avoid introspection tricks and just try the multimodal call first.
    try:
        b, _, h, w = sentinel.shape
        ignition = torch.zeros((b, 1, h, w), device=sentinel.device, dtype=sentinel.dtype)
        out = model(sentinel, landsat, other, ignition, era5_r, era5_t)
    except TypeError:
        out = model(sentinel)

    if not isinstance(out, (tuple, list)):
        raise RuntimeError("Model output is not a tuple/list; cannot find auxiliary landcover head output.")

    if len(out) < 2:
        raise RuntimeError("Model output tuple/list has <2 elements; landcover head output not found.")

    land_logits = out[1]
    if land_logits.dim() != 4:
        raise RuntimeError(f"Expected landcover logits (B,C,H,W). Got shape: {tuple(land_logits.shape)}")
    return land_logits


# ---------------------------
# Plotting
# ---------------------------
def save_landcover_panel(
    out_path: str,
    rgb_hw3: np.ndarray,
    gt_hw: np.ndarray,
    pred_hw: np.ndarray,
    title: str,
) -> None:
    """
    4-panel: RGB, GT landcover, Pred landcover, Agreement map
    """
    agree = (gt_hw == pred_hw).astype(np.uint8)

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(title, fontsize=12)

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(rgb_hw3)
    ax1.set_title("Pseudo-RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_hw)
    ax2.set_title("GT landcover")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(pred_hw)
    ax3.set_title("Pred landcover")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(agree, vmin=0, vmax=1)
    ax4.set_title("Correct pixels")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_class_histogram(
    out_path: str,
    num_classes: int,
    gt_labels: List[np.ndarray],
    pred_labels: List[np.ndarray],
    title: str,
) -> None:
    gt_all = np.concatenate([x.reshape(-1) for x in gt_labels], axis=0)
    pr_all = np.concatenate([x.reshape(-1) for x in pred_labels], axis=0)

    gt_counts = np.bincount(gt_all.clip(0, num_classes - 1), minlength=num_classes)
    pr_counts = np.bincount(pr_all.clip(0, num_classes - 1), minlength=num_classes)

    x = np.arange(num_classes)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - 0.2, gt_counts, width=0.4, label="GT")
    ax.bar(x + 0.2, pr_counts, width=0.4, label="Pred")
    ax.set_title(title)
    ax.set_xlabel("landcover class index")
    ax.set_ylabel("pixel count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_confusion_heatmap(out_path: str, cm: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoint_main/best_model_4.pth", help="Checkpoint with auxiliary head")
    p.add_argument("--out_dir", type=str, default="Results/landcover_demo", help="Output folder")
    p.add_argument("--num_samples", type=int, default=2, help="Number of example images to save (you want 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full val loader; >0 = limit")
    p.add_argument("--full_eval", type=int, default=0, help="1 = compute mIoU over full val set (slower); 0 = skip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # device
    try:
        from main import DEVICE
        device = DEVICE
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    from main import build_datasets_and_loaders
    _train_ds, val_ds, _train_loader, val_loader = build_datasets_and_loaders()

    # model
    from main import build_model
    model = build_model().to(device)

    state = unwrap_state_dict(torch_load_any(args.ckpt, device))
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys (strict=False), sample: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (strict=False), sample: {unexpected[:5]}")

    model.eval()

    ensure_dir(args.out_dir)
    figs_dir = os.path.join(args.out_dir, "figures")
    samples_dir = os.path.join(args.out_dir, "samples")
    ensure_dir(figs_dir)
    ensure_dir(samples_dir)

    # Pick 2 samples: one with high burned fraction, one random
    # (This makes the presentation feel intentional.)
    burn_fracs = []
    for i in range(len(val_ds)):
        try:
            sample = val_ds[i]
            gt = sample[-1].detach().cpu().numpy()
            burn_fracs.append(float((gt > 0.5).mean()))
        except Exception:
            burn_fracs.append(0.0)

    idx_sorted = np.argsort(np.array(burn_fracs))
    idx_high = int(idx_sorted[-1]) if len(idx_sorted) > 0 else 0
    idx_rand = int(random.randrange(len(val_ds))) if len(val_ds) > 0 else 0

    chosen = [idx_high, idx_rand]
    chosen = chosen[: max(1, args.num_samples)]

    gt_list = []
    pred_list = []

    # Run and save the 2 example panels
    for j, idx in enumerate(chosen):
        sentinel, landsat, other, era5_r, era5_t, landcover_gt, gt_ba = val_ds[idx]

        # tensors
        s = sentinel.unsqueeze(0).to(device)
        l = landsat.unsqueeze(0).to(device)
        o = other.unsqueeze(0).to(device)
        er = era5_r.unsqueeze(0).to(device)
        et = era5_t.unsqueeze(0).to(device)

        land_logits = forward_landcover(model, s, l, o, er, et)  # (1,C,H,W)
        pred = torch.argmax(land_logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)

        lc_gt_np = landcover_gt.detach().cpu().numpy()
        gt_labels = landcover_to_labels(lc_gt_np)

        rgb = sentinel_to_rgb(sentinel.detach().cpu().numpy())

        gt_list.append(gt_labels)
        pred_list.append(pred)

        out_path = os.path.join(samples_dir, f"landcover_demo_{j+1:02d}_idx{idx}.png")
        title = f"Landcover demo (idx={idx}) | burned_frac={burn_fracs[idx]:.3f}"
        save_landcover_panel(out_path, rgb, gt_labels, pred, title)

    # Derive number of classes from logits (safe)
    # (Using last land_logits from above, or do a mini forward if needed.)
    num_classes = int(land_logits.shape[1])  # type: ignore[name-defined]

    # Small summary figure across the selected 2 samples
    save_class_histogram(
        out_path=os.path.join(figs_dir, "landcover_class_hist_selected_samples.png"),
        num_classes=num_classes,
        gt_labels=gt_list,
        pred_labels=pred_list,
        title="Landcover class distribution (GT vs Pred) on selected samples",
    )

    # Optional: full validation landcover mIoU
    if int(args.full_eval) == 1:
        max_batches = None if args.max_batches == 0 else int(args.max_batches)
        cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

        batch_count = 0
        for batch in val_loader:
            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                break

            sentinel = batch[0].to(device)
            landsat = batch[1].to(device)
            other = batch[2].to(device)
            era5_r = batch[3].to(device)
            era5_t = batch[4].to(device)
            land_gt = batch[5].detach().cpu().numpy()  # on CPU ok

            land_logits_b = forward_landcover(model, sentinel, landsat, other, era5_r, era5_t)
            pred_b = torch.argmax(land_logits_b, dim=1).detach().cpu().numpy().astype(np.int64)

            b = pred_b.shape[0]
            for i in range(b):
                gt_i = landcover_to_labels(land_gt[i])
                pr_i = pred_b[i]
                cm_total += compute_confusion(num_classes, gt_i, pr_i)

        iou_c, miou = per_class_iou(cm_total)
        np.savetxt(os.path.join(figs_dir, "landcover_per_class_iou.txt"), iou_c, fmt="%.6f")
        with open(os.path.join(figs_dir, "landcover_miou.txt"), "w", encoding="utf-8") as f:
            f.write(f"mIoU={miou:.6f}\n")

        save_confusion_heatmap(
            os.path.join(figs_dir, "landcover_confusion_matrix.png"),
            cm_total,
            title="Landcover confusion matrix (val set)",
        )

        print(f"[OK] Landcover full-eval mIoU: {miou:.4f}")

    print("\n[OK] Landcover demo outputs written to:", args.out_dir)
    print("Saved 2 sample panels in:", samples_dir)
    print("Saved summary figures in:", figs_dir)


if __name__ == "__main__":
    main()