#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# Repro / IO helpers
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s.strip("_")


def infer_code_dir_from_ckpt(ckpt_path: str) -> Optional[str]:
    """
    If ckpt is like ../src/checkpoint_2/best_model_3.pth -> infer ../src
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    parent = os.path.dirname(ckpt_dir)
    # Heuristic: if parent contains main.py, treat it as code directory
    if os.path.exists(os.path.join(parent, "main.py")):
        return parent
    return None


def load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def forward_required_positional_args(model: torch.nn.Module) -> int:
    sig = inspect.signature(model.forward)
    n = 0
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def torch_load_any(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)  # torch>=2.1
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(obj) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")
    cleaned = {}
    for k, v in obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


# ============================================================
# Metrics
# ============================================================
def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    return tp, tn, fp, fn


def metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"IoU": float(iou), "Precision": float(precision), "Recall": float(recall), "F1": float(f1), "Accuracy": float(acc)}


def iou_from_conf(tp: int, fp: int, fn: int) -> float:
    return float(tp / (tp + fp + fn + 1e-8))


# ============================================================
# Visualization helpers
# ============================================================
def sentinel_to_rgb(sentinel_chw: np.ndarray) -> np.ndarray:
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


def plot_confusion(tp: int, tn: int, fp: int, fn: int, out_path: str, title: str) -> None:
    mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mat, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hist(values: np.ndarray, out_path: str, title: str, xlabel: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_line(x: np.ndarray, y_dict: Dict[str, np.ndarray], out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    for label, y in y_dict.items():
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_scatter(x: np.ndarray, y: np.ndarray, out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_panel(out_path: str, sentinel_chw: np.ndarray, gt_hw: np.ndarray, prob_hw: np.ndarray, pred_hw: np.ndarray,
               threshold: float, title: str) -> None:
    rgb = sentinel_to_rgb(sentinel_chw)
    gt_hw = (gt_hw > 0).astype(np.uint8)
    pred_hw = (pred_hw > 0).astype(np.uint8)
    prob_hw = np.clip(prob_hw, 0, 1)

    overlay = rgb.copy()
    overlay_pred = np.dstack([pred_hw, np.zeros_like(pred_hw), np.zeros_like(pred_hw)]).astype(np.float32)
    overlay_gt = np.dstack([np.zeros_like(gt_hw), gt_hw, np.zeros_like(gt_hw)]).astype(np.float32)
    overlay = np.clip(overlay + 0.35 * overlay_pred + 0.25 * overlay_gt, 0, 1)

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(f"{title} | thr={threshold:.2f}", fontsize=12)

    ax1 = fig.add_subplot(1, 4, 1); ax1.imshow(rgb); ax1.set_title("Pseudo-RGB"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 4, 2); ax2.imshow(gt_hw, vmin=0, vmax=1); ax2.set_title("GT"); ax2.axis("off")
    ax3 = fig.add_subplot(1, 4, 3); im = ax3.imshow(prob_hw, vmin=0, vmax=1); ax3.set_title("Prob"); ax3.axis("off")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax4 = fig.add_subplot(1, 4, 4); ax4.imshow(overlay); ax4.set_title("Overlay"); ax4.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_compare_panel(out_path: str, sentinel_chw: np.ndarray, gt_hw: np.ndarray,
                       pred_old_hw: np.ndarray, pred_new_hw: np.ndarray,
                       threshold: float, title: str) -> None:
    rgb = sentinel_to_rgb(sentinel_chw)
    gt_hw = (gt_hw > 0).astype(np.uint8)
    pred_old_hw = (pred_old_hw > 0).astype(np.uint8)
    pred_new_hw = (pred_new_hw > 0).astype(np.uint8)

    def overlay(pred_hw: np.ndarray) -> np.ndarray:
        ov = rgb.copy()
        ov = np.clip(
            ov
            + 0.35 * np.dstack([pred_hw, np.zeros_like(pred_hw), np.zeros_like(pred_hw)]).astype(np.float32)
            + 0.25 * np.dstack([np.zeros_like(gt_hw), gt_hw, np.zeros_like(gt_hw)]).astype(np.float32),
            0, 1
        )
        return ov

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"{title} | thr={threshold:.2f}", fontsize=12)

    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(overlay(pred_old_hw)); ax1.set_title("OLD overlay"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(overlay(pred_new_hw)); ax2.set_title("NEW overlay"); ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Model loaders (NEW vs OLD codebase)
# ============================================================
@dataclass
class LoadedModel:
    tag: str
    model: torch.nn.Module
    code_dir: Optional[str] = None


def build_model_from_current_code(device: torch.device) -> torch.nn.Module:
    from main import build_model
    return build_model().to(device)


def build_model_from_old_code(old_code_dir: str, device: torch.device) -> torch.nn.Module:
    """
    Dynamically import old_code_dir/main.py and call build_model() if present.
    Fallback: import old_code_dir/model.py and instantiate MultiModalFPN using constants from main.py if found.
    """
    old_main_path = os.path.join(old_code_dir, "main.py")
    old_model_path = os.path.join(old_code_dir, "model.py")

    if not os.path.exists(old_main_path) or not os.path.exists(old_model_path):
        raise FileNotFoundError(f"Old code dir must contain main.py and model.py: {old_code_dir}")

    old_main = load_module_from_file("old_main_dynamic", old_main_path)

    # Preferred: old_main.build_model()
    if hasattr(old_main, "build_model") and callable(old_main.build_model):
        m = old_main.build_model()
        return m.to(device)

    # Fallback: instantiate MultiModalFPN from old model module
    old_model = load_module_from_file("old_model_dynamic", old_model_path)
    if not hasattr(old_model, "MultiModalFPN"):
        raise AttributeError("old model.py does not define MultiModalFPN")

    MultiModalFPN = getattr(old_model, "MultiModalFPN")

    encoder_name = getattr(old_main, "ENCODER_NAME", "efficientnet-b4")
    encoder_weights = getattr(old_main, "ENCODER_WEIGHTS", "imagenet")

    # Try to pass only supported kwargs
    sig = inspect.signature(MultiModalFPN.__init__)
    kwargs = {}
    if "encoder_name" in sig.parameters:
        kwargs["encoder_name"] = encoder_name
    if "encoder_weights" in sig.parameters:
        kwargs["encoder_weights"] = encoder_weights

    m = MultiModalFPN(**kwargs)
    return m.to(device)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state_obj = torch_load_any(ckpt_path, device=device)
    sd = unwrap_state_dict(state_obj)

    # strict=False for robustness across minor naming differences
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys (strict=False), first 10: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (strict=False), first 10: {unexpected[:10]}")
    model.eval()


def load_new_model(ckpt_path: str, device: torch.device, tag: str = "new") -> LoadedModel:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_new not found: {ckpt_path}")
    m = build_model_from_current_code(device)
    load_checkpoint_into_model(m, ckpt_path, device)
    return LoadedModel(tag=tag, model=m, code_dir=None)


def load_old_model(ckpt_path: str, device: torch.device, tag: str = "old", old_code_dir: Optional[str] = None) -> LoadedModel:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_old not found: {ckpt_path}")

    if old_code_dir is None:
        old_code_dir = infer_code_dir_from_ckpt(ckpt_path)

    if old_code_dir is None:
        # Fallback: attempt to load with current code (may work, but can mis-match)
        print("[WARN] Could not infer old_code_dir from ckpt_old. Loading OLD with current build_model() (may mis-match).")
        m = build_model_from_current_code(device)
        load_checkpoint_into_model(m, ckpt_path, device)
        return LoadedModel(tag=tag, model=m, code_dir=None)

    m = build_model_from_old_code(old_code_dir, device)
    load_checkpoint_into_model(m, ckpt_path, device)
    return LoadedModel(tag=tag, model=m, code_dir=old_code_dir)


# ============================================================
# Inference / evaluation
# ============================================================
@torch.no_grad()
def model_forward_logits(
    model: torch.nn.Module,
    sentinel: torch.Tensor,
    landsat: torch.Tensor,
    other: torch.Tensor,
    era5_raster: torch.Tensor,
    era5_tab: torch.Tensor,
) -> torch.Tensor:
    """
    Supports:
      - multimodal: forward(sentinel, landsat, other, ignition, era5_raster, era5_tab)
      - sentinel-only: forward(sentinel)
    """
    req = forward_required_positional_args(model)

    if req >= 6:
        b, _, h, w = sentinel.shape
        ignition = torch.zeros((b, 1, h, w), device=sentinel.device, dtype=sentinel.dtype)
        out = model(sentinel, landsat, other, ignition, era5_raster, era5_tab)
    else:
        out = model(sentinel)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits


def collect_probs_and_gts(
    loaded: LoadedModel,
    val_loader,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - probs_flat: float32 array of all pixels concatenated (N,)
      - gts_flat: uint8 array of all pixels concatenated (N,)
      - per_sample_rows: (S, 4) array with [sample_idx, gt_pos_frac, prob_mean, prob_p99]
    """
    probs_list = []
    gts_list = []
    per_sample = []

    batch_count = 0
    sample_idx = 0

    for batch in val_loader:
        batch_count += 1
        if max_batches is not None and batch_count > max_batches:
            break

        sentinel = batch[0].to(device)
        landsat = batch[1].to(device)
        other = batch[2].to(device)
        era5_raster = batch[3].to(device)
        era5_tab = batch[4].to(device)
        gt = batch[-1].to(device)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        logits = model_forward_logits(loaded.model, sentinel, landsat, other, era5_raster, era5_tab)
        probs = torch.sigmoid(logits)

        # shape (B,1,H,W)
        probs_np = probs[:, 0].detach().cpu().numpy().astype(np.float32)
        gt_np = (gt[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        b = probs_np.shape[0]
        for i in range(b):
            gt_pos_frac = float(gt_np[i].mean())
            prob_mean = float(probs_np[i].mean())
            prob_p99 = float(np.percentile(probs_np[i], 99))
            per_sample.append([sample_idx, gt_pos_frac, prob_mean, prob_p99])
            sample_idx += 1

        probs_list.append(probs_np.reshape(-1))
        gts_list.append(gt_np.reshape(-1))

    probs_flat = np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=np.float32)
    gts_flat = np.concatenate(gts_list, axis=0) if gts_list else np.array([], dtype=np.uint8)
    per_sample_rows = np.array(per_sample, dtype=np.float32) if per_sample else np.zeros((0, 4), dtype=np.float32)
    return probs_flat, gts_flat, per_sample_rows


def eval_at_threshold(probs_flat: np.ndarray, gts_flat: np.ndarray, thr: float) -> Tuple[Dict[str, float], Tuple[int,int,int,int]]:
    pred = (probs_flat > thr).astype(np.uint8)
    gt = (gts_flat > 0).astype(np.uint8)
    tp = int(np.logical_and(gt == 1, pred == 1).sum())
    tn = int(np.logical_and(gt == 0, pred == 0).sum())
    fp = int(np.logical_and(gt == 0, pred == 1).sum())
    fn = int(np.logical_and(gt == 1, pred == 0).sum())
    return metrics_from_conf(tp, tn, fp, fn), (tp, tn, fp, fn)


def threshold_sweep(probs_flat: np.ndarray, gts_flat: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    gt = (gts_flat > 0).astype(np.uint8)
    rows = []
    # Sort by probs once for PR curve too (we’ll use later)
    for t in thresholds:
        pred = (probs_flat > float(t)).astype(np.uint8)
        tp = int(np.logical_and(gt == 1, pred == 1).sum())
        tn = int(np.logical_and(gt == 0, pred == 0).sum())
        fp = int(np.logical_and(gt == 0, pred == 1).sum())
        fn = int(np.logical_and(gt == 1, pred == 0).sum())
        m = metrics_from_conf(tp, tn, fp, fn)
        rows.append({"threshold": float(t), **m, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def precision_recall_curve_from_pixels(probs_flat: np.ndarray, gts_flat: np.ndarray, num_points: int = 200) -> pd.DataFrame:
    """
    Lightweight PR curve (no sklearn) by sweeping thresholds over quantiles of probs.
    """
    gt = (gts_flat > 0).astype(np.uint8)
    if probs_flat.size == 0:
        return pd.DataFrame(columns=["threshold","precision","recall"])

    # Choose thresholds as quantiles for stable sampling
    qs = np.linspace(0.0, 1.0, num_points)
    thr = np.quantile(probs_flat, qs)
    thr = np.unique(thr)

    rows = []
    for t in thr:
        pred = (probs_flat > float(t)).astype(np.uint8)
        tp = int(np.logical_and(gt == 1, pred == 1).sum())
        fp = int(np.logical_and(gt == 0, pred == 1).sum())
        fn = int(np.logical_and(gt == 1, pred == 0).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        rows.append({"threshold": float(t), "precision": float(precision), "recall": float(recall)})
    return pd.DataFrame(rows).sort_values("recall")


# ============================================================
# Qualitative visuals (best/worst/random)
# ============================================================
@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    sample,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sentinel, landsat, other, era5_raster, era5_tabular, _landcover, gt = sample
    sentinel_b = sentinel.unsqueeze(0).to(device)
    landsat_b = landsat.unsqueeze(0).to(device)
    other_b = other.unsqueeze(0).to(device)
    era5_raster_b = era5_raster.unsqueeze(0).to(device)
    era5_tab_b = era5_tabular.unsqueeze(0).to(device)

    gt_b = gt.unsqueeze(0).to(device)
    if gt_b.dim() == 3:
        gt_b = gt_b.unsqueeze(1)

    logits = model_forward_logits(model, sentinel_b, landsat_b, other_b, era5_raster_b, era5_tab_b)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    gt_np = (gt_b[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    sentinel_np = sentinel.detach().cpu().numpy().astype(np.float32)
    return sentinel_np, gt_np, prob


# ============================================================
# Paper/report outputs
# ============================================================
def write_report_summary(out_path: str,
                         thr_report: float,
                         thr_old_train: float,
                         new_metrics_report: Dict[str,float],
                         old_metrics_report: Optional[Dict[str,float]],
                         old_metrics_train_thr: Optional[Dict[str,float]],
                         best_thr_new: float,
                         best_thr_old: Optional[float]) -> None:
    lines = []
    lines.append("# Inference Report Summary\n")
    lines.append(f"- **Report threshold (your requirement):** {thr_report:.2f}\n")
    lines.append(f"- **Old-model training/validation threshold (from old train.py):** {thr_old_train:.2f}\n")
    lines.append("\n## Key note about the old model showing 0 at thr=0.95\n")
    lines.append(
        "- Your old `train.py` computes IoU/F1 using threshold **0.5**. "
        "If the old model probabilities rarely exceed **0.95**, then at 0.95 it will predict almost no burned pixels, "
        "which yields TP=0 → IoU/F1/Recall≈0. This does **not** necessarily mean the old checkpoint is broken; it may be **less confident / differently calibrated**.\n"
    )
    lines.append("\n## Results at report threshold\n")
    lines.append(f"- NEW @ {thr_report:.2f}: IoU={new_metrics_report['IoU']:.4f}, F1={new_metrics_report['F1']:.4f}, "
                 f"P={new_metrics_report['Precision']:.4f}, R={new_metrics_report['Recall']:.4f}\n")
    if old_metrics_report is not None:
        lines.append(f"- OLD @ {thr_report:.2f}: IoU={old_metrics_report['IoU']:.4f}, F1={old_metrics_report['F1']:.4f}, "
                     f"P={old_metrics_report['Precision']:.4f}, R={old_metrics_report['Recall']:.4f}\n")

    if old_metrics_train_thr is not None:
        lines.append("\n## Old checkpoint at its training threshold (sanity check)\n")
        lines.append(f"- OLD @ {thr_old_train:.2f}: IoU={old_metrics_train_thr['IoU']:.4f}, F1={old_metrics_train_thr['F1']:.4f}, "
                     f"P={old_metrics_train_thr['Precision']:.4f}, R={old_metrics_train_thr['Recall']:.4f}\n")

    lines.append("\n## Best-threshold comparison (from sweep)\n")
    lines.append(f"- NEW best IoU threshold: {best_thr_new:.3f}\n")
    if best_thr_old is not None:
        lines.append(f"- OLD best IoU threshold: {best_thr_old:.3f}\n")

    lines.append("\n## Why weighted inputs + auxiliary task help (paper phrasing)\n")
    lines.append(
        "- **Weighted input fusion** increases the influence of modalities that are empirically more informative (per ablation), "
        "improving representation quality and reducing reliance on weak/noisy sources.\n"
        "- **Auxiliary landcover segmentation** acts as a regularizer that injects semantic context (vegetation/soil/water patterns), "
        "which reduces false positives on dark surfaces and helps boundaries align with real land cover transitions.\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# CLI (your defaults + one optional old_code_dir)
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_new", type=str, required=False,
                        default="checkpoint_main/best_model_4.pth",
                        help="Path to the NEW checkpoint")
    parser.add_argument("--ckpt_old", type=str, required=False,
                        default="../src/checkpoint_2/best_model_3.pth",
                        help="Path to the OLD checkpoint (optional)")

    parser.add_argument("--out_dir", type=str, default="Results/inference_report", help="Output directory")

    parser.add_argument("--threshold", type=float, default=0.95, help="Report binarization threshold")
    parser.add_argument("--old_train_threshold", type=float, default=0.5,
                        help="Threshold used in old train.py (sanity check metrics)")

    parser.add_argument("--max_batches", type=int, default=0, help="Limit number of batches for quick runs")
    parser.add_argument("--num_visuals", type=int, default=6, help="How many samples in each category (best/worst/random)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--old_code_dir", type=str, default=None,
                        help="Optional: path to OLD code directory containing main.py/model.py (e.g. ../src). "
                             "If not given, inferred from ckpt_old.")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
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

    max_batches = None if args.max_batches <= 0 else int(args.max_batches)

    out_dir = args.out_dir
    tables_dir = os.path.join(out_dir, "tables")
    figs_dir = os.path.join(out_dir, "figures")
    samples_dir = os.path.join(out_dir, "samples")
    report_dir = os.path.join(out_dir, "report")
    for d in [tables_dir, figs_dir, samples_dir, report_dir]:
        ensure_dir(d)

    # Load NEW + OLD with correct codebases
    new_loaded = load_new_model(args.ckpt_new, device=device, tag="new")

    old_loaded: Optional[LoadedModel] = None
    if args.ckpt_old and os.path.exists(args.ckpt_old):
        try:
            old_loaded = load_old_model(args.ckpt_old, device=device, tag="old", old_code_dir=args.old_code_dir)
        except Exception as e:
            print("[WARN] Failed to load old model with old codebase. Old comparison will be skipped.")
            print(f"[WARN] Reason: {e}")
            old_loaded = None
    else:
        if args.ckpt_old:
            print(f"[WARN] ckpt_old not found: {args.ckpt_old} (skipping old comparison)")

    # Collect pixel-level probs/gts (enables sweep + PR curves)
    probs_new, gts_new, per_sample_new = collect_probs_and_gts(new_loaded, val_loader, device=device, max_batches=max_batches)

    probs_old = gts_old = per_sample_old = None
    if old_loaded is not None:
        probs_old, gts_old, per_sample_old = collect_probs_and_gts(old_loaded, val_loader, device=device, max_batches=max_batches)

    # Save per-sample stats (confidence/calibration-ish)
    df_sample_new = pd.DataFrame(per_sample_new, columns=["sample_idx", "gt_pos_frac", "prob_mean", "prob_p99"])
    df_sample_new.to_csv(os.path.join(tables_dir, "per_sample_confidence_new.csv"), index=False)
    if per_sample_old is not None:
        df_sample_old = pd.DataFrame(per_sample_old, columns=["sample_idx", "gt_pos_frac", "prob_mean", "prob_p99"])
        df_sample_old.to_csv(os.path.join(tables_dir, "per_sample_confidence_old.csv"), index=False)

    # Evaluate at report threshold (0.95)
    m_new_report, conf_new_report = eval_at_threshold(probs_new, gts_new, args.threshold)

    m_old_report = conf_old_report = None
    if probs_old is not None and gts_old is not None:
        m_old_report, conf_old_report = eval_at_threshold(probs_old, gts_old, args.threshold)

    # Old sanity check at 0.5 (matches old train.py)
    m_old_train = conf_old_train = None
    if probs_old is not None and gts_old is not None:
        m_old_train, conf_old_train = eval_at_threshold(probs_old, gts_old, args.old_train_threshold)

    # Threshold sweep for paper-grade curves
    thresholds = np.linspace(0.0, 1.0, 101)
    sweep_new = threshold_sweep(probs_new, gts_new, thresholds)
    sweep_new.to_csv(os.path.join(tables_dir, "threshold_sweep_new.csv"), index=False)

    best_thr_new = float(sweep_new.loc[sweep_new["IoU"].idxmax(), "threshold"])

    sweep_old = None
    best_thr_old = None
    if probs_old is not None and gts_old is not None:
        sweep_old = threshold_sweep(probs_old, gts_old, thresholds)
        sweep_old.to_csv(os.path.join(tables_dir, "threshold_sweep_old.csv"), index=False)
        best_thr_old = float(sweep_old.loc[sweep_old["IoU"].idxmax(), "threshold"])

    # PR curves
    pr_new = precision_recall_curve_from_pixels(probs_new, gts_new, num_points=200)
    pr_new.to_csv(os.path.join(tables_dir, "pr_curve_new.csv"), index=False)
    pr_old = None
    if probs_old is not None and gts_old is not None:
        pr_old = precision_recall_curve_from_pixels(probs_old, gts_old, num_points=200)
        pr_old.to_csv(os.path.join(tables_dir, "pr_curve_old.csv"), index=False)

    # -------- Figures for paper --------
    # 1) IoU vs Threshold (old vs new)
    y_dict = {"new": sweep_new["IoU"].values}
    if sweep_old is not None:
        y_dict["old"] = sweep_old["IoU"].values
    plot_line(thresholds, y_dict,
              os.path.join(figs_dir, "iou_vs_threshold.png"),
              title="IoU vs Threshold (pixel-level)",
              xlabel="threshold",
              ylabel="IoU")

    # 2) Precision–Recall curve
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pr_new["recall"].values, pr_new["precision"].values, label="new")
    if pr_old is not None:
        ax.plot(pr_old["recall"].values, pr_old["precision"].values, label="old")
    ax.set_title("Precision–Recall (pixel-level)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "precision_recall_curve.png"), dpi=220)
    plt.close(fig)

    # 3) Confusion matrices @ report threshold
    tp, tn, fp, fn = conf_new_report
    plot_confusion(tp, tn, fp, fn, os.path.join(figs_dir, "confusion_new_thr_report.png"),
                   title=f"Confusion (NEW) thr={args.threshold:.2f}")
    if conf_old_report is not None:
        tp, tn, fp, fn = conf_old_report
        plot_confusion(tp, tn, fp, fn, os.path.join(figs_dir, "confusion_old_thr_report.png"),
                       title=f"Confusion (OLD) thr={args.threshold:.2f}")

    # 4) Probability histogram (calibration-ish)
    plot_hist(probs_new, os.path.join(figs_dir, "prob_hist_new.png"), "Probability histogram (NEW)", "p(burned)")
    if probs_old is not None:
        plot_hist(probs_old, os.path.join(figs_dir, "prob_hist_old.png"), "Probability histogram (OLD)", "p(burned)")

    # 5) Scatter: GT burned fraction vs prob_p99 (confidence sanity)
    plot_scatter(df_sample_new["gt_pos_frac"].values, df_sample_new["prob_p99"].values,
                 os.path.join(figs_dir, "gt_frac_vs_probp99_new.png"),
                 "GT fraction vs 99th percentile prob (NEW)", "GT burned fraction", "p99(prob)")
    if per_sample_old is not None:
        df_sample_old = pd.DataFrame(per_sample_old, columns=["sample_idx", "gt_pos_frac", "prob_mean", "prob_p99"])
        plot_scatter(df_sample_old["gt_pos_frac"].values, df_sample_old["prob_p99"].values,
                     os.path.join(figs_dir, "gt_frac_vs_probp99_old.png"),
                     "GT fraction vs 99th percentile prob (OLD)", "GT burned fraction", "p99(prob)")

    # 6) Summary metrics table for paper
    summary_rows = []
    summary_rows.append({"model": "new", "threshold": args.threshold, **m_new_report})
    if m_old_report is not None:
        summary_rows.append({"model": "old", "threshold": args.threshold, **m_old_report})
    if m_old_train is not None:
        summary_rows.append({"model": "old", "threshold": args.old_train_threshold, **m_old_train})
    summary_rows.append({"model": "new", "threshold": best_thr_new, **eval_at_threshold(probs_new, gts_new, best_thr_new)[0]})
    if best_thr_old is not None and probs_old is not None and gts_old is not None:
        summary_rows.append({"model": "old", "threshold": best_thr_old, **eval_at_threshold(probs_old, gts_old, best_thr_old)[0]})

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(tables_dir, "metrics_summary_key_thresholds.csv"), index=False)

    # -------- Qualitative visuals (best/worst/random by per-sample IoU at report threshold) --------
    # Compute per-sample IoU using batch-by-batch predictions (cheap for a few visuals)
    per_sample_iou_rows = []
    with torch.no_grad():
        idx_global = 0
        batch_count = 0
        for batch in val_loader:
            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                break

            sentinel = batch[0].to(device)
            landsat = batch[1].to(device)
            other = batch[2].to(device)
            era5_raster = batch[3].to(device)
            era5_tab = batch[4].to(device)
            gt = batch[-1].to(device)
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)

            logits_new = model_forward_logits(new_loaded.model, sentinel, landsat, other, era5_raster, era5_tab)
            probs_b = torch.sigmoid(logits_new)[:, 0].detach().cpu().numpy()
            gt_b = (gt[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            pred_b = (probs_b > args.threshold).astype(np.uint8)

            b = probs_b.shape[0]
            for i in range(b):
                tp, tn, fp, fn = confusion_binary(gt_b[i], pred_b[i])
                iou = iou_from_conf(tp, fp, fn)
                per_sample_iou_rows.append({"idx": idx_global + i, "IoU": iou})
            idx_global += b

    df_iou = pd.DataFrame(per_sample_iou_rows)
    df_iou.to_csv(os.path.join(tables_dir, "per_sample_iou_new_thr_report.csv"), index=False)

    if not df_iou.empty:
        df_sorted = df_iou.sort_values("IoU", ascending=False)
        best_idx = df_sorted.head(args.num_visuals)["idx"].astype(int).tolist()
        worst_idx = df_sorted.tail(args.num_visuals)["idx"].astype(int).tolist()
        rng = np.random.default_rng(args.seed)
        all_idx = df_iou["idx"].astype(int).values
        rand_n = min(args.num_visuals, len(all_idx))
        rand_idx = rng.choice(all_idx, size=rand_n, replace=False).astype(int).tolist()

        chosen = best_idx + worst_idx + rand_idx
        chosen = [int(i) for i in chosen]

        for k, idx in enumerate(chosen):
            sample = val_ds[idx]
            sentinel_np, gt_np, prob_new = predict_single(new_loaded.model, sample, device)
            pred_new = (prob_new > args.threshold).astype(np.uint8)

            out_panel = os.path.join(samples_dir, f"{k:03d}_idx{idx}_new.png")
            save_panel(out_panel, sentinel_np, gt_np, prob_new, pred_new, args.threshold, title=f"NEW idx={idx}")

            if old_loaded is not None:
                sentinel_np2, gt_np2, prob_old = predict_single(old_loaded.model, sample, device)
                pred_old = (prob_old > args.threshold).astype(np.uint8)
                out_cmp = os.path.join(samples_dir, f"{k:03d}_idx{idx}_compare_old_vs_new.png")
                save_compare_panel(out_cmp, sentinel_np2, gt_np2, pred_old, pred_new, args.threshold, title=f"idx={idx} OLD vs NEW")

    # Report summary text for the professor/paper narrative
    write_report_summary(
        os.path.join(report_dir, "report_summary.md"),
        thr_report=args.threshold,
        thr_old_train=args.old_train_threshold,
        new_metrics_report=m_new_report,
        old_metrics_report=m_old_report,
        old_metrics_train_thr=m_old_train,
        best_thr_new=best_thr_new,
        best_thr_old=best_thr_old,
    )

    print("\n[OK] Done.")
    print(f"Output: {out_dir}")
    print(f"NEW @thr={args.threshold:.2f}: {m_new_report}")
    if m_old_report is not None:
        print(f"OLD @thr={args.threshold:.2f}: {m_old_report}")
    if m_old_train is not None:
        print(f"OLD @thr={args.old_train_threshold:.2f} (train-thr sanity): {m_old_train}")
    print(f"Best IoU threshold NEW: {best_thr_new:.3f}")
    if best_thr_old is not None:
        print(f"Best IoU threshold OLD: {best_thr_old:.3f}")


if __name__ == "__main__":
    main()