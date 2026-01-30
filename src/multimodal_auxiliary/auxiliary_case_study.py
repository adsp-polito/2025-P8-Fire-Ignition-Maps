#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sentinel_to_rgb(sentinel_chw: np.ndarray) -> np.ndarray:
    """Pseudo-RGB from first 3 channels with percentile stretch."""
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


def torch_load_any(path: str, device: torch.device) -> dict:
    """Compatible torch.load across versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(obj) -> dict:
    """Supports raw state_dict or {'state_dict': ...}, and strips common prefixes."""
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


# ============================================================
# Label handling (FIXED)
# ============================================================
def landcover_to_labelmap(lc: np.ndarray) -> np.ndarray:
    """
    Handles landcover stored as:
      - (H, W) integer labels
      - (1, H, W) single-channel label raster  ✅ your case
      - (C, H, W) one-hot/logits
    Returns (H, W) integer labels.
    """
    if lc.ndim == 2:
        return lc.astype(np.int64)
    if lc.ndim == 3:
        if lc.shape[0] == 1:
            return lc[0].astype(np.int64)  # ✅ critical fix
        return np.argmax(lc, axis=0).astype(np.int64)
    raise ValueError(f"Unsupported landcover shape: {lc.shape}")


def dominant_landcover_class(lc: np.ndarray, ignore_index: Optional[int] = None) -> int:
    labels = landcover_to_labelmap(lc)
    flat = labels.reshape(-1)
    if ignore_index is not None:
        flat = flat[flat != ignore_index]
    if flat.size == 0:
        return -1
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(counts)])


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
    return {"IoU": float(iou), "Precision": float(precision), "Recall": float(recall), "F1": float(f1)}


# ============================================================
# Model loading + forward
# ============================================================
def forward_required_positional_args(model: torch.nn.Module) -> int:
    import inspect

    sig = inspect.signature(model.forward)
    n = 0
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    is_multimodal: bool


def load_model_from_current_code(ckpt_path: str, device: torch.device, name: str) -> LoadedModel:
    # Build using current codebase (main.build_model)
    from main import build_model

    model = build_model().to(device)

    sd = unwrap_state_dict(torch_load_any(ckpt_path, device))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Back-compat: modality_weights may be missing in older ckpts
    if missing and any("modality_weights" in k for k in missing):
        try:
            from main import MODALITY_WEIGHTS

            if hasattr(model, "modality_weights"):
                w = torch.tensor(MODALITY_WEIGHTS, device=device, dtype=torch.float32)
                mw = getattr(model, "modality_weights")
                if isinstance(mw, torch.nn.Parameter):
                    mw.data.copy_(w)
                else:
                    mw.copy_(w)
        except Exception:
            pass

    if missing:
        print(f"[WARN] {name}: missing keys (strict=False) sample: {missing[:8]}")
    if unexpected:
        print(f"[WARN] {name}: unexpected keys (strict=False) sample: {unexpected[:8]}")

    model.eval()
    req = forward_required_positional_args(model)
    is_multimodal = req >= 6
    return LoadedModel(name=name, model=model, is_multimodal=is_multimodal)


@torch.no_grad()
def forward_logits(
    loaded: LoadedModel,
    sentinel: torch.Tensor,
    landsat: torch.Tensor,
    other: torch.Tensor,
    era5_r: torch.Tensor,
    era5_t: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (burned_logits, landcover_logits_or_None)
    Supports:
      - multimodal: model(sentinel, landsat, other, ignition, era5_r, era5_t)
      - sentinel-only: model(sentinel)
    Output formats:
      - burned logits only
      - (burned_logits, landcover_logits, ...)
    """
    if loaded.is_multimodal:
        b, _, h, w = sentinel.shape
        ignition = torch.zeros((b, 1, h, w), device=sentinel.device, dtype=sentinel.dtype)
        out = loaded.model(sentinel, landsat, other, ignition, era5_r, era5_t)
    else:
        out = loaded.model(sentinel)

    if isinstance(out, (tuple, list)):
        burned = out[0]
        land = out[1] if len(out) > 1 else None
        return burned, land
    return out, None


# ============================================================
# Plots
# ============================================================
def plot_delta_hist(df: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["delta_iou"].values, bins=30)
    ax.set_title("ΔIoU = IoU_new − IoU_old (thr=0.95)")
    ax.set_xlabel("ΔIoU")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta_vs_burnfrac(df: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df["gt_burn_frac"].values, df["delta_iou"].values, s=10)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("ΔIoU vs GT burned fraction")
    ax.set_xlabel("GT burned fraction")
    ax.set_ylabel("ΔIoU (new-old)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mean_delta_by_landcover(df: pd.DataFrame, out_path: str) -> None:
    grp = df.groupby("landcover_dom")["delta_iou"].agg(["mean", "count", "std"]).sort_values("mean", ascending=False)
    # Only show classes that actually appear
    x = grp.index.astype(int).tolist()
    mean = grp["mean"].values
    count = grp["count"].values
    std = grp["std"].fillna(0.0).values

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([str(i) for i in x], mean)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Mean ΔIoU by dominant landcover class")
    ax.set_xlabel("dominant landcover class")
    ax.set_ylabel("mean ΔIoU")

    # annotate counts
    for i, (m, n) in enumerate(zip(mean, count)):
        ax.text(i, m, f"n={int(n)}", ha="center", va="bottom" if m >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_winrate_by_landcover(df: pd.DataFrame, out_path: str) -> None:
    grp = df.groupby("landcover_dom")["delta_iou"].apply(lambda s: float((s > 0).mean()))
    counts = df.groupby("landcover_dom")["delta_iou"].count()
    grp = grp.sort_values(ascending=False)

    x = grp.index.astype(int).tolist()
    y = grp.values

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([str(i) for i in x], y)
    ax.set_title("Win-rate by dominant landcover class (ΔIoU>0)")
    ax.set_xlabel("dominant landcover class")
    ax.set_ylabel("win rate")

    for i, cls in enumerate(x):
        ax.text(i, y[i], f"n={int(counts.loc[cls])}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta_by_burnfrac_bins(df: pd.DataFrame, out_path: str, bins: List[float]) -> None:
    bins = sorted(bins)
    labels = []
    means = []
    counts = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sub = df[(df["gt_burn_frac"] >= lo) & (df["gt_burn_frac"] < hi)]
        labels.append(f"{lo:.2f}-{hi:.2f}")
        means.append(float(sub["delta_iou"].mean()) if len(sub) else 0.0)
        counts.append(int(len(sub)))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, means)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Mean ΔIoU by GT burned-fraction bins")
    ax.set_xlabel("GT burned fraction bin")
    ax.set_ylabel("mean ΔIoU")

    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, m, f"n={n}", ha="center", va="bottom" if m >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Panels
# ============================================================
def save_compare_panel(
    out_path: str,
    sentinel_chw: np.ndarray,
    gt_hw: np.ndarray,
    pred_old_hw: np.ndarray,
    pred_new_hw: np.ndarray,
    landcover_aux_hw: Optional[np.ndarray],
    thr: float,
    title: str,
) -> None:
    rgb = sentinel_to_rgb(sentinel_chw)
    gt = (gt_hw > 0).astype(np.uint8)
    po = (pred_old_hw > 0).astype(np.uint8)
    pn = (pred_new_hw > 0).astype(np.uint8)

    def overlay(pred: np.ndarray) -> np.ndarray:
        ov = rgb.copy()
        ov = np.clip(
            ov
            + 0.35 * np.dstack([pred, np.zeros_like(pred), np.zeros_like(pred)]).astype(np.float32)
            + 0.25 * np.dstack([np.zeros_like(gt), gt, np.zeros_like(gt)]).astype(np.float32),
            0,
            1,
        )
        return ov

    cols = 4 if landcover_aux_hw is not None else 3
    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(f"{title} | thr={thr:.2f}", fontsize=12)

    ax1 = fig.add_subplot(1, cols, 1)
    ax1.imshow(overlay(po))
    ax1.set_title("OLD overlay")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, cols, 2)
    ax2.imshow(overlay(pn))
    ax2.set_title("NEW overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, cols, 3)
    ax3.imshow(gt, vmin=0, vmax=1)
    ax3.set_title("GT mask")
    ax3.axis("off")

    if landcover_aux_hw is not None:
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(landcover_aux_hw)
        ax4.set_title("Landcover (aux pred)")
        ax4.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_new", type=str, default="checkpoint_main/best_model_4.pth")
    p.add_argument("--ckpt_old", type=str, default="../src/checkpoint_2/best_model_3.pth")
    p.add_argument("--out_dir", type=str, default="Results/aux_case_study")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--max_batches", type=int, default=0, help="0=full val, else limit")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore_landcover", type=int, default=-9999, help="set to actual ignore index; -9999 disables")
    p.add_argument(
        "--burn_bins",
        type=str,
        default="0,0.01,0.05,0.10,0.20,0.40,1.01",
        help="Comma-separated GT burn fraction bin edges",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ignore_index = None if args.ignore_landcover == -9999 else int(args.ignore_landcover)
    burn_bins = [float(x) for x in args.burn_bins.split(",") if x.strip() != ""]

    # device
    try:
        from main import DEVICE

        device = DEVICE
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    from main import build_datasets_and_loaders

    _train_ds, val_ds, _train_loader, val_loader = build_datasets_and_loaders()

    # output dirs
    out_dir = args.out_dir
    tables_dir = os.path.join(out_dir, "tables")
    figs_dir = os.path.join(out_dir, "figures")
    samples_dir = os.path.join(out_dir, "samples")
    report_dir = os.path.join(out_dir, "report")
    for d in [tables_dir, figs_dir, samples_dir, report_dir]:
        ensure_dir(d)
    ensure_dir(os.path.join(samples_dir, "top_improvements"))
    ensure_dir(os.path.join(samples_dir, "top_regressions"))

    max_batches = None if args.max_batches == 0 else int(args.max_batches)

    # models
    new_m = load_model_from_current_code(args.ckpt_new, device, "new")
    old_m = load_model_from_current_code(args.ckpt_old, device, "old")

    rows: List[Dict] = []

    batch_count = 0
    global_idx = 0
    saw_landcover_logits = False

    # ---------- pass 1: compute per-sample deltas ----------
    for batch in val_loader:
        batch_count += 1
        if max_batches is not None and batch_count > max_batches:
            break

        sentinel = batch[0].to(device)
        landsat = batch[1].to(device)
        other = batch[2].to(device)
        era5_r = batch[3].to(device)
        era5_t = batch[4].to(device)

        landcover = batch[5].detach().cpu().numpy()  # (B,1,H,W) or (B,H,W) etc
        gt = batch[-1].to(device)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        logits_new, landlog_new = forward_logits(new_m, sentinel, landsat, other, era5_r, era5_t)
        logits_old, _ = forward_logits(old_m, sentinel, landsat, other, era5_r, era5_t)

        if landlog_new is not None:
            saw_landcover_logits = True

        prob_new = torch.sigmoid(logits_new)[:, 0].detach().cpu().numpy()
        prob_old = torch.sigmoid(logits_old)[:, 0].detach().cpu().numpy()
        gt_np = (gt[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        pred_new = (prob_new > args.threshold).astype(np.uint8)
        pred_old = (prob_old > args.threshold).astype(np.uint8)

        b = gt_np.shape[0]
        for i in range(b):
            idx = global_idx + i

            tp_n, tn_n, fp_n, fn_n = confusion_binary(gt_np[i], pred_new[i])
            tp_o, tn_o, fp_o, fn_o = confusion_binary(gt_np[i], pred_old[i])

            m_n = metrics_from_conf(tp_n, tn_n, fp_n, fn_n)
            m_o = metrics_from_conf(tp_o, tn_o, fp_o, fn_o)

            gt_frac = float(gt_np[i].mean())

            lc_i = landcover[i]
            lc_dom = dominant_landcover_class(lc_i, ignore_index=ignore_index)

            rows.append(
                {
                    "idx": idx,
                    "gt_burn_frac": gt_frac,
                    "landcover_dom": lc_dom,
                    "iou_new": m_n["IoU"],
                    "iou_old": m_o["IoU"],
                    "delta_iou": m_n["IoU"] - m_o["IoU"],
                    "f1_new": m_n["F1"],
                    "f1_old": m_o["F1"],
                }
            )

        global_idx += b

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, "per_sample_delta.csv"), index=False)

    # ---------- summary ----------
    frac_win = float((df["delta_iou"] > 0).mean()) if not df.empty else 0.0
    mean_delta = float(df["delta_iou"].mean()) if not df.empty else 0.0
    med_delta = float(df["delta_iou"].median()) if not df.empty else 0.0

    # ---------- plots ----------
    if not df.empty:
        plot_delta_hist(df, os.path.join(figs_dir, "delta_iou_hist.png"))
        plot_delta_vs_burnfrac(df, os.path.join(figs_dir, "delta_iou_vs_burnfrac.png"))
        plot_mean_delta_by_landcover(df, os.path.join(figs_dir, "delta_iou_by_landcover.png"))
        plot_winrate_by_landcover(df, os.path.join(figs_dir, "winrate_by_landcover.png"))
        plot_delta_by_burnfrac_bins(df, os.path.join(figs_dir, "delta_iou_by_burnfrac_bins.png"), burn_bins)

        # also save a small table aggregated by landcover for report
        agg = (
            df.groupby("landcover_dom")
            .agg(
                n=("delta_iou", "count"),
                mean_delta=("delta_iou", "mean"),
                median_delta=("delta_iou", "median"),
                win_rate=("delta_iou", lambda s: float((s > 0).mean())),
            )
            .reset_index()
            .sort_values("mean_delta", ascending=False)
        )
        agg.to_csv(os.path.join(tables_dir, "delta_by_landcover_summary.csv"), index=False)

    # ---------- qualitative: top improvements / regressions ----------
    if not df.empty:
        topk = int(args.topk)
        best = df.sort_values("delta_iou", ascending=False).head(topk)["idx"].astype(int).tolist()
        worst = df.sort_values("delta_iou", ascending=True).head(topk)["idx"].astype(int).tolist()

        for kind, indices in [("top_improvements", best), ("top_regressions", worst)]:
            for rank, idx in enumerate(indices):
                sample = val_ds[int(idx)]
                sentinel, landsat, other, era5_r, era5_t, lc_gt, gt = sample

                # single forward
                with torch.no_grad():
                    s = sentinel.unsqueeze(0).to(device)
                    l = landsat.unsqueeze(0).to(device)
                    o = other.unsqueeze(0).to(device)
                    er = era5_r.unsqueeze(0).to(device)
                    et = era5_t.unsqueeze(0).to(device)

                    gt_np = (gt.detach().cpu().numpy() > 0.5).astype(np.uint8)

                    ln_new, land_new = forward_logits(new_m, s, l, o, er, et)
                    ln_old, _ = forward_logits(old_m, s, l, o, er, et)

                    prob_new = torch.sigmoid(ln_new)[0, 0].detach().cpu().numpy()
                    prob_old = torch.sigmoid(ln_old)[0, 0].detach().cpu().numpy()

                    pred_new = (prob_new > args.threshold).astype(np.uint8)
                    pred_old = (prob_old > args.threshold).astype(np.uint8)

                # Landcover for panel: prefer AUX prediction if available
                land_to_plot = None
                if land_new is not None:
                    land_to_plot = np.argmax(land_new[0].detach().cpu().numpy(), axis=0).astype(np.int64)
                else:
                    # fallback: show GT landcover (still useful), with correct label handling
                    land_to_plot = landcover_to_labelmap(lc_gt.detach().cpu().numpy())

                dval = float(df.loc[df["idx"] == idx, "delta_iou"].values[0])
                out_path = os.path.join(samples_dir, kind, f"{rank:02d}_idx{idx}.png")
                save_compare_panel(
                    out_path=out_path,
                    sentinel_chw=sentinel.detach().cpu().numpy(),
                    gt_hw=gt_np,
                    pred_old_hw=pred_old,
                    pred_new_hw=pred_new,
                    landcover_aux_hw=land_to_plot,
                    thr=args.threshold,
                    title=f"{kind} idx={idx} ΔIoU={dval:+.3f}",
                )

    # ---------- report ----------
    with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Auxiliary Task Case Study (No-Retrain)\n\n")
        f.write(f"- threshold: **{args.threshold:.2f}**\n")
        f.write(f"- max_batches: **{args.max_batches}** (0 = full set)\n\n")
        f.write("## Key results\n")
        f.write(f"- Mean ΔIoU (new-old): **{mean_delta:+.4f}**\n")
        f.write(f"- Median ΔIoU (new-old): **{med_delta:+.4f}**\n")
        f.write(f"- Fraction of samples where NEW > OLD (ΔIoU>0): **{frac_win*100:.1f}%**\n\n")
        f.write("## Notes\n")
        if saw_landcover_logits:
            f.write("- Landcover(aux) panels show **NEW model’s auxiliary landcover prediction**.\n")
        else:
            f.write("- NEW model did not return auxiliary landcover logits in this run; panels show GT landcover instead.\n")
        f.write("- Use `delta_iou_by_landcover.png` and `winrate_by_landcover.png` to discuss where auxiliary context helps.\n")
        f.write("- Use the top_improvements/top_regressions panels as qualitative evidence.\n")

    print("\n[OK] Wrote updated aux case study pack to:", out_dir)
    print(f"Mean ΔIoU (new-old): {mean_delta:+.4f}")
    print(f"Median ΔIoU (new-old): {med_delta:+.4f}")
    print(f"Win-rate (ΔIoU>0): {frac_win*100:.1f}%")
    if not saw_landcover_logits:
        print("[WARN] NEW model did not output landcover logits; check your forward() outputs if you expected them.")


if __name__ == "__main__":
    main()