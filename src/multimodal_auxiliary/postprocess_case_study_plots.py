#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_bins(s: str) -> List[float]:
    xs = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    xs = sorted(xs)
    if len(xs) < 2:
        raise ValueError("Need at least 2 bin edges.")
    return xs


def assign_bins(x: np.ndarray, edges: List[float]) -> Tuple[np.ndarray, List[str]]:
    # returns bin index in [0..len(edges)-2], -1 for out of range
    idx = np.digitize(x, edges, right=False) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return idx, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="Results/aux_case_study/tables/per_sample_delta.csv")
    ap.add_argument("--out_dir", type=str, default="Results/postprocess")
    ap.add_argument("--burn_bins", type=str, default="0,0.01,0.05,0.10,0.20,0.40,1.01")
    ap.add_argument("--thr_label", type=str, default="0.95", help="Only used in titles")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    figs = os.path.join(args.out_dir, "figures")
    tables = os.path.join(args.out_dir, "tables")
    ensure_dir(figs)
    ensure_dir(tables)

    df = pd.read_csv(args.csv)

    # sanity checks
    needed = {"gt_burn_frac", "delta_iou", "iou_new", "iou_old"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # -------------------------
    # 1) ΔIoU histogram
    # -------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["delta_iou"].values, bins=30)
    ax.set_title(f"ΔIoU = IoU_new − IoU_old (thr={args.thr_label})")
    ax.set_xlabel("ΔIoU")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "delta_iou_hist.png"), dpi=220)
    plt.close(fig)

    # -------------------------
    # 2) IoU histogram NEW vs OLD (your request)
    # -------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["iou_old"].values, bins=30, alpha=0.6, label="old")
    ax.hist(df["iou_new"].values, bins=30, alpha=0.6, label="new")
    ax.set_title(f"IoU distribution (thr={args.thr_label})")
    ax.set_xlabel("IoU")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "iou_hist_new_vs_old.png"), dpi=220)
    plt.close(fig)

    # -------------------------
    # 3) Bin by GT burned fraction
    # -------------------------
    edges = parse_bins(args.burn_bins)
    bin_idx, bin_labels = assign_bins(df["gt_burn_frac"].values, edges)
    df["burn_bin"] = bin_idx

    # Filter valid bins
    dff = df[df["burn_bin"] >= 0].copy()

    # Summary table by bins
    rows = []
    for b in range(len(bin_labels)):
        sub = dff[dff["burn_bin"] == b]
        if len(sub) == 0:
            continue
        rows.append({
            "burn_bin": bin_labels[b],
            "n": int(len(sub)),
            "mean_delta_iou": float(sub["delta_iou"].mean()),
            "median_delta_iou": float(sub["delta_iou"].median()),
            "win_rate_delta_gt0": float((sub["delta_iou"] > 0).mean()),
            "mean_iou_old": float(sub["iou_old"].mean()),
            "mean_iou_new": float(sub["iou_new"].mean()),
        })
    out_table = pd.DataFrame(rows)
    out_table.to_csv(os.path.join(tables, "summary_table_by_bins.csv"), index=False)

    # -------------------------
    # 4) Mean ΔIoU by burn bins (bar w/ n)
    # -------------------------
    means = []
    ns = []
    labels = []
    for b, lab in enumerate(bin_labels):
        sub = dff[dff["burn_bin"] == b]
        if len(sub) == 0:
            continue
        labels.append(lab)
        means.append(float(sub["delta_iou"].mean()))
        ns.append(int(len(sub)))

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, means)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Mean ΔIoU by GT burned-fraction bins")
    ax.set_xlabel("GT burned fraction bin")
    ax.set_ylabel("mean ΔIoU")
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m, f"n={n}", ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "delta_iou_mean_by_burn_bins.png"), dpi=220)
    plt.close(fig)

    # -------------------------
    # 5) Box plot of ΔIoU by burn bins (this is what I meant)
    # -------------------------
    data = []
    lab2 = []
    for b, lab in enumerate(bin_labels):
        sub = dff[dff["burn_bin"] == b]["delta_iou"].values
        if len(sub) == 0:
            continue
        data.append(sub)
        lab2.append(lab)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(data, labels=lab2, showfliers=True)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("ΔIoU distribution by GT burned-fraction bins (box plot)")
    ax.set_xlabel("GT burned fraction bin")
    ax.set_ylabel("ΔIoU")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "delta_iou_boxplot_by_burn_bins.png"), dpi=220)
    plt.close(fig)

    # -------------------------
    # 6) Cumulative win-rate vs minimum burn fraction (very persuasive)
    # -------------------------
    # For each threshold t, consider samples with gt_burn_frac >= t
    ts = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    win_rates = []
    counts = []
    for t in ts:
        sub = df[df["gt_burn_frac"] >= t]
        counts.append(int(len(sub)))
        if len(sub) == 0:
            win_rates.append(np.nan)
        else:
            win_rates.append(float((sub["delta_iou"] > 0).mean()))

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, win_rates, marker="o")
    ax.set_title("Cumulative win-rate of NEW vs OLD vs min GT burned fraction")
    ax.set_xlabel("Min GT burned fraction threshold")
    ax.set_ylabel("Win-rate: P(ΔIoU>0)")
    for x, y, n in zip(ts, win_rates, counts):
        if np.isfinite(y):
            ax.text(x, y, f"n={n}", fontsize=8, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "cumulative_winrate_vs_min_burnfrac.png"), dpi=220)
    plt.close(fig)

    print("[OK] Wrote plots to:", figs)
    print("[OK] Wrote tables to:", tables)


if __name__ == "__main__":
    main()