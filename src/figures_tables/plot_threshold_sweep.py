import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Paste your sweep results here, OR read from a file if you already saved it.
# Format: (thr, iou, precision, recall, f1)
SWEEP = [
    (0.95, 0.4026, 0.5761, 0.5722, 0.5741),
    (0.90, 0.3762, 0.4659, 0.6615, 0.5467),
    (0.85, 0.3447, 0.4017, 0.7086, 0.5127),
    (0.80, 0.3228, 0.3631, 0.7442, 0.4881),
    (0.75, 0.3030, 0.3330, 0.7705, 0.4650),
    (0.70, 0.2875, 0.3113, 0.7903, 0.4466),
    (0.65, 0.2746, 0.2938, 0.8074, 0.4308),
    (0.60, 0.2602, 0.2759, 0.8206, 0.4130),
    (0.55, 0.2466, 0.2595, 0.8331, 0.3957),
    (0.50, 0.2341, 0.2446, 0.8442, 0.3793),
    (0.45, 0.2211, 0.2298, 0.8541, 0.3622),
    (0.40, 0.2097, 0.2169, 0.8632, 0.3467),
    (0.35, 0.1980, 0.2039, 0.8719, 0.3305),
    (0.30, 0.1852, 0.1900, 0.8808, 0.3126),
    (0.25, 0.1725, 0.1763, 0.8895, 0.2943),
    (0.20, 0.1604, 0.1633, 0.8993, 0.2764),
    (0.15, 0.1475, 0.1496, 0.9110, 0.2571),
    (0.10, 0.1316, 0.1330, 0.9256, 0.2326),
    (0.05, 0.1106, 0.1114, 0.9403, 0.1992),
]

OUT_DIR = "slide_figures/threshold"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    thrs = np.array([x[0] for x in SWEEP])
    iou  = np.array([x[1] for x in SWEEP])
    p    = np.array([x[2] for x in SWEEP])
    r    = np.array([x[3] for x in SWEEP])
    f1   = np.array([x[4] for x in SWEEP])

    best_idx = int(np.argmax(iou))

    # Plot IoU vs threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thrs, iou, marker="o")
    plt.scatter([thrs[best_idx]], [iou[best_idx]], s=80)
    plt.xlabel("Threshold")
    plt.ylabel("IoU (burned area)")
    plt.title("IoU vs Threshold (Validation)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iou_vs_threshold.png"), dpi=220)
    plt.close()

    # Plot Precision-Recall curve (from threshold sweep points)
    plt.figure(figsize=(6, 6))
    plt.plot(r, p, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (Validation, threshold sweep)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=220)
    plt.close()

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "threshold_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "iou", "precision", "recall", "f1"])
        for row in SWEEP:
            w.writerow(row)

    print("[saved]", os.path.join(OUT_DIR, "iou_vs_threshold.png"))
    print("[saved]", os.path.join(OUT_DIR, "pr_curve.png"))
    print("[saved]", csv_path)
    print(f"Best IoU: {iou[best_idx]:.4f} at thr={thrs[best_idx]:.2f}")

if __name__ == "__main__":
    main()