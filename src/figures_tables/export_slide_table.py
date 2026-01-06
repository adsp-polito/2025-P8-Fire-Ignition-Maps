import os
import csv

SWEEP = [
    (0.95, 0.4026, 0.5761, 0.5722, 0.5741),
    (0.50, 0.2341, 0.2446, 0.8442, 0.3793),
]

OUT_DIR = "slide_figures/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    out = os.path.join(OUT_DIR, "metrics_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Setting", "Threshold", "IoU", "Precision", "Recall", "F1"])
        w.writerow(["Best threshold (val)", *SWEEP[0]])
        w.writerow(["Default threshold",    *SWEEP[1]])
    print("[saved]", out)

if __name__ == "__main__":
    main()