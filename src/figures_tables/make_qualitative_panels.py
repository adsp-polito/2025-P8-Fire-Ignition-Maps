import os
import numpy as np
import matplotlib.pyplot as plt

from main import set_seed, build_datasets_and_loaders

# ========== CONFIG ==========
SEED = 42
OUT_DIR = "slide_figures/qualitative"
NPY_DIR = "deploy_outputs"          # where prob_*.npy and mask_*.npy are
SAMPLE_IDXS = [0, 1, 2, 3, 4]       # must match the ones you saved in deploy_inference
THRESHOLD = 0.95                   # threshold used to create mask_*.npy

# For a clean RGB preview from Sentinel-2 (common choice):
# Sentinel bands order in many pipelines: [B1..B12], but confirm your dataset.
# Typical RGB = B4 (red), B3 (green), B2 (blue). Indices depend on your band ordering.
# If your Sentinel tensor is [12,H,W] in standard S2 order: B2=1, B3=2, B4=3 (0-based)
RGB_IDX = (3, 2, 1)  # (R,G,B) indices in your sentinel tensor; change if needed.

# ========== HELPERS ==========
def to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    """Normalize [H,W,3] float/any to uint8 for display."""
    x = x.astype(np.float32)
    # robust scaling using percentiles
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    return (255 * x).astype(np.uint8)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ========== MAIN ==========
def main():
    ensure_dir(OUT_DIR)
    set_seed(SEED)

    # Build loaders to access the SAME validation split
    _, _, _, val_loader = build_datasets_and_loaders()
    val_dataset = val_loader.dataset

    for idx in SAMPLE_IDXS:
        # Load npy outputs
        prob_path = os.path.join(NPY_DIR, f"prob_{idx}.npy")
        mask_path = os.path.join(NPY_DIR, f"mask_{idx}.npy")
        if not os.path.exists(prob_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing {prob_path} or {mask_path}")

        prob = np.load(prob_path)  # [H,W] probability
        pred = np.load(mask_path)  # [H,W] binary mask 0/1

        # Get the corresponding dataset sample to fetch Sentinel + GT
        sample = val_dataset[idx]
        image_sentinel = sample[0].numpy()  # [12,H,W]
        gt_mask = sample[-1].numpy()        # [1,H,W] or [H,W]

        if gt_mask.ndim == 3:
            gt = gt_mask[0]
        else:
            gt = gt_mask

        # RGB preview
        rgb = np.stack([image_sentinel[RGB_IDX[0]],
                        image_sentinel[RGB_IDX[1]],
                        image_sentinel[RGB_IDX[2]]], axis=-1)
        rgb_u8 = to_uint8_rgb(rgb)

        # Error map: FP=red, FN=blue, TP=green
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)
        tp = (pred == 1) & (gt == 1)
        err = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        err[fp] = [255, 0, 0]
        err[fn] = [0, 0, 255]
        err[tp] = [0, 255, 0]

        # Plot panel
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_u8)
        ax1.set_title("Sentinel-2 RGB (for context)")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(prob, vmin=0, vmax=1)
        ax2.set_title("Predicted probability map")
        ax2.axis("off")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("P(burned)")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(gt, vmin=0, vmax=1)
        ax3.set_title("Ground truth mask")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(err)
        ax4.set_title(f"Error map @ thr={THRESHOLD} (TP=green, FP=red, FN=blue)")
        ax4.axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"panel_{idx}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()