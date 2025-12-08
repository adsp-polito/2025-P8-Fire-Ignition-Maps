import numpy as np
from dataset import PiedmontDataset

ROOT_DIR = "../data/"          # same as main.py
GEOJSON_PATH = "../geojson/piedmont_2012_2024_fa.geojson"  # same as main.py
TARGET_SIZE = (256, 256)


def main():
    ds = PiedmontDataset(
        root_dir=ROOT_DIR,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False,
        apply_augmentations=False,
        global_stats=None,        # stats not needed here
        initial_fire_dirs=None,
    )

    total_pos = 0
    total_pixels = 0

    for idx in range(len(ds)):
        sample = ds[idx]
        gt_mask_tensor = sample[-1]  # last element in tuple
        gt_np = gt_mask_tensor.numpy()
        if gt_np.ndim == 3:
            gt_np = gt_np[0]

        pos = np.sum(gt_np > 0.5)
        pixels = gt_np.size

        total_pos += pos
        total_pixels += pixels

        if idx < 5:
            print(f"Sample {idx}: positive pixels = {pos} / {pixels}")

    frac = total_pos / max(1, total_pixels)
    print(f"\nGlobal positive pixel fraction ~ {frac:.6f} ({frac*100:.4f}%)")


if __name__ == "__main__":
    main()