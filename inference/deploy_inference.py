import os
import torch
import numpy as np

from main import build_model, DEVICE, CHECKPOINT_DIR, model_name
from dataset import PiedmontDataset

# ---- DEPLOY CONFIG ----
THRESHOLD = 0.95  # chosen from val sweep
ROOT_DIR = "../data/"
GEOJSON_PATH = "../geojson/piedmont_2012_2024_fa.geojson"
TARGET_SIZE = (256, 256)

OUTPUT_DIR = "deploy_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@torch.no_grad()
def predict_one_sample(model, sample):
    (
        image_sentinel,
        image_landsat,
        other_data,
        era5_raster,
        era5_tabular,
        landcover_input,
        gt_mask,  # may be unused in deployment
    ) = sample

    # Add batch dim if needed
    if image_sentinel.dim() == 3:
        image_sentinel = image_sentinel.unsqueeze(0)
        image_landsat = image_landsat.unsqueeze(0)
        other_data = other_data.unsqueeze(0)
        era5_raster = era5_raster.unsqueeze(0)
        era5_tabular = era5_tabular.unsqueeze(0)
        landcover_input = landcover_input.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0) if gt_mask is not None and gt_mask.dim() == 3 else gt_mask

    image_sentinel = image_sentinel.to(DEVICE, non_blocking=True)
    image_landsat = image_landsat.to(DEVICE, non_blocking=True)
    other_data = other_data.to(DEVICE, non_blocking=True)
    era5_raster = era5_raster.to(DEVICE, non_blocking=True)
    era5_tabular = era5_tabular.to(DEVICE, non_blocking=True)

    # ignition map = zeros (as in your training baseline)
    # infer spatial from gt_mask if available else from sentinel
    H, W = image_sentinel.shape[-2], image_sentinel.shape[-1]
    ignition_pt = torch.zeros((image_sentinel.size(0), 1, H, W), device=DEVICE)

    logits_ba, _ = model(
        image_sentinel,
        image_landsat,
        other_data,
        ignition_pt,
        era5_raster,
        era5_tabular,
    )

    prob = torch.sigmoid(logits_ba)[0, 0].detach().cpu().numpy()
    pred_mask = (prob > THRESHOLD).astype(np.uint8)
    return prob, pred_mask


def main():
    # Load model
    model = build_model().to(DEVICE)
    ckpt = os.path.join(CHECKPOINT_DIR, model_name)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # Load dataset (deploy: pick whichever subset you want)
    ds = PiedmontDataset(
        root_dir=ROOT_DIR,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False,
        apply_augmentations=False,
        global_stats=None,          # IMPORTANT: use same global_stats used in training if required
        initial_fire_dirs=None,
    )

    # Run a few samples
    for idx in [0, 1, 2, 3, 4]:
        prob, mask = predict_one_sample(model, ds[idx])
        np.save(os.path.join(OUTPUT_DIR, f"prob_{idx}.npy"), prob)
        np.save(os.path.join(OUTPUT_DIR, f"mask_{idx}.npy"), mask)
        print(f"[saved] idx={idx} prob->prob_{idx}.npy mask->mask_{idx}.npy")

    print(f"Done. Threshold used: {THRESHOLD}. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()