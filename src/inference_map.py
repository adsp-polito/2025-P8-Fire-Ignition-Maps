import os
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from main import build_datasets_and_loaders, build_model, DEVICE, CHECKPOINT_DIR

VIS_DIR = "inference_maps"


def _extract_burned_logits(model_output: Any) -> torch.Tensor:
    """
    Extract burned-area logits from model output.

    Supports:
    - Tensor: direct logits [B, 1, H, W]
    - Dict: looks for burned-area keys
    - Tuple/List: uses the first element
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, dict):
        for key in ["burned_area", "burned_area_logits", "ba_logits", "ba"]:
            if key in model_output:
                return model_output[key]
        raise KeyError(
            "Model output is a dict but no burned-area key was found. "
            "Expected one of: 'burned_area', 'burned_area_logits', 'ba_logits', 'ba'. "
            f"Got keys: {list(model_output.keys())}"
        )

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model output is an empty tuple/list.")
        return model_output[0]

    raise TypeError(
        f"Unsupported model output type: {type(model_output)}. "
        f"Expected Tensor, dict, or tuple/list."
    )


def _tensor_to_rgb_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert [C, H, W] tensor (Sentinel bands) into displayable RGB [H, W, 3] in [0,1].

    Uses first 3 channels as pseudo-RGB (or repeats if fewer than 3).
    Applies per-image min-max normalization for visualization.
    """
    if t.ndim != 3:
        raise ValueError(f"Expected tensor with shape [C, H, W], got {t.shape}.")

    c, _, _ = t.shape

    if c >= 3:
        rgb = t[:3, :, :]
    else:
        rgb = t.repeat(3 // c + 1, 1, 1)[:3, :, :]

    rgb_np = rgb.numpy()
    rgb_min = rgb_np.min()
    rgb_max = rgb_np.max()
    denom = rgb_max - rgb_min

    if denom < 1e-6:
        rgb_norm = np.zeros_like(rgb_np)
    else:
        rgb_norm = (rgb_np - rgb_min) / denom

    rgb_norm = np.clip(rgb_norm, 0.0, 1.0)
    rgb_norm = np.transpose(rgb_norm, (1, 2, 0))  # [H, W, 3]

    return rgb_norm


def visualize_val_samples(
    num_batches: int = 3,
    max_per_batch: int = 4,
    save_dir: str = VIS_DIR,
) -> None:
    """
    Run the trained model on a few validation batches and save 3-panel figures:

        - Sentinel input (pseudo-RGB)
        - Ground-truth burned area mask
        - Predicted burned area mask

    Assumes PiedmontDataset __getitem__ returns:
        (sentinel, landsat, other_data, era5_raster, era5_tabular, landcover, gt_mask)

    And MultiModalFPN.forward expects:
        forward(sentinel_image, landsat_image, dem_image, ignition_map, era5_raster, era5_tabular)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Rebuild datasets/loaders exactly as in training
    _, val_dataset, _, val_loader = build_datasets_and_loaders()

    # Load model + checkpoint
    model = build_model().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train the model first so that 'best_model.pth' exists."
        )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Iterate over a few batches and save images
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= num_batches:
                break

            # Batch is expected to be a list/tuple of length 7 from PiedmontDataset:
            # (sentinel, landsat, other_data, era5_raster, era5_tabular, landcover, gt_mask)
            if not isinstance(batch, (list, tuple)) or len(batch) < 7:
                raise TypeError(
                    "Expected batch from PiedmontDataset to be a list/tuple of length 7: "
                    "(sentinel, landsat, other_data, era5_raster, era5_tabular, landcover, gt_mask). "
                    f"Got type={type(batch)}, len={len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}."
                )

            sentinel_images = batch[0].to(DEVICE)    # [B, C_sentinel, H, W]
            landsat_images = batch[1].to(DEVICE)     # [B, C_landsat, H, W]
            other_data = batch[2].to(DEVICE)         # [B, 2, H, W] = DEM + streets
            era5_raster = batch[3].to(DEVICE)        # [B, C_era5, H, W]
            era5_tabular = batch[4].to(DEVICE)       # [B, C_tabular]
            # landcover = batch[5]  # unused here
            gt_masks = batch[6].to(DEVICE)           # [B, 1, H, W] or [B, H, W]

            # Split other_data back into DEM and ignition_map placeholders
            # In dataset: other_data_tensor = concat([dem_tensor, streets_tensor], dim=0)
            # so channel 0 = DEM, channel 1 = streets. We map:
            #   dem_image    <- other_data[:, 0:1, ...]
            #   ignition_map <- other_data[:, 1:2, ...]
            dem_image = other_data[:, 0:1, :, :]
            ignition_map = other_data[:, 1:2, :, :]

            # Ensure gt_masks has shape [B, 1, H, W]
            if gt_masks.ndim == 3:
                gt_masks = gt_masks.unsqueeze(1)

            # Forward pass through the full multimodal model
            outputs = model(
                sentinel_images,
                landsat_images,
                dem_image,
                ignition_map,
                era5_raster,
                era5_tabular,
            )

            burned_logits = _extract_burned_logits(outputs)  # [B, 1, H, W] expected

            probs = torch.sigmoid(burned_logits)
            preds = (probs > 0.5).float()

            batch_size = sentinel_images.size(0)
            n_to_show = min(batch_size, max_per_batch)

            for i in range(n_to_show):
                img_t = sentinel_images[i].cpu()   # [C, H, W]
                gt_t = gt_masks[i, 0].cpu()        # [H, W]
                pr_t = preds[i, 0].cpu()           # [H, W]

                rgb_img = _tensor_to_rgb_image(img_t)
                gt_np = gt_t.numpy()
                pr_np = pr_t.numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                axs[0].imshow(rgb_img)
                axs[0].set_title("Sentinel input (pseudo-RGB)")
                axs[0].axis("off")

                axs[1].imshow(gt_np, cmap="gray")
                axs[1].set_title("Ground truth burned area")
                axs[1].axis("off")

                axs[2].imshow(pr_np, cmap="gray")
                axs[2].set_title("Predicted burned area")
                axs[2].axis("off")

                fig.tight_layout()
                out_path = os.path.join(save_dir, f"val_b{b_idx}_i{i}.png")
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                print(f"Saved visualization: {out_path}")


def run_inference_on_single_tile(tile_path: str, output_path: str) -> None:
    """
    Inference on ONE preprocessed Sentinel tile saved as .npy [C, H, W].

    NOTE: That tile must be preprocessed exactly like in your dataset
    (same bands, scaling, normalization, resize).

    For the multimodal model, we build zero placeholders for the other modalities.
    """
    # Load model
    model = build_model().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train the model first so that 'best_model.pth' exists."
        )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Load Sentinel tile
    arr = np.load(tile_path)  # [C_sentinel, H, W]
    if arr.ndim != 3:
        raise ValueError(f"Expected tile with shape [C, H, W], got {arr.shape}.")

    sentinel = torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE)  # [1, C_sentinel, H, W]
    _, _, H, W = sentinel.shape

    # Build zero placeholders for other modalities consistent with build_model()
    # From main.build_model():
    #   in_channels_landsat=16
    #   in_channels_other_data=2   (DEM + streets)
    #   in_channels_era5_raster=2
    #   in_channels_era5_tabular=1
    landsat = torch.zeros((1, 16, H, W), device=DEVICE, dtype=torch.float32)
    other_data = torch.zeros((1, 2, H, W), device=DEVICE, dtype=torch.float32)
    dem_image = other_data[:, 0:1, :, :]
    ignition_map = other_data[:, 1:2, :, :]
    era5_raster = torch.zeros((1, 2, H, W), device=DEVICE, dtype=torch.float32)
    era5_tabular = torch.zeros((1, 1), device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            sentinel,
            landsat,
            dem_image,
            ignition_map,
            era5_raster,
            era5_tabular,
        )
        burned_logits = _extract_burned_logits(outputs)        # [1, 1, H, W]
        probs = torch.sigmoid(burned_logits)
        pred_mask = (probs > 0.5).float()[0, 0].cpu().numpy()  # [H, W]

    plt.figure(figsize=(5, 5))
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")
    plt.title("Predicted burned area")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved single-tile prediction mask to {output_path}")


def run_inference_map() -> None:
    visualize_val_samples(num_batches=3, max_per_batch=4, save_dir=VIS_DIR)


if __name__ == "__main__":
    run_inference_map()