
import os
import torch
import torch.nn as nn

from main import (
    build_datasets_and_loaders,
    build_model,
    DEVICE,
    CHECKPOINT_DIR,
    model_name,
    set_seed,
)
from train import val


def run_inference_on_val():
    # MUST match training split
    set_seed(42)

    # Rebuild datasets/loaders exactly like training
    _, val_dataset, _, val_loader = build_datasets_and_loaders()

    # Load model
    model = build_model().to(DEVICE)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Check CHECKPOINT_DIR/model_name in main.py or your saved files."
        )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Losses (must match training)
    burned_area_loss_fn = nn.BCEWithLogitsLoss()
    landcover_loss_fn = nn.CrossEntropyLoss()

    avg_total_loss, avg_ba_loss, avg_lc_loss, iou_ba = val(
        model=model,
        dataloader=val_loader,
        burned_area_loss_fn=burned_area_loss_fn,
        landcover_loss_fn=landcover_loss_fn,
        device=DEVICE,
        writer=None,
        epoch=0,
    )

    print("=== Inference on Val Dataset (REPRODUCIBLE) ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total loss: {avg_total_loss:.4f}")
    print(f"Burned area loss: {avg_ba_loss:.4f}")
    print(f"IoU (burned area @ thr=0.5): {iou_ba:.4f}")


if __name__ == "__main__":
    run_inference_on_val()