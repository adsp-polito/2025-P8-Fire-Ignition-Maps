import os
import torch
import torch.nn as nn

from main import build_datasets_and_loaders, build_model, DEVICE
from train import val


CHECKPOINT_DIR = "paras_model"

checkpoint_path = os.path.join(CHECKPOINT_DIR, "unet_sentinel_best.pth")


def run_inference_on_val():
    # Rebuild datasets and loaders in the same way as training
    _, val_dataset, _, val_loader = build_datasets_and_loaders()

    # Build model and load best checkpoint
    model = build_model().to(DEVICE)
    #checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)

    burned_area_loss_fn = nn.BCEWithLogitsLoss()
    # Dummy landcover loss to satisfy val(...) signature
    landcover_loss_fn = nn.CrossEntropyLoss()

    model.eval()
    avg_total_loss, avg_ba_loss, avg_land_loss, iou_ba = val(
        model=model,
        dataloader=val_loader,
        burned_area_loss_fn=burned_area_loss_fn,
        landcover_loss_fn=landcover_loss_fn,
        device=DEVICE,
        writer=None,
        epoch=0,
    )

    print("=== Inference on Val Dataset ===")
    print(f"Total loss: {avg_total_loss:.4f}")
    print(f"Burned area loss: {avg_ba_loss:.4f}")
    print(f"IoU (burned area): {iou_ba:.4f}")


if __name__ == "__main__":
    run_inference_on_val()