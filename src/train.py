import torch
import numpy as np
from utils import fast_hist, fire_area_iou  # per_class_iou not needed for this baseline


def train(
    model,
    optimizer,
    dataloader,
    burned_area_loss_fn,
    landcover_loss_fn,  # kept for signature compatibility, but unused
    device,
    writer,
    epoch,
    w_mask: float,
    w_landcover: float,
):
    """
    Training loop for SINGLE-TASK burned-area segmentation.

    Current PiedmontDataset __getitem__ returns:
        (
            image_sentinel,   # [B, 12, H, W]
            image_landsat,    # [B, 16, H, W] (placeholder)
            other_data,       # [B, 2, H, W]  (DEM + streets, placeholder)
            era5_raster,      # [B, 2, H, W]  (placeholder)
            era5_tabular,     # [B, 1]        (placeholder)
            landcover_input,  # [B, 1, H, W]  (placeholder, NOT GT)
            gt_mask,          # [B, 1, H, W] OR [B, H, W] depending on path
        )

    Baseline behavior:
        - Only burned-area head is trained.
        - Landcover head is ignored (loss = 0).
        - Ignition map is synthesized as zeros.
    """
    model.train()

    # Confusion matrix for burned area (binary)
    # hist_ba[gt, pred]: rows = GT {0,1}, cols = prediction {0,1}
    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    total_ba_loss = 0.0
    total_lc_loss = 0.0  # stays 0.0 in this baseline

    for batch_idx, batch in enumerate(dataloader):
        (
            image_sentinel,
            image_landsat,
            other_data,
            era5_raster,
            era5_tabular,
            landcover_input,  # not used as GT here
            gt_mask,
        ) = batch

        image_sentinel = image_sentinel.to(device)
        image_landsat = image_landsat.to(device)
        other_data = other_data.to(device)
        era5_raster = era5_raster.to(device)
        era5_tabular = era5_tabular.to(device)
        landcover_input = landcover_input.to(device)  # unused, but moved for completeness
        gt_mask = gt_mask.to(device)

        # Ensure mask has shape [B, 1, H, W] for BCEWithLogitsLoss
        if gt_mask.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            gt_mask_ch = gt_mask.unsqueeze(1)
        else:
            gt_mask_ch = gt_mask  # already [B, 1, H, W]

        # Dummy ignition map (all zeros, same spatial shape as mask)
        ignition_pt = torch.zeros_like(gt_mask_ch, device=device)

        # Forward pass through the multimodal model
        # NOTE: model internally uses only Sentinel in this baseline
        outputs_burned_area, _ = model(
            image_sentinel,
            image_landsat,
            other_data,
            ignition_pt,
            era5_raster,
            era5_tabular,
        )

        # outputs_burned_area is [B, 1, H, W]; match it directly with gt_mask_ch
        logits_ba = outputs_burned_area

        # Burned-area loss
        loss_burned_area = burned_area_loss_fn(
            logits_ba,
            gt_mask_ch.float(),
        )

        # Landcover loss is disabled in this baseline
        loss_landcover = torch.tensor(0.0, device=device)

        # Total loss is effectively just burned-area loss
        loss = w_mask * loss_burned_area + w_landcover * loss_landcover

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_ba_loss += float(loss_burned_area.item())
        total_lc_loss += float(loss_landcover.item())

        # --- Metrics: burned-area IoU components ---
        with torch.no_grad():
            # logits_ba: [B, 1, H, W], gt_mask_ch: [B, 1, H, W]
            predicted_ba = (torch.sigmoid(logits_ba) > 0.5).float()

            targets_ba_flat = gt_mask_ch.squeeze(1).cpu().numpy().astype(int).flatten()
            predicted_ba_flat = predicted_ba.squeeze(1).cpu().numpy().astype(int).flatten()

            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)

    # Epoch-level metrics
    iou_ba = fire_area_iou(hist_ba)

    # Derive precision, recall, F1 from confusion matrix
    tn, fp, fn, tp = (
        hist_ba[0, 0],
        hist_ba[0, 1],
        hist_ba[1, 0],
        hist_ba[1, 1],
    )
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_total_loss = total_loss / max(1, len(dataloader))
    avg_ba_loss = total_ba_loss / max(1, len(dataloader))
    avg_land_loss = total_lc_loss / max(1, len(dataloader))  # stays 0.0

    # Optional TensorBoard logging
    if writer is not None:
        writer.add_scalar("Loss/total_train", avg_total_loss, epoch)
        writer.add_scalar("Loss/burned_area_train", avg_ba_loss, epoch)
        writer.add_scalar("Loss/landcover_train", avg_land_loss, epoch)
        writer.add_scalar("IoU/burned_area_train", iou_ba, epoch)
        writer.add_scalar("Metrics/precision_train", precision, epoch)
        writer.add_scalar("Metrics/recall_train", recall, epoch)
        writer.add_scalar("Metrics/f1_train", f1, epoch)

    return avg_total_loss, avg_ba_loss, avg_land_loss, iou_ba


def val(
    model,
    dataloader,
    burned_area_loss_fn,
    landcover_loss_fn,  # kept for signature compatibility, but unused
    device,
    writer,
    epoch,
):
    """
    Validation loop for SINGLE-TASK burned-area segmentation.

    Same assumptions as in train():
        - 7-tuple batches from the dataset
        - Only burned-area head is evaluated
    """
    model.eval()

    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    loss_ba_sum = 0.0
    loss_land_sum = 0.0  # stays 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            (
                image_sentinel,
                image_landsat,
                other_data,
                era5_raster,
                era5_tabular,
                landcover_input,  # not used in baseline
                gt_mask,
            ) = batch

            image_sentinel = image_sentinel.to(device)
            image_landsat = image_landsat.to(device)
            other_data = other_data.to(device)
            era5_raster = era5_raster.to(device)
            era5_tabular = era5_tabular.to(device)
            landcover_input = landcover_input.to(device)
            gt_mask = gt_mask.to(device)

            # Ensure mask has shape [B, 1, H, W]
            if gt_mask.dim() == 3:  # [B, H, W]
                gt_mask_ch = gt_mask.unsqueeze(1)
            else:
                gt_mask_ch = gt_mask

            ignition_pt = torch.zeros_like(gt_mask_ch, device=device)

            outputs_burned_area, _ = model(
                image_sentinel,
                image_landsat,
                other_data,
                ignition_pt,
                era5_raster,
                era5_tabular,
            )

            logits_ba = outputs_burned_area

            loss_burned_area = burned_area_loss_fn(
                logits_ba,
                gt_mask_ch.float(),
            )
            loss_landcover = torch.tensor(0.0, device=device)

            total_batch_loss = loss_burned_area + loss_landcover
            total_loss += float(total_batch_loss.item())
            loss_ba_sum += float(loss_burned_area.item())
            loss_land_sum += float(loss_landcover.item())

            # Metrics
            predicted_ba = (torch.sigmoid(logits_ba) > 0.5).float()
            targets_ba_flat = gt_mask_ch.squeeze(1).cpu().numpy().astype(int).flatten()
            predicted_ba_flat = predicted_ba.squeeze(1).cpu().numpy().astype(int).flatten()
            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)

    iou_ba = fire_area_iou(hist_ba)

    tn, fp, fn, tp = (
        hist_ba[0, 0],
        hist_ba[0, 1],
        hist_ba[1, 0],
        hist_ba[1, 1],
    )
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_total_loss = total_loss / max(1, len(dataloader))
    avg_ba_loss = loss_ba_sum / max(1, len(dataloader))
    avg_land_loss = loss_land_sum / max(1, len(dataloader))  # 0.0

    if writer is not None:
        writer.add_scalar("Loss/total_val", avg_total_loss, epoch)
        writer.add_scalar("Loss/burned_area_val", avg_ba_loss, epoch)
        writer.add_scalar("Loss/landcover_val", avg_land_loss, epoch)
        writer.add_scalar("IoU/burned_area_val", iou_ba, epoch)
        writer.add_scalar("Metrics/precision_val", precision, epoch)
        writer.add_scalar("Metrics/recall_val", recall, epoch)
        writer.add_scalar("Metrics/f1_val", f1, epoch)

    return avg_total_loss, avg_ba_loss, avg_land_loss, iou_ba