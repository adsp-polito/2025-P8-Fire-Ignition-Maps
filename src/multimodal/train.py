import torch
import numpy as np
from utils import fast_hist, fire_area_iou  # per_class_iou not needed for this baseline

# ---- Gradient Accumulation Control ----
ACCUM_STEPS = 8  # effective_batch = BATCH_SIZE * ACCUM_STEPS


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute the Dice loss for binary segmentation.

    Args:
        logits: Raw logits from the model of shape (B, 1, H, W).
        targets: Ground-truth mask of shape (B, 1, H, W) or (B, H, W).
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice loss averaged over the batch.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    probs = torch.sigmoid(logits)
    targets = targets.float()
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


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
    Training loop for SINGLE-TASK burned-area segmentation with gradient accumulation.
    """
    model.train()

    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    total_ba_loss = 0.0
    total_lc_loss = 0.0

    printed_device_info = False
    LOG_EVERY = 50

    # ---- IMPORTANT: with accumulation, call zero_grad() ONCE before loop ----
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        (
            image_sentinel,
            image_landsat,
            other_data,
            era5_raster,
            era5_tabular,
            landcover_input,
            gt_mask,
        ) = batch

        image_sentinel = image_sentinel.to(device, non_blocking=True)
        image_landsat = image_landsat.to(device, non_blocking=True)
        other_data = other_data.to(device, non_blocking=True)
        era5_raster = era5_raster.to(device, non_blocking=True)
        era5_tabular = era5_tabular.to(device, non_blocking=True)
        landcover_input = landcover_input.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)

        if not printed_device_info:
            try:
                print(
                    f"[train][epoch {epoch}] device={device} | "
                    f"model_param_device={next(model.parameters()).device} | "
                    f"batch_sentinel_device={image_sentinel.device} | "
                    f"ACCUM_STEPS={ACCUM_STEPS}"
                )
            except StopIteration:
                print(
                    f"[train][epoch {epoch}] device={device} | "
                    f"model has no parameters? (unexpected) | "
                    f"batch_sentinel_device={image_sentinel.device} | "
                    f"ACCUM_STEPS={ACCUM_STEPS}"
                )
            printed_device_info = True

        # Ensure mask has shape [B, 1, H, W]
        if gt_mask.dim() == 3:
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

        bce_loss = burned_area_loss_fn(logits_ba, gt_mask_ch.float())
        dice = dice_loss(logits_ba, gt_mask_ch)
        loss_burned_area = 0.5 * bce_loss + 0.5 * dice

        loss_landcover = torch.tensor(0.0, device=device)

        loss = w_mask * loss_burned_area + w_landcover * loss_landcover

        # ---- Gradient accumulation: scale loss down ----
        loss_scaled = loss / ACCUM_STEPS
        loss_scaled.backward()

        # ---- Step optimizer every ACCUM_STEPS ----
        is_update_step = ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(dataloader))
        if is_update_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Telemetry (optional)
        if device.type == "cuda" and (batch_idx % LOG_EVERY == 0):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"[gpu] batch={batch_idx} | allocated={allocated:.1f}MB | reserved={reserved:.1f}MB")

        total_loss += float(loss.item())
        total_ba_loss += float(loss_burned_area.item())
        total_lc_loss += float(loss_landcover.item())

        with torch.no_grad():
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
    avg_ba_loss = total_ba_loss / max(1, len(dataloader))
    avg_land_loss = total_lc_loss / max(1, len(dataloader))

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
    Validation loop (no grad accumulation needed).
    """
    model.eval()

    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    loss_ba_sum = 0.0
    loss_land_sum = 0.0

    printed_device_info = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            (
                image_sentinel,
                image_landsat,
                other_data,
                era5_raster,
                era5_tabular,
                landcover_input,
                gt_mask,
            ) = batch

            image_sentinel = image_sentinel.to(device, non_blocking=True)
            image_landsat = image_landsat.to(device, non_blocking=True)
            other_data = other_data.to(device, non_blocking=True)
            era5_raster = era5_raster.to(device, non_blocking=True)
            era5_tabular = era5_tabular.to(device, non_blocking=True)
            landcover_input = landcover_input.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)

            if not printed_device_info:
                try:
                    print(
                        f"[val][epoch {epoch}] device={device} | "
                        f"model_param_device={next(model.parameters()).device} | "
                        f"batch_sentinel_device={image_sentinel.device}"
                    )
                except StopIteration:
                    print(
                        f"[val][epoch {epoch}] device={device} | "
                        f"model has no parameters? (unexpected) | "
                        f"batch_sentinel_device={image_sentinel.device}"
                    )
                printed_device_info = True

            if gt_mask.dim() == 3:
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

            bce_loss = burned_area_loss_fn(logits_ba, gt_mask_ch.float())
            dice = dice_loss(logits_ba, gt_mask_ch)
            loss_burned_area = 0.5 * bce_loss + 0.5 * dice
            loss_landcover = torch.tensor(0.0, device=device)

            total_batch_loss = loss_burned_area + loss_landcover
            total_loss += float(total_batch_loss.item())
            loss_ba_sum += float(loss_burned_area.item())
            loss_land_sum += float(loss_landcover.item())

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
    avg_land_loss = loss_land_sum / max(1, len(dataloader))

    if writer is not None:
        writer.add_scalar("Loss/total_val", avg_total_loss, epoch)
        writer.add_scalar("Loss/burned_area_val", avg_ba_loss, epoch)
        writer.add_scalar("Loss/landcover_val", avg_land_loss, epoch)
        writer.add_scalar("IoU/burned_area_val", iou_ba, epoch)
        writer.add_scalar("Metrics/precision_val", precision, epoch)
        writer.add_scalar("Metrics/recall_val", recall, epoch)
        writer.add_scalar("Metrics/f1_val", f1, epoch)

    return avg_total_loss, avg_ba_loss, avg_land_loss, iou_ba
