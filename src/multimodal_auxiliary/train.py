import torch
import numpy as np
from utils import fast_hist, fire_area_iou

# ---- Gradient Accumulation Control ----
ACCUM_STEPS = 8  # effective_batch = BATCH_SIZE * ACCUM_STEPS


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary segmentation (burned / not burned)."""
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


def _landcover_targets_from_input(landcover_input: torch.Tensor) -> torch.Tensor:
    """Convert landcover input tensor to class-index targets [B,H,W] (long)."""
    lc = landcover_input
    # Possible formats:
    # 1) [B,H,W] already indices
    # 2) [B,1,H,W] indices in the single channel
    # 3) [B,C,H,W] one-hot or probabilities
    if lc.dim() == 3:
        return lc.long()
    if lc.dim() == 4 and lc.size(1) == 1:
        return lc[:, 0].long()
    if lc.dim() == 4 and lc.size(1) > 1:
        return lc.argmax(dim=1).long()
    raise ValueError(f"Unexpected landcover_input shape: {tuple(lc.shape)}")


def train(
    model,
    optimizer,
    dataloader,
    burned_area_loss_fn,
    landcover_loss_fn,
    device,
    writer,
    epoch,
    w_mask: float,
    w_landcover: float,
    threshold: float = 0.5,
):
    """Training loop for burned-area segmentation + optional landcover auxiliary task."""
    model.train()

    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    total_ba_loss = 0.0
    total_lc_loss = 0.0

    printed_device_info = False
    LOG_EVERY = 50

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
                    f"ACCUM_STEPS={ACCUM_STEPS} | thr={threshold}"
                )
            except StopIteration:
                print(f"[train][epoch {epoch}] device={device} | model has no parameters? (unexpected)")
            printed_device_info = True

        # Ensure mask has shape [B, 1, H, W]
        gt_mask_ch = gt_mask.unsqueeze(1) if gt_mask.dim() == 3 else gt_mask

        ignition_pt = torch.zeros_like(gt_mask_ch, device=device)

        logits_ba, logits_lc = model(
            image_sentinel,
            image_landsat,
            other_data,
            ignition_pt,
            era5_raster,
            era5_tabular,
        )

        # Burned area loss: BCE + Dice
        bce_loss = burned_area_loss_fn(logits_ba, gt_mask_ch.float())
        dice = dice_loss(logits_ba, gt_mask_ch)
        loss_burned_area = 0.5 * bce_loss + 0.5 * dice

        # Landcover auxiliary loss (optional)
        if w_landcover > 0:
            lc_target = _landcover_targets_from_input(landcover_input)
            loss_landcover = landcover_loss_fn(logits_lc, lc_target)
        else:
            loss_landcover = torch.tensor(0.0, device=device)

        loss = w_mask * loss_burned_area + w_landcover * loss_landcover

        loss_scaled = loss / ACCUM_STEPS
        loss_scaled.backward()

        is_update_step = ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(dataloader))
        if is_update_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and (batch_idx % LOG_EVERY == 0):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"[gpu] batch={batch_idx} | allocated={allocated:.1f}MB | reserved={reserved:.1f}MB")

        total_loss += float(loss.item())
        total_ba_loss += float(loss_burned_area.item())
        total_lc_loss += float(loss_landcover.item())

        with torch.no_grad():
            predicted_ba = (torch.sigmoid(logits_ba) > threshold).float()
            targets_ba_flat = gt_mask_ch.squeeze(1).detach().cpu().numpy().astype(int).flatten()
            predicted_ba_flat = predicted_ba.squeeze(1).detach().cpu().numpy().astype(int).flatten()
            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)

    iou_ba = fire_area_iou(hist_ba)

    tn, fp, fn, tp = hist_ba[0, 0], hist_ba[0, 1], hist_ba[1, 0], hist_ba[1, 1]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_total_loss = total_loss / max(1, len(dataloader))
    avg_ba_loss = total_ba_loss / max(1, len(dataloader))
    avg_lc_loss = total_lc_loss / max(1, len(dataloader))

    if writer is not None:
        writer.add_scalar("Loss/total_train", avg_total_loss, epoch)
        writer.add_scalar("Loss/burned_area_train", avg_ba_loss, epoch)
        writer.add_scalar("Loss/landcover_train", avg_lc_loss, epoch)
        writer.add_scalar("IoU/burned_area_train", iou_ba, epoch)
        writer.add_scalar("Metrics/precision_train", precision, epoch)
        writer.add_scalar("Metrics/recall_train", recall, epoch)
        writer.add_scalar("Metrics/f1_train", f1, epoch)

    return avg_total_loss, avg_ba_loss, avg_lc_loss, iou_ba


def val(
    model,
    dataloader,
    burned_area_loss_fn,
    landcover_loss_fn,
    device,
    writer,
    epoch,
    w_mask: float = 1.0,
    w_landcover: float = 0.0,
    threshold: float = 0.5,
):
    """Validation loop (no grad accumulation)."""
    model.eval()

    hist_ba = np.zeros((2, 2), dtype=np.int64)

    total_loss = 0.0
    loss_ba_sum = 0.0
    loss_lc_sum = 0.0

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
                        f"batch_sentinel_device={image_sentinel.device} | thr={threshold}"
                    )
                except StopIteration:
                    print(f"[val][epoch {epoch}] device={device} | model has no parameters? (unexpected)")
                printed_device_info = True

            gt_mask_ch = gt_mask.unsqueeze(1) if gt_mask.dim() == 3 else gt_mask
            ignition_pt = torch.zeros_like(gt_mask_ch, device=device)

            logits_ba, logits_lc = model(
                image_sentinel,
                image_landsat,
                other_data,
                ignition_pt,
                era5_raster,
                era5_tabular,
            )

            bce_loss = burned_area_loss_fn(logits_ba, gt_mask_ch.float())
            dice = dice_loss(logits_ba, gt_mask_ch)
            loss_burned_area = 0.5 * bce_loss + 0.5 * dice

            lc_target = _landcover_targets_from_input(landcover_input)
            loss_landcover = landcover_loss_fn(logits_lc, lc_target)

            total_batch_loss = w_mask * loss_burned_area + w_landcover * loss_landcover
            total_loss += float(total_batch_loss.item())
            loss_ba_sum += float(loss_burned_area.item())
            loss_lc_sum += float(loss_landcover.item())

            predicted_ba = (torch.sigmoid(logits_ba) > threshold).float()
            targets_ba_flat = gt_mask_ch.squeeze(1).detach().cpu().numpy().astype(int).flatten()
            predicted_ba_flat = predicted_ba.squeeze(1).detach().cpu().numpy().astype(int).flatten()
            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)

    iou_ba = fire_area_iou(hist_ba)

    tn, fp, fn, tp = hist_ba[0, 0], hist_ba[0, 1], hist_ba[1, 0], hist_ba[1, 1]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_total_loss = total_loss / max(1, len(dataloader))
    avg_ba_loss = loss_ba_sum / max(1, len(dataloader))
    avg_lc_loss = loss_lc_sum / max(1, len(dataloader))

    if writer is not None:
        writer.add_scalar("Loss/total_val", avg_total_loss, epoch)
        writer.add_scalar("Loss/burned_area_val", avg_ba_loss, epoch)
        writer.add_scalar("Loss/landcover_val", avg_lc_loss, epoch)
        writer.add_scalar("IoU/burned_area_val", iou_ba, epoch)
        writer.add_scalar("Metrics/precision_val", precision, epoch)
        writer.add_scalar("Metrics/recall_val", recall, epoch)
        writer.add_scalar("Metrics/f1_val", f1, epoch)

    return avg_total_loss, avg_ba_loss, avg_lc_loss, iou_ba
