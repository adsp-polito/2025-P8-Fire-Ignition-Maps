import os
from collections import defaultdict
from typing import Tuple, Optional, Dict, Any

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp  # kept for compatibility; may be used elsewhere


def resample_image(
    input_tif_path: str,
    output_tif_path: str,
    target_transform: Tuple,
    target_crs: rasterio.crs.CRS,
    target_shape: Tuple[int, int],
    resampling_method: Resampling = Resampling.bilinear,
) -> None:
    """
    Resample a multi-band raster to a target grid and write it to disk,
    preserving band descriptions when possible.

    Args:
        input_tif_path: Path to the input GeoTIFF.
        output_tif_path: Path where the resampled GeoTIFF will be written.
        target_transform: Affine transform of the target grid.
        target_crs: CRS of the target grid.
        target_shape: (height, width) of the target grid.
        resampling_method: Rasterio Resampling method to use.
    """
    with rasterio.open(input_tif_path) as src:
        # Allocate array for resampled data: (bands, H, W)
        resampled_data = np.empty(
            (src.count, target_shape[0], target_shape[1]),
            dtype=src.dtypes[0],
        )

        # Keep band descriptions if they exist
        band_descriptions = list(src.descriptions) if src.descriptions is not None else [None] * src.count

        # Reproject each band independently
        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=resampled_data[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method,
            )

        # Prepare output profile
        profile = src.profile.copy()
        profile.update(
            width=target_shape[1],
            height=target_shape[0],
            transform=target_transform,
            crs=target_crs,
            nodata=src.nodata,
            dtype=resampled_data.dtype,
            count=src.count,
        )

        # Write resampled raster and propagate band descriptions
        os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(resampled_data)
            for i, desc in enumerate(band_descriptions):
                if desc is not None:
                    dst.set_band_description(i + 1, desc)


def read_image_and_metadata(image_path: str):
    """
    Read a raster image and its key metadata.

    Returns:
        img_data: np.ndarray of shape (C, H, W)
        transform: rasterio Affine transform
        crs: rasterio CRS
        shape: (height, width)
        resolution: (xres, yres)
        band_descriptions: list of band description strings
    """
    with rasterio.open(image_path) as src:
        img_data = src.read()  # (C, H, W)
        resolution = (src.res[0], src.res[1])
        band_descriptions = (
            list(src.descriptions)
            if src.descriptions is not None
            else [f"Band_{i + 1}" for i in range(src.count)]
        )
        return img_data, src.transform, src.crs, src.shape, resolution, band_descriptions


def get_cloud_score_on_gt(gt_mask_path: str, cm_mask_path: str) -> Optional[float]:
    """
    Compute a cloud score over the fire area defined by a ground-truth mask.

    The score is the sum of the cloud-mask pixel values where the GT mask equals 1.

    Args:
        gt_mask_path: Path to the GT mask TIFF (fire area).
        cm_mask_path: Path to the cloud mask TIFF.

    Returns:
        Cloud score as a float, or None if something goes wrong.
    """
    try:
        with rasterio.open(gt_mask_path) as gt_src, rasterio.open(cm_mask_path) as cm_src:
            # Basic consistency checks
            if (gt_src.crs != cm_src.crs) or (gt_src.transform != cm_src.transform) or (gt_src.shape != cm_src.shape):
                # If grids do not match, reproject CM to GT grid
                cm_reprojected = np.zeros(gt_src.shape, dtype=cm_src.dtypes[0])
                reproject(
                    source=rasterio.band(cm_src, 1),
                    destination=cm_reprojected,
                    src_transform=cm_src.transform,
                    src_crs=cm_src.crs,
                    dst_transform=gt_src.transform,
                    dst_crs=gt_src.crs,
                    resampling=Resampling.nearest,
                )
                cm_data = cm_reprojected
            else:
                cm_data = cm_src.read(1)

            gt_data = gt_src.read(1)
            gt_binary = (gt_data > 0).astype(np.uint8)

            cloud_score = float(np.sum(cm_data * gt_binary))
            return cloud_score
    except Exception as e:
        print(f"[get_cloud_score_on_gt] Error computing cloud score: {e}")
        return None


def find_best_image_in_folder(fire_folder_path: str) -> Optional[Dict[str, Any]]:
    """
    Find the best Sentinel image in a fire folder, based on cloud-mask score.

    Expected pattern (can be adjusted if needed):
        - Pre-fire Sentinel images contain 'pre_sentinel' and end with '.tif', and do NOT contain '_CM'.
        - Corresponding cloud masks have the same name plus '_CM.tif'.

    Args:
        fire_folder_path: Path to a folder for a single fire, e.g. 'piedmont/fire_1234'.

    Returns:
        A dict with keys:
            - 'sentinel_path': path to the chosen Sentinel image
            - 'cm_path': path to the corresponding cloud mask (or None)
            - 'score': cloud score (float; lower is better),
          or None if no valid Sentinel image is found.
    """
    # 1. Find GT mask (not strictly required to pick a file, but needed for cloud score)
    gt_mask_sentinel_path: Optional[str] = None
    for fname in os.listdir(fire_folder_path):
        if "GTSentinel" in fname and fname.endswith(".tif"):
            gt_mask_sentinel_path = os.path.join(fire_folder_path, fname)
            break

    sentinel_files = []
    for fname in os.listdir(fire_folder_path):
        if "pre_sentinel" in fname and fname.endswith(".tif") and "_CM" not in fname:
            sentinel_files.append(os.path.join(fire_folder_path, fname))

    if not sentinel_files:
        print(f"[find_best_image_in_folder] No Sentinel images found in {fire_folder_path}.")
        return None

    # If we cannot compute cloud scores (no GT or CM), just pick the first Sentinel image
    if gt_mask_sentinel_path is None:
        sentinel_path = sentinel_files[0]
        return {
            "sentinel_path": sentinel_path,
            "cm_path": None,
            "score": float("inf"),
            "path": sentinel_path,  # compatibility alias
        }

    image_scores = []
    for sentinel_path in sentinel_files:
        # Derive expected CM path from Sentinel name
        base, ext = os.path.splitext(sentinel_path)
        cm_path = base + "_CM" + ext
        if not os.path.exists(cm_path):
            continue

        score = get_cloud_score_on_gt(gt_mask_sentinel_path, cm_path)
        if score is None:
            continue

        image_scores.append(
            {
                "sentinel_path": sentinel_path,
                "cm_path": cm_path,
                "score": score,
                "path": sentinel_path,  # alias for compatibility
            }
        )

    if not image_scores:
        # Fallback: no CM-based scores, pick the first Sentinel file
        sentinel_path = sentinel_files[0]
        return {
            "sentinel_path": sentinel_path,
            "cm_path": None,
            "score": float("inf"),
            "path": sentinel_path,
        }

    # Choose the image with the lowest cloud score
    best_image = min(image_scores, key=lambda x: x["score"])
    return best_image


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """
    Compute per-class Intersection over Union from a confusion matrix.

    Args:
        hist: Confusion matrix of shape (n_classes, n_classes)
              hist[i, j] = number of pixels of class i predicted as class j.

    Returns:
        IoU array of shape (n_classes,).
    """
    epsilon = 1e-5
    # hist.sum(1): ground-truth pixels per class
    # hist.sum(0): predicted pixels per class
    true_positives = np.diag(hist)
    denominator = hist.sum(1) + hist.sum(0) - true_positives + epsilon
    return true_positives / denominator


def poly_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    iter: int,
    max_iter: int,
    power: float = 0.9,
) -> float:
    """
    Polynomial learning rate scheduler.

    lr(iter) = init_lr * (1 - iter / max_iter) ** power

    Args:
        optimizer: Optimizer whose learning rate will be updated.
        init_lr: Initial learning rate.
        iter: Current iteration (0-based).
        max_iter: Maximum number of iterations.
        power: Polynomial power.

    Returns:
        The updated learning rate.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    factor = max(0.0, 1.0 - float(iter) / float(max_iter))
    lr = init_lr * (factor ** power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def fire_area_iou(hist: np.ndarray) -> float:
    """
    Compute IoU for the binary 'fire area' problem, given a 2x2 confusion matrix.

    hist[1,1] = true positives (fire correctly predicted)
    hist[0,1] = false positives
    hist[1,0] = false negatives
    hist[0,0] = true negatives

    Returns:
        IoU for the fire class (class 1).
    """
    if hist.shape != (2, 2):
        raise ValueError("fire_area_iou expects a 2x2 confusion matrix for a binary problem.")

    true_positives = hist[1, 1]
    false_positives = hist[0, 1]
    false_negatives = hist[1, 0]

    denominator = true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return float(true_positives) / float(denominator)


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Compute a confusion matrix (histogram) for segmentation.

    Args:
        a: Predicted labels (H, W) or flattened array.
        b: Ground-truth labels (H, W) or flattened array.
        n: Number of classes.

    Returns:
        Confusion matrix of shape (n, n), where
        hist[i, j] = count of pixels with GT = i and prediction = j.
    """
    a = a.flatten()
    b = b.flatten()
    k = (a >= 0) & (a < n)
    # Map pairs (gt, pred) to a single index and count
    hist = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    return hist