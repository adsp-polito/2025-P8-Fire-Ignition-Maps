import os
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F  # may be used later in model code
from albumentations.pytorch import ToTensorV2

from augmentations import SentinelAugmentations
from utils import find_best_image_in_folder

# -----------------------------
# Constants
# -----------------------------

# Number of spectral bands
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15  # real Landsat bands (without the flag)
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1  # extra presence-flag band

# ERA5 structure
N_BANDS_ERA5 = 3
N_BANDS_ERA5_RASTER = 2      # raster-like ERA5 bands (e.g. wind components)
N_BANDS_ERA5_TABULAR = 1     # tabular-like ERA5 bands (e.g. aggregated variable)
ERA5_TABULAR_BAND_INDICES = [0]      # indices of tabular bands in full ERA5 array
ERA5_RASTER_BAND_INDICES = [1, 2]    # indices of raster bands in full ERA5 array

LANDCOVER_CLASSES = 12  # number of landcover classes (0..11)


class PiedmontDataset(torch.utils.data.Dataset):
    """
    Dataset for fire patches in Piedmont.

    Public interface is stable:
    - __init__(...)
    - __len__()
    - __getitem__(idx) -> 7-tuple
        (sentinel_image, landsat_image, other_data, era5_raster, era5_tabular, landcover, gt_mask)
    """

    def __init__(
        self,
        root_dir: str,
        geojson_path: str,
        target_size: Tuple[int, int] = (256, 256),
        compute_stats: bool = True,
        apply_augmentations: bool = False,
        global_stats: Optional[Dict[str, torch.Tensor]] = None,
        initial_fire_dirs: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.geojson_path = geojson_path
        self.target_size = target_size
        self.apply_augmentations = apply_augmentations

        # fire_id -> fire_date (YYYY-MM-DD or "N/A")
        self.fire_id_to_date_map = self._load_fire_dates_from_geojson()

        # Map: fire_dir -> sample_type ('both', 'sentinel_only', 'unknown')
        self.sample_type_map: Dict[str, str] = {}

        # Decide which fire directories to use
        if initial_fire_dirs is not None:
            self.fire_dirs = initial_fire_dirs
            print(f"Using pre-filtered list of fire directories: {len(self.fire_dirs)} directories.")
            self._populate_sample_type_map_for_initial_dirs()
        else:
            self.fire_dirs = self._get_relevant_fire_dirs()
            self._populate_sample_type_map()

        # Global stats for normalization
        if compute_stats and global_stats is None:
            print("No global_stats provided. Computing new global statistics for normalization...")
            self.global_stats = self._compute_global_stats_mean_std()
        elif global_stats is not None:
            self.global_stats = global_stats
        else:
            self.global_stats = None

        # Binning samples by number of burned pixels
        self.bin_thresholds = [1000, 5000, 20000]  # pixel thresholds
        self.sample_pixel_counts: Dict[int, int] = {}
        self.sample_bins: Dict[int, int] = {}
        self._map_samples_to_bins()

        # Train-time augmentations
        self.augmentor = SentinelAugmentations(target_size=self.target_size)

        # Eval-time transform (no geometric augmentations)
        self.eval_transform = A.Compose([
            ToTensorV2()
        ])

    # -----------------------------
    # Fire directory / date helpers
    # -----------------------------

    def _populate_sample_type_map_for_initial_dirs(self) -> None:
        """
        Populate self.sample_type_map for directories already given in initial_fire_dirs.
        """
        print("Populating sample_type_map for pre-filtered directories...")
        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)
            has_sentinel = any(
                ("pre_sentinel" in f) and f.endswith(".tif") and ("_CM" not in f)
                for f in files
            )
            has_landsat = any(
                ("pre_landsat" in f) and ("_10m.tif" in f) and ("_CM" not in f)
                for f in files
            )

            if has_sentinel and has_landsat:
                self.sample_type_map[fire_dir] = "both"
            elif has_sentinel:
                self.sample_type_map[fire_dir] = "sentinel_only"
            else:
                self.sample_type_map[fire_dir] = "unknown"
        print(f"Populated sample_type_map with {len(self.sample_type_map)} entries.")

    def _load_fire_dates_from_geojson(self) -> Dict[int, str]:
        """
        Load the GeoJSON and construct a mapping fire_id -> formatted fire_date.
        If 'Data_Foc' is missing or invalid, 'N/A' is used.
        """
        fire_id_to_date: Dict[int, str] = {}
        if not os.path.isfile(self.geojson_path):
            print(f"Warning: geojson file {self.geojson_path} not found. Date mapping will be empty.")
            return fire_id_to_date

        gdf = gpd.read_file(self.geojson_path)
        if "id" not in gdf.columns:
            raise KeyError("The GeoJSON must contain an 'id' column for fire IDs.")
        if "Data_Foc" not in gdf.columns:
            print("Warning: 'Data_Foc' column not found in GeoJSON. All dates will be set to 'N/A'.")
            for _, row in gdf.iterrows():
                fire_id_to_date[row["id"]] = "N/A"
            return fire_id_to_date

        for _, row in gdf.iterrows():
            fire_id = row["id"]
            data_foc = row["Data_Foc"]

            fire_date_str = "N/A"
            if isinstance(data_foc, str):
                try:
                    date_obj = datetime.strptime(data_foc, "%d/%m/%Y")
                    fire_date_str = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Warning: Unable to parse date '{data_foc}' for fire_id={fire_id}. Using 'N/A'.")
            elif isinstance(data_foc, (pd.Timestamp, datetime)):
                fire_date_str = data_foc.strftime("%Y-%m-%d")

            fire_id_to_date[fire_id] = fire_date_str

        return fire_id_to_date

    def _populate_sample_type_map(self) -> None:
        """
        Scan fire directories and classify each as:
        - 'both' (Sentinel + Landsat),
        - 'sentinel_only',
        - 'unknown' if no valid Sentinel was found.
        """
        print("Populating sample_type_map by scanning fire directories...")
        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)
            has_sentinel = any(
                ("pre_sentinel" in f) and f.endswith(".tif") and ("_CM" not in f)
                for f in files
            )
            has_landsat = any(
                ("pre_landsat" in f) and ("_10m.tif" in f) and ("_CM" not in f)
                for f in files
            )

            if has_sentinel and has_landsat:
                self.sample_type_map[fire_dir] = "both"
            elif has_sentinel:
                self.sample_type_map[fire_dir] = "sentinel_only"
            else:
                self.sample_type_map[fire_dir] = "unknown"
        print(f"sample_type_map populated with {len(self.sample_type_map)} entries.")

    def _get_relevant_fire_dirs(self) -> List[str]:
        """
        Returns fire_* directories that contain at least one pre_sentinel TIFF and a GTSentinel mask.
        """
        fire_dirs = [
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith("fire_")
        ]
        print(f"Found {len(fire_dirs)} total 'fire_' directories in root_dir.")

        valid_dirs: List[str] = []
        for fire_dir in fire_dirs:
            files = os.listdir(fire_dir)
            pre_sentinel_files = [
                f for f in files
                if ("pre_sentinel" in f) and f.endswith(".tif") and ("_CM" not in f)
            ]
            gt_files = [
                f for f in files
                if ("GTSentinel" in f) and f.endswith(".tif")
            ]
            has_sentinel = len(pre_sentinel_files) > 0
            has_gt = len(gt_files) > 0

            if has_sentinel and has_gt:
                valid_dirs.append(fire_dir)
            else:
                print(f"Skipping {fire_dir}: missing pre_sentinel images or GTSentinel mask.")

        return valid_dirs

    # -----------------------------
    # Global statistics
    # -----------------------------

    def _compute_global_stats_mean_std(self) -> Dict[str, torch.Tensor]:
        """
        Computes global mean and std for:
        - Sentinel bands
        - Landsat bands
        - DEM
        - ERA5 raster and tabular components

        It iterates over the first available image per fire_dir and aggregates statistics.
        """
        print("Computing global mean and standard deviation for normalization...")

        sum_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.float64)
        sum_sq_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.float64)
        count_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.int64)

        sum_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float64)
        sum_sq_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float64)
        count_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.int64)

        sum_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float64)
        sum_sq_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float64)
        count_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.int64)

        sum_values_era5_tabular = torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float64)
        sum_sq_values_era5_tabular = torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float64)
        num_era5_samples_for_stats = 0  # number of ERA5 samples used for tabular stats

        sum_values_dem = torch.zeros(1, dtype=torch.float64)
        sum_sq_values_dem = torch.zeros(1, dtype=torch.float64)
        count_values_dem = torch.zeros(1, dtype=torch.int64)

        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)

            # DEM stats
            dem_paths_in_dir = [f for f in files if ("dem_10m" in f) and f.endswith(".tif")]
            if dem_paths_in_dir:
                dem_path_file = dem_paths_in_dir[0]
                path_dem = os.path.join(fire_dir, dem_path_file)
                try:
                    img_np_dem = self._read_image(path_dem)
                    img_np_dem = np.nan_to_num(img_np_dem, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_dem = np.clip(img_np_dem, 0.0, 10000.0).astype(np.float64)
                    band_data = img_np_dem[:, :, 0].flatten()
                    if band_data.size > 0:
                        sum_values_dem += np.sum(band_data)
                        sum_sq_values_dem += np.sum(band_data**2)
                        count_values_dem += len(band_data)
                    else:
                        print(f"[STATS] DEM band in {path_dem} has no valid data for stats.")
                except Exception as e:
                    print(f"[STATS ERROR] Error processing {path_dem} for DEM stats: {e}")

            # ERA5 stats
            era5_paths_in_dir = [f for f in files if ("era5" in f) and f.endswith(".tif")]
            if era5_paths_in_dir:
                era5_path_file = era5_paths_in_dir[0]
                path_era5 = os.path.join(fire_dir, era5_path_file)
                try:
                    img_np_era5 = self._read_image(path_era5)
                    img_np_era5 = np.nan_to_num(img_np_era5, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_era5 = np.clip(img_np_era5, 0.0, 10000.0).astype(np.float64)

                    if img_np_era5.shape[2] != N_BANDS_ERA5:
                        print(
                            f"[STATS] Skipping {path_era5} for ERA5 stats: "
                            f"expected {N_BANDS_ERA5} bands, got {img_np_era5.shape[2]}."
                        )
                    else:
                        # ERA5 raster bands (pixel-wise stats)
                        for i, original_idx in enumerate(ERA5_RASTER_BAND_INDICES):
                            band_data = img_np_era5[:, :, original_idx].flatten()
                            if band_data.size > 0:
                                sum_values_era5_raster[i] += np.sum(band_data)
                                sum_sq_values_era5_raster[i] += np.sum(band_data**2)
                                count_values_era5_raster[i] += len(band_data)
                        # ERA5 tabular bands (sample-wise mean)
                        for i, original_idx in enumerate(ERA5_TABULAR_BAND_INDICES):
                            band_spatial_mean = np.mean(img_np_era5[:, :, original_idx])
                            sum_values_era5_tabular[i] += band_spatial_mean
                            sum_sq_values_era5_tabular[i] += band_spatial_mean**2
                        num_era5_samples_for_stats += 1
                except Exception as e:
                    print(f"[STATS ERROR] Error processing {path_era5} for ERA5 stats: {e}")

            # Sentinel stats
            pre_sentinel_files_in_dir = sorted([
                f for f in files
                if ("pre_sentinel" in f) and f.endswith(".tif") and ("_CM" not in f)
            ])
            first_pre_sentinel_file = pre_sentinel_files_in_dir[0] if pre_sentinel_files_in_dir else None
            if first_pre_sentinel_file:
                path_sentinel = os.path.join(fire_dir, first_pre_sentinel_file)
                try:
                    img_np_sentinel = self._read_image(path_sentinel)
                    img_np_sentinel = np.nan_to_num(img_np_sentinel, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_sentinel = np.clip(img_np_sentinel, 0.0, 10000.0).astype(np.float64)

                    if img_np_sentinel.shape[2] != N_BANDS_SENTINEL:
                        print(
                            f"[STATS] Skipping {path_sentinel} for Sentinel stats: "
                            f"expected {N_BANDS_SENTINEL} bands, got {img_np_sentinel.shape[2]}."
                        )
                    else:
                        for band_idx in range(N_BANDS_SENTINEL):
                            band_data = img_np_sentinel[:, :, band_idx].flatten()
                            if band_data.size > 0:
                                sum_values_sentinel[band_idx] += np.sum(band_data)
                                sum_sq_values_sentinel[band_idx] += np.sum(band_data**2)
                                count_values_sentinel[band_idx] += len(band_data)
                except Exception as e:
                    print(f"[STATS ERROR] Error processing {path_sentinel} for Sentinel stats: {e}")

            # Landsat stats
            landsat_10m_files = sorted([
                f for f in files
                if ("pre_landsat" in f) and ("_10m.tif" in f) and ("_CM" not in f)
            ])
            first_pre_landsat_file = landsat_10m_files[0] if landsat_10m_files else None
            if first_pre_landsat_file:
                path_landsat = os.path.join(fire_dir, first_pre_landsat_file)
                try:
                    img_np_landsat = self._read_image(path_landsat)
                    img_np_landsat = np.nan_to_num(img_np_landsat, nan=0.0, posinf=0.0, neginf=0.0)

                    if img_np_landsat.shape[2] != N_BANDS_LANDSAT_ACTUAL:
                        print(
                            f"[STATS] Skipping {path_landsat} for Landsat stats: "
                            f"expected {N_BANDS_LANDSAT_ACTUAL} bands, got {img_np_landsat.shape[2]}."
                        )
                    else:
                        for band_idx in range(N_BANDS_LANDSAT_ACTUAL):
                            band_data = img_np_landsat[:, :, band_idx].flatten()
                            if band_data.size > 0:
                                sum_values_landsat[band_idx] += np.sum(band_data)
                                sum_sq_values_landsat[band_idx] += np.sum(band_data**2)
                                count_values_landsat[band_idx] += len(band_data)
                except Exception as e:
                    print(f"[STATS ERROR] Error processing {path_landsat} for Landsat stats: {e}")

        # Build stats dict with defaults
        stats = {
            "mean_sentinel": torch.zeros(N_BANDS_SENTINEL, dtype=torch.float32),
            "std_sentinel": torch.ones(N_BANDS_SENTINEL, dtype=torch.float32),
            "mean_landsat": torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float32),
            "std_landsat": torch.ones(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float32),
            "mean_dem": torch.zeros(1, dtype=torch.float32),
            "std_dem": torch.ones(1, dtype=torch.float32),
            "mean_era5_raster": torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float32),
            "std_era5_raster": torch.ones(N_BANDS_ERA5_RASTER, dtype=torch.float32),
            "mean_era5_tabular": torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float32),
            "std_era5_tabular": torch.ones(N_BANDS_ERA5_TABULAR, dtype=torch.float32),
        }

        # Sentinel stats
        for b in range(N_BANDS_SENTINEL):
            if count_values_sentinel[b] > 0:
                mean_val = sum_values_sentinel[b] / count_values_sentinel[b]
                var_val = (sum_sq_values_sentinel[b] / count_values_sentinel[b]) - mean_val**2
                var_val = torch.clamp(var_val, min=0.0)
                std_val = torch.sqrt(var_val)
                std_val = torch.clamp(std_val, min=1e-6)
                stats["mean_sentinel"][b] = mean_val.to(torch.float32)
                stats["std_sentinel"][b] = std_val.to(torch.float32)
            else:
                print(f"[STATS WARNING] No Sentinel data for band {b}; using (0,1).")

        # Landsat stats
        for b in range(N_BANDS_LANDSAT_ACTUAL):
            if count_values_landsat[b] > 0:
                mean_val = sum_values_landsat[b] / count_values_landsat[b]
                var_val = (sum_sq_values_landsat[b] / count_values_landsat[b]) - mean_val**2
                var_val = torch.clamp(var_val, min=0.0)
                std_val = torch.sqrt(var_val)
                std_val = torch.clamp(std_val, min=1e-6)
                stats["mean_landsat"][b] = mean_val.to(torch.float32)
                stats["std_landsat"][b] = std_val.to(torch.float32)
            else:
                print(f"[STATS WARNING] No Landsat data for band {b}; using (0,1).")

        # DEM stats
        if count_values_dem.item() > 0:
            mean_val = sum_values_dem / count_values_dem
            var_val = (sum_sq_values_dem / count_values_dem) - mean_val**2
            var_val = torch.clamp(var_val, min=0.0)
            std_val = torch.sqrt(var_val)
            std_val = torch.clamp(std_val, min=1e-6)
            stats["mean_dem"] = mean_val.to(torch.float32)
            stats["std_dem"] = std_val.to(torch.float32)
        else:
            print("[STATS WARNING] No DEM data; using (0,1).")

        # ERA5 raster stats
        for b in range(N_BANDS_ERA5_RASTER):
            if count_values_era5_raster[b] > 0:
                mean_val = sum_values_era5_raster[b] / count_values_era5_raster[b]
                var_val = (sum_sq_values_era5_raster[b] / count_values_era5_raster[b]) - mean_val**2
                var_val = torch.clamp(var_val, min=0.0)
                std_val = torch.sqrt(var_val)
                std_val = torch.clamp(std_val, min=1e-6)
                stats["mean_era5_raster"][b] = mean_val.to(torch.float32)
                stats["std_era5_raster"][b] = std_val.to(torch.float32)
            else:
                print(f"[STATS WARNING] No ERA5 raster data for band {b}; using (0,1).")

        # ERA5 tabular stats
        for b in range(N_BANDS_ERA5_TABULAR):
            if num_era5_samples_for_stats > 0:
                mean_val = sum_values_era5_tabular[b] / num_era5_samples_for_stats
                var_val = (sum_sq_values_era5_tabular[b] / num_era5_samples_for_stats) - mean_val**2
                var_val = torch.clamp(var_val, min=0.0)
                std_val = torch.sqrt(var_val)
                std_val = torch.clamp(std_val, min=1e-6)
                stats["mean_era5_tabular"][b] = mean_val.to(torch.float32)
                stats["std_era5_tabular"][b] = std_val.to(torch.float32)
            else:
                print(f"[STATS WARNING] No ERA5 tabular data for band {b}; using (0,1).")

        print("Global stats computation completed.")
        return stats

    # -----------------------------
    # Normalization helpers
    # -----------------------------

    def _normalize_bands_global(self, img_tensor: torch.Tensor, sensor_type: str = "sentinel") -> torch.Tensor:
        """
        Normalize spectral bands (Sentinel or Landsat) using global mean/std.
        """
        if self.global_stats is None:
            return img_tensor

        if sensor_type == "sentinel":
            mean = self.global_stats["mean_sentinel"].view(-1, 1, 1).to(img_tensor.device)
            std = self.global_stats["std_sentinel"].view(-1, 1, 1).to(img_tensor.device)
        elif sensor_type == "landsat":
            mean = self.global_stats["mean_landsat"].view(-1, 1, 1).to(img_tensor.device)
            std = self.global_stats["std_landsat"].view(-1, 1, 1).to(img_tensor.device)
        else:
            raise ValueError("sensor_type must be 'sentinel' or 'landsat'")

        std = torch.clamp(std, min=1e-6)
        return (img_tensor - mean) / std

    def _normalize_dem_global(self, dem_tensor: torch.Tensor) -> torch.Tensor:
        if self.global_stats is None:
            return dem_tensor
        mean_dem = self.global_stats["mean_dem"].view(1, 1, 1).to(dem_tensor.device)
        std_dem = self.global_stats["std_dem"].view(1, 1, 1).to(dem_tensor.device)
        std_dem = torch.clamp(std_dem, min=1e-6)
        return (dem_tensor - mean_dem) / std_dem

    def _normalize_era5_global(self, era5_tensor: torch.Tensor) -> torch.Tensor:
        if self.global_stats is None:
            return era5_tensor
        mean_era5 = self.global_stats["mean_era5_raster"].view(-1, 1, 1).to(era5_tensor.device)
        std_era5 = self.global_stats["std_era5_raster"].view(-1, 1, 1).to(era5_tensor.device)
        std_era5 = torch.clamp(std_era5, min=1e-6)
        return (era5_tensor - mean_era5) / std_era5

    def _normalize_era5_tabular(self, era5_tabular_tensor: torch.Tensor) -> torch.Tensor:
        if self.global_stats is None:
            return era5_tabular_tensor
        mean_tab = self.global_stats["mean_era5_tabular"].to(era5_tabular_tensor.device)
        std_tab = self.global_stats["std_era5_tabular"].to(era5_tabular_tensor.device)
        std_tab = torch.clamp(std_tab, min=1e-6)
        return (era5_tabular_tensor - mean_tab) / std_tab

    # -----------------------------
    # Binning by burned area
    # -----------------------------

    def _map_samples_to_bins(self) -> None:
        """
        Assign each sample index to a bin based on the number of burned pixels
        in its GTSentinel mask.
        """
        print("Mapping samples to bins based on burned pixel count...")
        for idx, fire_dir in enumerate(self.fire_dirs):
            files = os.listdir(fire_dir)
            gt_mask_path = None
            for f in files:
                if ("GTSentinel" in f) and f.endswith(".tif"):
                    gt_mask_path = os.path.join(fire_dir, f)
                    break
            if gt_mask_path is None:
                print(f"Warning: No GTSentinel mask found in {fire_dir}. Skipping sample {idx} for bin mapping.")
                continue

            gt_mask_np = self._read_mask(gt_mask_path)
            pixel_count = int(np.sum(gt_mask_np == 1))
            self.sample_pixel_counts[idx] = pixel_count

            if pixel_count < self.bin_thresholds[0]:
                self.sample_bins[idx] = 0
            elif pixel_count < self.bin_thresholds[1]:
                self.sample_bins[idx] = 1
            elif pixel_count < self.bin_thresholds[2]:
                self.sample_bins[idx] = 2
            else:
                self.sample_bins[idx] = 3

        print("Bin mapping completed. Distribution by bin:")
        for bin_idx in range(len(self.bin_thresholds) + 1):
            count_in_bin = sum(1 for v in self.sample_bins.values() if v == bin_idx)
            print(f"  Bin {bin_idx}: {count_in_bin} samples")

    # -----------------------------
    # Basic dataset protocol
    # -----------------------------

    def __len__(self) -> int:
        return len(self.fire_dirs)

    # -----------------------------
    # Low-level I/O helpers
    # -----------------------------

    def _read_image(self, path: str) -> np.ndarray:
        """
        Read a multi-band TIFF with rasterio and return (H, W, C) float32, clipped to [0, 10000].
        """
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        img = np.clip(img, 0.0, 10000.0)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        return img

    def _read_mask(self, path: str) -> np.ndarray:
        """
        Read a single-band mask TIFF and return (H, W) uint8 with values 0 or 1.
        """
        with rasterio.open(path) as src:
            mask = src.read(1)
        mask = np.nan_to_num(mask, nan=0).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        return mask

    def _read_streets(self, path: str) -> np.ndarray:
        """
        Read a single-band streets TIFF and return (H, W) float32.
        """
        with rasterio.open(path) as src:
            streets = src.read(1)
        streets = np.nan_to_num(streets, nan=0).astype(np.float32)
        return streets

    def _read_dem(self, path: str) -> np.ndarray:
        """
        Read a single-band DEM TIFF and return (H, W) float32.
        """
        with rasterio.open(path) as src:
            dem = src.read(1)
        dem = np.nan_to_num(dem, nan=0).astype(np.float32)
        return dem

    # -----------------------------
    # Sampling from bins
    # -----------------------------

    def get_random_sample_from_bin(self, target_bin: int) -> Dict[str, torch.Tensor]:
        """
        Return a random sample from the given bin index.
        """
        import random

        candidates = [idx for idx, b in self.sample_bins.items() if b == target_bin]
        if not candidates:
            raise ValueError(f"No samples found in bin {target_bin}.")

        random_idx = random.choice(candidates)
        sample = self[random_idx]
        return sample

    # -----------------------------
    # Core: __getitem__
    # -----------------------------

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sentinel-only baseline implementation.

        Still returns the full 7-tensor tuple expected by the training code:
        (sentinel, landsat, other_data, era5_raster, era5_tabular, landcover, gt_mask)

        - Sentinel image and GTSentinel mask are real.
        - Landsat, DEM, streets, ERA5, landcover, ignition are zero placeholders.
        - Global normalization is applied only to Sentinel.
        """
        fire_dir = self.fire_dirs[idx]
        files = os.listdir(fire_dir)

        # Keep compatibility with existing utils: allow either 'sentinel_path' or 'path'
        best_image_info = find_best_image_in_folder(fire_dir)
        if best_image_info is None:
            raise FileNotFoundError(
                f"No valid Sentinel image found in '{os.path.basename(fire_dir)}'. "
                f"Cannot load sample index {idx}."
            )

        if "sentinel_path" in best_image_info:
            sentinel_path = best_image_info["sentinel_path"]
        elif "path" in best_image_info:
            sentinel_path = best_image_info["path"]
        else:
            raise KeyError(
                "find_best_image_in_folder must return a dict with key 'sentinel_path' "
                "or 'path'."
            )

        pre_sentinel_np = self._read_image(sentinel_path)

        # Enforce target_size if needed (defensive; if patches are already the right size, this is a no-op)
        h, w, _ = pre_sentinel_np.shape
        if (h, w) != self.target_size:
            pre_sentinel_np = cv2.resize(
                pre_sentinel_np,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Zero placeholders for all non-Sentinel data
        H, W = self.target_size
        landsat_resampled_np = np.zeros((H, W, N_BANDS_LANDSAT_ACTUAL), dtype=np.float32)
        landsat_presence_flag = np.zeros((H, W, 1), dtype=np.float32)
        landsat_input_to_aug = np.concatenate([landsat_resampled_np, landsat_presence_flag], axis=2)

        streets_np = np.zeros((H, W, 1), dtype=np.float32)
        dem_np = np.zeros((H, W, 1), dtype=np.float32)
        ignition_np = np.zeros((H, W, 1), dtype=np.float32)
        era5_raster_np = np.zeros((H, W, N_BANDS_ERA5_RASTER), dtype=np.float32)
        landcover_np = np.zeros((H, W, 1), dtype=np.float32)

        # Ground truth mask
        gt_mask_path = None
        for f in files:
            if ("GTSentinel" in f) and f.endswith(".tif"):
                gt_mask_path = os.path.join(fire_dir, f)
                break
        if gt_mask_path is None:
            raise FileNotFoundError(
                f"No GTSentinel mask found in {fire_dir} for index {idx}. "
                "This should not happen after filtering."
            )
        gt_mask_np = self._read_mask(gt_mask_path)

        # Apply augmentations or simple ToTensor
        if self.apply_augmentations:
            transformed = self.augmentor(
                image=pre_sentinel_np,
                landsat_image=landsat_input_to_aug,
                streets_image=streets_np,
                dem_image=dem_np,
                ignition_pt=ignition_np,
                era5_image=era5_raster_np,
                landcover_np=landcover_np,
                mask=gt_mask_np,
            )
            sentinel_image_tensor = transformed["sentinel_image"]
            landsat_image_tensor = transformed["landsat_image"]
            streets_tensor = transformed["streets_image"]
            dem_tensor = transformed["dem_image"]
            era5_tensor = transformed["era5_image"]
            ignition_tensor = transformed["ignition_pt"]
            gt_mask_tensor = transformed["mask"]
            landcover_tensor = transformed["landcover_np"]
        else:
            ts_sentinel = self.eval_transform(image=pre_sentinel_np, mask=gt_mask_np)
            sentinel_image_tensor = ts_sentinel["image"]
            gt_mask_tensor = ts_sentinel["mask"]

            landsat_image_tensor = self.eval_transform(image=landsat_input_to_aug)["image"]
            streets_tensor = self.eval_transform(image=streets_np)["image"]
            dem_tensor = self.eval_transform(image=dem_np)["image"]
            era5_tensor = self.eval_transform(image=era5_raster_np)["image"]
            ignition_tensor = self.eval_transform(image=ignition_np)["image"]
            landcover_tensor = self.eval_transform(image=landcover_np)["image"]

        # Normalize Sentinel only (baseline)
        if self.global_stats:
            sentinel_image_tensor = self._normalize_bands_global(
                sentinel_image_tensor, sensor_type="sentinel"
            )

        # Other data concatenation: DEM + streets
        other_data_tensor = torch.concat([dem_tensor, streets_tensor], dim=0)

        # ERA5 tabular is zero for now (Sentinel-only baseline)
        era5_tabular = torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float32)

        return (
            sentinel_image_tensor,
            landsat_image_tensor,
            other_data_tensor,
            era5_tensor,
            era5_tabular,
            landcover_tensor,
            gt_mask_tensor,
        )

    # -----------------------------
    # Info helper
    # -----------------------------

    def get_sample_info(self, idx: int) -> dict:
        """
        Returns a dictionary describing the sample at index idx:
        fire_id, directory, Sentinel/Landsat filenames, GT mask file, fire date, sample_type.
        """
        fire_dir = self.fire_dirs[idx]
        dir_name = os.path.basename(fire_dir)
        match = re.search(r"fire_(\d+)", dir_name)
        fire_id = int(match.group(1)) if match else None

        best_image_info = find_best_image_in_folder(fire_dir)
        if best_image_info is not None:
            if "sentinel_path" in best_image_info:
                pre_image_file_sentinel = os.path.basename(best_image_info["sentinel_path"])
            elif "path" in best_image_info:
                pre_image_file_sentinel = os.path.basename(best_image_info["path"])
            else:
                pre_image_file_sentinel = "N/A"
        else:
            pre_image_file_sentinel = "N/A"

        pre_image_file_landsat = "N/A"
        for f in os.listdir(fire_dir):
            if ("pre_landsat" in f) and ("_10m.tif" in f) and ("_CM" not in f):
                pre_image_file_landsat = f
                break

        gt_mask_file = "N/A"
        for f in os.listdir(fire_dir):
            if ("GTSentinel" in f) and f.endswith(".tif"):
                gt_mask_file = f
                break

        return {
            "fire_id": fire_id,
            "directory": fire_dir,
            "pre_image_file_sentinel": pre_image_file_sentinel,
            "pre_image_file_landsat": pre_image_file_landsat,
            "gt_mask_file": gt_mask_file,
            "fire_date": self.fire_id_to_date_map.get(fire_id, "N/A"),
            "sample_type": self.sample_type_map.get(fire_dir, "unknown"),
        }