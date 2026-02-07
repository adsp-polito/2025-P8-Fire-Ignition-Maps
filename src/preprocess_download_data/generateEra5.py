import os
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box, shape
from datetime import datetime, timedelta
import re
import time
import sys  # Import sys to access command-line arguments

# IMPORTANT: Make sure this import is correct for your project
from modelWithLandsat.utils import find_best_image_in_folder

# --- Utility Functions (unchanged) ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Extracts the acquisition date from a Sentinel filename with the format:
    'fire_xxxx_YYYY-MM-DD_pre_sentinel_N.tif' (where xxxx is the fire ID).

    Args:
        filepath (str): Full path or filename of the Sentinel file.

    Returns:
        datetime.datetime: Parsed datetime object (date only), or None if the format does not match.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'fire_\d+_(\d{4}-\d{2}-\d{2})_pre_sentinel_\d+\.tif', filename)

    if match:
        datetime_str = match.group(1)
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d')
        except ValueError:
            print(f"WARNING: Unable to parse date '{datetime_str}' from file: {filename}")
            return None
    else:
        print(f"WARNING: No date pattern found in filename: {filename}")
        return None


def get_image_spatial_info(image_path: str, target_crs: str):
    """
    Extracts centroid and bounding box of a TIFF image in the target CRS.

    Args:
        image_path (str): Path to the TIFF file (e.g., Sentinel/Landsat).
        target_crs (str): Target CRS (e.g., "EPSG:32632").

    Returns:
        tuple: (centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds) in WGS84 for CDS search.
               Returns (None, None, None) on error.
    """
    try:
        with rasterio.open(image_path) as src:
            src_bounds = src.bounds
            src_crs = src.crs

            # Compute centroid in source CRS
            centroid_x_src = (src_bounds.left + src_bounds.right) / 2
            centroid_y_src = (src_bounds.bottom + src_bounds.top) / 2

            # Build a GeoDataFrame with the centroid in source CRS
            gdf_centroid_src = gpd.GeoDataFrame(
                geometry=[gpd.points_from_xy([centroid_x_src], [centroid_y_src])[0]],
                crs=src_crs
            )

            # Convert centroid to WGS84 (EPSG:4326)
            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326)
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]

            # Compute a WGS84 bounding box based on centroid and a fixed size.
            # This is for the CDS API request. ERA5 has ~0.1° resolution (~10 km).
            # A +/- 0.5° box (~50 km) around the centroid should comfortably cover
            # the 2.56 km x 2.56 km patch and provide enough data for interpolation.
            # CDS API order is [north, west, south, east]
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,  # north
                centroid_lon_wgs84 - 0.5,  # west
                centroid_lat_wgs84 - 0.5,  # south
                centroid_lon_wgs84 + 0.5   # east
            ]

            print(
                f"  Spatial info: Centroid (WGS84) "
                f"({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f}), "
                f"CDS BBox {bbox_wgs84_for_cds}"
            )

            return centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds
    except Exception as e:
        print(f"❌ Error extracting spatial info from '{os.path.basename(image_path)}': {e}")
        return None, None, None


def download_and_process_era5_land(
    fire_id: str,
    image_datetime: datetime,
    centroid_lon_wgs84: float,
    centroid_lat_wgs84: float,
    bbox_cds: list,
    config: dict
) -> bool:
    """
    Downloads ERA5-Land 'derived-era5-land-daily-statistics' data from the CDS API
    for the specified date and area, then clips and resamples it to the desired patch size
    (256x256 at 10 m) and saves a single multi-band TIFF.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    TARGET_RESOLUTION_M = config.get("target_resolution_m", 10)  # Target resolution for ERA5 data (10 m)
    patch_size_pixels = config.get("patch_size_pixels", 256)
    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632")

    os.makedirs(FIRE_SAVE_FOLDER, exist_ok=True)

    client = cdsapi.Client()
    dataset = "derived-era5-land-daily-statistics"  # Daily statistics dataset

    era5_variables_config = [
        {"cds_name": "2m_temperature", "xr_name": "t2m"},
        {"cds_name": "10m_u_component_of_wind", "xr_name": "u10"},
        {"cds_name": "10m_v_component_of_wind", "xr_name": "v10"}
    ]

    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M  # 256 * 10 = 2560 meters

    gdf_centroid_wgs84 = gpd.GeoDataFrame(
        geometry=[gpd.points_from_xy([centroid_lon_wgs84], [centroid_lat_wgs84])[0]],
        crs="EPSG:4326"
    )
    gdf_centroid_target_crs = gdf_centroid_wgs84.to_crs(TARGET_CRS_FOR_FIRES)
    centroid_x_target_crs = gdf_centroid_target_crs.geometry.x.iloc[0]
    centroid_y_target_crs = gdf_centroid_target_crs.geometry.y.iloc[0]

    final_left = centroid_x_target_crs - (fixed_patch_size_meters / 2)
    final_right = centroid_x_target_crs + (fixed_patch_size_meters / 2)
    final_bottom = centroid_y_target_crs - (fixed_patch_size_meters / 2)
    final_top = centroid_y_target_crs + (fixed_patch_size_meters / 2)

    final_transform = rasterio.transform.from_bounds(
        final_left, final_bottom, final_right, final_top,
        width=patch_size_pixels,
        height=patch_size_pixels
    )
    final_output_crs = TARGET_CRS_FOR_FIRES

    reprojected_data_bands = []
    band_names = []

    success_all_vars = True
    MAX_RETRIES = 5
    RETRY_DELAY_SEC = 10

    for var_info in era5_variables_config:
        var_cds_name = var_info["cds_name"]
        var_xr_name = var_info["xr_name"]

        temp_nc_file = os.path.join(
            FIRE_SAVE_FOLDER,
            f"temp_era5_land_{var_cds_name.replace(' ', '_')}_{image_datetime.strftime('%Y%m%d')}.nc"
        )

        request_params = {
            "variable": var_cds_name,
            "year": str(image_datetime.year),
            "month": f"{image_datetime.month:02d}",
            "day": f"{image_datetime.day:02d}",
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "3_hourly",
            "area": bbox_cds,
            "format": "netcdf",
        }

        download_success = False
        for attempt in range(MAX_RETRIES):
            try:
                print(f"  Downloading '{var_cds_name}' (attempt {attempt + 1}/{MAX_RETRIES})...")
                client.retrieve(dataset, request_params, temp_nc_file)
                download_success = True
                print(f"  Download '{var_cds_name}' completed.")
                break
            except Exception as e:
                print(f"  ❌ Download error for '{var_cds_name}': {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying in {RETRY_DELAY_SEC} seconds...")
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    print(f"  ❌ Failed to download '{var_cds_name}' after {MAX_RETRIES} attempts.")
                    success_all_vars = False
                    break

        if not download_success:
            continue

        try:
            with xr.open_dataset(temp_nc_file, engine="netcdf4") as ds_era5:
                era5_data_array_slice = ds_era5[var_xr_name].isel(valid_time=0).squeeze()

                if len(era5_data_array_slice.dims) != 2:
                    print(f"  ❌ Variable '{var_xr_name}' is not 2D ({era5_data_array_slice.dims}). Skipping.")
                    success_all_vars = False
                    os.remove(temp_nc_file)
                    continue

                era5_np_array = era5_data_array_slice.values

                min_lon_src = era5_data_array_slice.longitude.min().item()
                max_lon_src = era5_data_array_slice.longitude.max().item()
                min_lat_src = era5_data_array_slice.latitude.min().item()
                max_lat_src = era5_data_array_slice.latitude.max().item()

                src_transform_era5 = rasterio.transform.from_bounds(
                    min_lon_src, min_lat_src, max_lon_src, max_lat_src,
                    era5_np_array.shape[1], era5_np_array.shape[0]
                )
                src_crs_era5 = "EPSG:4326"

                era5_data_reprojected = np.zeros((patch_size_pixels, patch_size_pixels), dtype=np.float32)
                era5_fill_value = np.nan

                reproject(
                    source=era5_np_array,
                    destination=era5_data_reprojected,
                    src_transform=src_transform_era5,
                    src_crs=src_crs_era5,
                    dst_transform=final_transform,
                    dst_crs=final_output_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=era5_fill_value,
                    num_threads=os.cpu_count()
                )

                reprojected_data_bands.append(era5_data_reprojected)
                band_names.append(var_cds_name)
                print(f"  Processing '{var_cds_name}' completed.")

        except Exception as e:
            print(f"  ❌ Processing error for '{var_cds_name}' (fire ID {fire_id}): {e}")
            success_all_vars = False
        finally:
            if os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                print(f"  Temporary file '{os.path.basename(temp_nc_file)}' removed.")

    if success_all_vars and reprojected_data_bands:
        era5_output_filename = f"fire_{fire_id}_era5_multi_band_{image_datetime.strftime('%Y%m%d')}.tif"
        out_path_era5_tif = os.path.join(FIRE_SAVE_FOLDER, era5_output_filename)

        output_profile = {
            "height": patch_size_pixels,
            "width": patch_size_pixels,
            "count": len(reprojected_data_bands),
            "dtype": reprojected_data_bands[0].dtype,
            "crs": final_output_crs,
            "transform": final_transform,
            "nodata": -9999.0,
            "driver": "GTiff"
        }

        try:
            with rasterio.open(out_path_era5_tif, "w", **output_profile) as dst:
                for i, band_data in enumerate(reprojected_data_bands):
                    dst.write(band_data, i + 1)

                for i, name in enumerate(band_names):
                    dst.set_band_description(i + 1, name)

            print(f"🎉 ERA5 multi-band TIFF saved: {os.path.basename(out_path_era5_tif)}")
            print(f"  Bands included: {', '.join(band_names)}")
        except Exception as e:
            print(f"❌ Error writing multi-band TIFF: {e}")
            success_all_vars = False
    elif not reprojected_data_bands:
        print("⚠️ No reprojected ERA5 data available for multi-band TIFF.")
        success_all_vars = False

    return success_all_vars


# --- Processed ID logging functions (unchanged) ---
def log_processed_id(log_file: str, fire_id: str):
    with open(log_file, "a") as f:
        f.write(f"{fire_id}\n")


def get_processed_ids(log_file: str) -> set:
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


# --- Main wrapper function (unchanged) ---
def generate_era5_for_specific_fire(single_fire_id: str, config: dict):
    """
    Wrapper function to start the ERA5 generation process for a single fire.

    Args:
        single_fire_id (str): Fire ID to process.
        config (dict): Configuration dictionary.
    """
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")
    era5_output_filename_check = f"fire_{single_fire_id}_era5_multi_band_"  # Prefix for ERA5 existence check

    if not os.path.isdir(fire_folder_path):
        print(f"ℹ️ Folder '{fire_folder_path}' not found for ID: {single_fire_id}. Skipping.")
        return False

    era5_file_exists = any(
        f.startswith(era5_output_filename_check) and f.endswith(".tif")
        for f in os.listdir(fire_folder_path)
    )

    if era5_file_exists:
        print(f"✅ ERA5 raster already present for fire ID: {single_fire_id}. Skipping.")
        return True  # Consider already processed successfully

    print(f"\n--- Starting ERA5 processing for fire ID: {single_fire_id} ---")
    print(f"  Looking for Sentinel in: {fire_folder_path}")

    best_image_info = find_best_image_in_folder(fire_folder_path)
    if best_image_info is None or "sentinel_path" not in best_image_info:
        print(f"❌ No valid Sentinel image found for ID: {single_fire_id}. Cannot query ERA5.")
        return False

    pre_sentinel_image_path = best_image_info["sentinel_path"]
    image_datetime = parse_datetime_from_sentinel_filename(pre_sentinel_image_path)

    if not image_datetime:
        print(
            f"❌ Unable to determine Sentinel image date from "
            f"'{os.path.basename(pre_sentinel_image_path)}'."
        )
        return False

    print(
        f"  Sentinel image date: {image_datetime.isoformat()} from: "
        f"{os.path.basename(pre_sentinel_image_path)}"
    )

    centroid_lon_wgs84, centroid_lat_wgs84, bbox_cds = get_image_spatial_info(
        pre_sentinel_image_path, config["target_crs"]
    )
    if centroid_lon_wgs84 is None:
        print(f"❌ Unable to get spatial info from Sentinel for ID: {single_fire_id}.")
        return False

    success = download_and_process_era5_land(
        fire_id=single_fire_id,
        image_datetime=image_datetime,
        centroid_lon_wgs84=centroid_lon_wgs84,
        centroid_lat_wgs84=centroid_lat_wgs84,
        bbox_cds=bbox_cds,
        config=config
    )

    if success:
        print(f"✅ ERA5 process completed for fire ID: {single_fire_id}.")
    else:
        print(f"❌ ERA5 process failed for fire ID: {single_fire_id}. Check logs.")
    print(f"--- End processing for fire ID: {single_fire_id} ---\n")
    return success


# --- Main execution block (MODIFIED for array jobs) ---
if __name__ == "__main__":
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson",
        "root_dataset_folder": "piedmont_new",
        "target_crs": "EPSG:32632",
        "patch_size_pixels": 256,
        "target_resolution_m": 10,
    }

    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERROR: Root dataset folder '{config['root_dataset_folder']}' does not exist.")
        print(
            "Ensure that 'root_dataset_folder' in the configuration points to the directory "
            "containing the fire folders (e.g., 'piedmont_new')."
        )
        exit("Cannot proceed.")

    log_file_path = os.path.join(config["root_dataset_folder"], "processed_fire_ids_era5.log")
    processed_ids = get_processed_ids(log_file_path)

    # Collect fire_XXX folder IDs
    all_fire_ids = []
    for item in sorted(os.listdir(config["root_dataset_folder"])):
        full_path = os.path.join(config["root_dataset_folder"], item)
        if os.path.isdir(full_path) and item.startswith("fire_"):
            fire_id_str = item.replace("fire_", "")
            all_fire_ids.append(fire_id_str)

    print(f"Found {len(all_fire_ids)} fire folders in: {config['root_dataset_folder']}")
    print(f"Already processed fires (from log): {processed_ids}")

    # Array job argument handling
    task_id = 0
    num_tasks = 1
    # sys.argv[0] is the script name
    if len(sys.argv) > 2:  # at least two args: task_id and num_tasks
        try:
            task_id = int(sys.argv[1])
            num_tasks = int(sys.argv[2])
            print(f"Running as an array task: Task ID {task_id} out of {num_tasks} total.")
        except ValueError:
            print("WARNING: Invalid SLURM_ARRAY_TASK_ID/COUNT arguments. Running sequentially.")
    else:
        print("Running sequentially (not as a Slurm array job).")

    # Filter fire IDs to process for this specific array task
    fire_ids_for_this_task = []
    for i, fire_id_str in enumerate(all_fire_ids):
        # Modulo logic distributes IDs round-robin across tasks
        if i % num_tasks == task_id:
            fire_ids_for_this_task.append(fire_id_str)

    print(f"This task ({task_id}/{num_tasks}) will process {len(fire_ids_for_this_task)} fires.")

    processed_count_current_run = 0
    for fire_id_str in fire_ids_for_this_task:
        # This is critical to avoid reprocessing work already completed
        # either by this job in a previous run or by other jobs in the current run
        if fire_id_str in processed_ids:
            print(f"ℹ️ Fire ID: {fire_id_str} already in the log or ERA5 file already exists. Skipping.")
            continue

        success = generate_era5_for_specific_fire(fire_id_str, config)
        if success:
            processed_count_current_run += 1
            # Log the ID only if processing succeeded
            log_processed_id(log_file_path, fire_id_str)
        else:
            print(f"ERA5 process for fire ID {fire_id_str} failed. Not added to the log.")

    print(f"\nProcessing completed for task {task_id}. Processed {processed_count_current_run} new fires.")
    print(f"Successfully processed fire IDs were recorded in: {log_file_path}")
    print("\nScript finished.")
