import os
import sys
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
from datetime import datetime
import re
import time
# We no longer use distance_transform_edt; we use dilation to create a localized area instead
from scipy.ndimage import generate_binary_structure, binary_dilation

# IMPORTANT: Make sure this import is correct for your project
from modelWithLandsat.utils import find_best_image_in_folder


# --- Utility Functions ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Extracts the date from a Sentinel filename specific to fires.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'fire_\d+_(\d{4}-\d{2}-\d{2})_pre_sentinel_\d+\.tif', filename)
    if match:
        datetime_str = match.group(1)
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d')
        except ValueError:
            return None
    return None


def get_image_spatial_info(image_path: str):
    """
    Extracts the centroid (lon, lat in WGS84) from a raster image (e.g., Sentinel).
    """
    try:
        with rasterio.open(image_path) as src:
            src_bounds = src.bounds
            src_crs = src.crs

            centroid_x_src = (src_bounds.left + src_bounds.right) / 2
            centroid_y_src = (src_bounds.bottom + src_bounds.top) / 2

            gdf_centroid_src = gpd.GeoDataFrame(
                geometry=[gpd.points_from_xy([centroid_x_src], [centroid_y_src])[0]],
                crs=src_crs
            )

            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326)
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]

            print(
                f"   Spatial info (for logs): Image centroid (WGS84) "
                f"({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f})"
            )

            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,
                centroid_lon_wgs84 - 0.5,
                centroid_lat_wgs84 - 0.5,
                centroid_lon_wgs84 + 0.5
            ]

            return centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds
    except Exception as e:
        print(f"❌ Error extracting spatial info from '{os.path.basename(image_path)}': {e}")
        return None, None, None


def log_processed_id(log_file: str, fire_id: str):
    """
    Records a processed fire ID into a log file.
    """
    with open(log_file, "a") as f:
        f.write(f"{fire_id}\n")


def get_processed_ids(log_file: str) -> set:
    """
    Reads already-processed fire IDs from the log file.
    """
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


# --- Function to Generate Ignition Point Raster (NEW VERSION) ---
def generate_ignition_point_raster(
    fire_id: str,
    geojson_gdf: gpd.GeoDataFrame,
    config: dict,
    spread_radius: int = 3
) -> bool:
    """
    Creates a 256x256 TIFF containing an ignition point and a small surrounding area.
    Value 1.0 corresponds to the ignition area, and 0.0 to background pixels.

    Args:
        fire_id (str): Fire ID.
        geojson_gdf (gpd.GeoDataFrame): GeoDataFrame containing ignition points.
        config (dict): Configuration dictionary.
        spread_radius (int): Size of the area around the ignition point.
            A value of 1 creates a 3x3 pixel area.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    ignition_output_filename = f"fire_{fire_id}_ignition_pt.tif"
    out_path_ignition_tif = os.path.join(FIRE_SAVE_FOLDER, ignition_output_filename)

    # Check if the file already exists to avoid unnecessary regeneration
    if os.path.exists(out_path_ignition_tif):
        print(f"✅ Ignition point raster (red dot) already present for fire ID: {fire_id}. Skipping.")
        return True

    print(f"\n--- Starting ignition point raster (red dot) generation for fire ID: {fire_id} ---")

    # 1. Find the existing pre-fire Sentinel image.
    best_image_info = find_best_image_in_folder(FIRE_SAVE_FOLDER)
    if best_image_info is None or "sentinel_path" not in best_image_info:
        print(
            f"❌ No valid Sentinel image found in '{FIRE_SAVE_FOLDER}'. "
            f"Cannot align the ignition map."
        )
        return False

    pre_sentinel_image_path = best_image_info["sentinel_path"]
    print(f"   Found Sentinel image for alignment: {os.path.basename(pre_sentinel_image_path)}")

    # 2. Get spatial profile directly from Sentinel image
    try:
        with rasterio.open(pre_sentinel_image_path) as src_sentinel:
            output_transform = src_sentinel.transform
            output_crs = src_sentinel.crs
            patch_size_pixels_h = src_sentinel.height
            patch_size_pixels_w = src_sentinel.width

            if (patch_size_pixels_h != config["patch_size_pixels"] or
                    patch_size_pixels_w != config["patch_size_pixels"]):
                print(
                    f"⚠️ Warning: Sentinel image size ({patch_size_pixels_h}x{patch_size_pixels_w}) "
                    f"does not match config['patch_size_pixels'] "
                    f"({config['patch_size_pixels']}x{config['patch_size_pixels']}). "
                    f"Using Sentinel dimensions."
                )

            print(
                f"   Output profile derived from Sentinel: "
                f"Dims {patch_size_pixels_h}x{patch_size_pixels_w}, CRS: {output_crs}"
            )
    except Exception as e:
        print(f"❌ Error opening Sentinel for spatial info '{os.path.basename(pre_sentinel_image_path)}': {e}")
        return False

    # 3. Extract ignition point from GeoJSON
    try:
        fire_feature = geojson_gdf[geojson_gdf["id"] == int(fire_id)]

        if fire_feature.empty:
            print(f"❌ Fire ID {fire_id} not found in GeoJSON. Cannot create ignition point raster.")
            return False

        ignition_x_geojson = fire_feature["point_x"].iloc[0]
        ignition_y_geojson = fire_feature["point_y"].iloc[0]

        gdf_ignition_geojson_crs = gpd.GeoDataFrame(
            geometry=[Point(ignition_x_geojson, ignition_y_geojson)],
            crs="EPSG:3857"
        )

        gdf_ignition_target_crs = gdf_ignition_geojson_crs.to_crs(output_crs)
        ignition_x_target_crs = gdf_ignition_target_crs.geometry.x.iloc[0]
        ignition_y_target_crs = gdf_ignition_target_crs.geometry.y.iloc[0]

        row, col = rasterio.transform.rowcol(output_transform, ignition_x_target_crs, ignition_y_target_crs)

        # --- Create a binary image with a localized area around the ignition point ---
        ignition_raster_final = np.zeros((patch_size_pixels_h, patch_size_pixels_w), dtype=np.float32)

        if 0 <= row < patch_size_pixels_h and 0 <= col < patch_size_pixels_w:
            # Create dilation kernel with a cross shape
            kernel = generate_binary_structure(2, 1)

            # Temporary raster with only the ignition pixel set to 1
            temp_ignition_pixel = np.zeros_like(ignition_raster_final, dtype=np.uint8)
            temp_ignition_pixel[row, col] = 1

            # Apply dilation to create a wider area
            dilated_area = binary_dilation(
                temp_ignition_pixel,
                structure=kernel,
                iterations=spread_radius
            )
            ignition_raster_final[dilated_area] = 1.0

            print(f"   Ignition point + neighborhood (radius {spread_radius}) placed at pixel ({row}, {col}).")
        else:
            print(
                f"⚠️ Ignition point for fire ID {fire_id} is OUT OF BOUNDS of the raster. "
                f"The ignition raster will be all zeros."
            )

        # 4. Define output TIFF profile
        output_profile = {
            "height": patch_size_pixels_h,
            "width": patch_size_pixels_w,
            "count": 1,
            "dtype": ignition_raster_final.dtype,
            "crs": output_crs,
            "transform": output_transform,
            "nodata": 0.0,
            "driver": "GTiff"
        }

        # 5. Save ignition point raster
        with rasterio.open(out_path_ignition_tif, "w", **output_profile) as dst:
            dst.write(ignition_raster_final, 1)

        print(f"🎉 Ignition point (red dot) TIFF saved: {os.path.basename(out_path_ignition_tif)}")
        return True

    except Exception as e:
        print(f"❌ Error while creating ignition point raster for fire ID {fire_id}: {e}")
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    # General dataset/process configuration
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

    print(f"Loading GeoJSON from: {config['geojson_path']}")
    try:
        main_geojson_gdf = gpd.read_file(config["geojson_path"])
        if "id" not in main_geojson_gdf.columns:
            print("ERROR: Column 'id' not found in the GeoJSON. Required to match 'fire_XXX' folders.")
            exit("Cannot proceed.")

        if "point_x" not in main_geojson_gdf.columns or "point_y" not in main_geojson_gdf.columns:
            print("ERROR: Columns 'point_x' and/or 'point_y' not found in the GeoJSON.")
            print("Ensure the GeoJSON contains these columns for the ignition point.")
            exit("Cannot proceed.")

    except Exception as e:
        print(f"ERROR: Unable to load GeoJSON from '{config['geojson_path']}': {e}")
        exit("Cannot proceed.")
    print("GeoJSON loaded successfully.")

    # Log file handling for already processed IDs
    log_file_path = os.path.join(config["root_dataset_folder"], "processed_fire_ids_ignition_points.log")
    processed_ids = get_processed_ids(log_file_path)

    print(f"Found {len(os.listdir(config['root_dataset_folder']))} fire folders in: {config['root_dataset_folder']}")
    print(f"Fires with ignition points already processed (from log): {len(processed_ids)}")

    all_fire_ids = [
        item.replace("fire_", "")
        for item in sorted(os.listdir(config["root_dataset_folder"]))
        if os.path.isdir(os.path.join(config["root_dataset_folder"], item)) and item.startswith("fire_")
    ]

    processed_count_current_run = 0
    for fire_id_str in all_fire_ids:
        if (fire_id_str in processed_ids and
                os.path.exists(os.path.join(
                    config["root_dataset_folder"],
                    f"fire_{fire_id_str}",
                    f"fire_{fire_id_str}_ignition_pt.tif"
                ))):
            print(f"ℹ️ Fire ID: {fire_id_str} already processed and output file exists. Skipping.")
            continue

        success = generate_ignition_point_raster(fire_id_str, main_geojson_gdf, config)
        if success:
            processed_count_current_run += 1
            log_processed_id(log_file_path, fire_id_str)
        else:
            print(f"Ignition point process failed for fire ID {fire_id_str}. Not added to the log.")

    print(f"\nProcessing completed. Generated {processed_count_current_run} new ignition point rasters.")
    print(f"Successfully processed fire IDs were recorded in: {log_file_path}")
    print("\nScript finished.")
