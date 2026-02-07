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
from scipy.ndimage import distance_transform_edt  # Import for Euclidean distance
# import cv2  # Not used in this script; remove or keep if used elsewhere

# IMPORTANT: Make sure this import is correct for your project
# This function is required to find the Sentinel image used to derive georeferencing
# Ensure the 'modelWithLandsat.utils' path matches your project structure
from modelWithLandsat.utils import find_best_image_in_folder

# --- Utility Functions ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Extracts the date from a Sentinel filename specific to fires.
    """
    filename = os.path.basename(filepath)
    # Regex to capture the date in YYYY-MM-DD format
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
    This function is used for debugging/logging only and is no longer used to define
    the output patch bounds for the ignition point.
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

            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326)  # Always convert to WGS84
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]

            print(
                f"   Spatial info (for logs): Image centroid (WGS84) "
                f"({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f})"
            )

            # bbox_wgs84_for_cds is not strictly needed for this script,
            # but kept for signature compatibility
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,   # north
                centroid_lon_wgs84 - 0.5,   # west
                centroid_lat_wgs84 - 0.5,   # south
                centroid_lon_wgs84 + 0.5    # east
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


# --- Function to Generate Ignition Point Raster (MODIFIED) ---
def generate_ignition_point_raster(fire_id: str, geojson_gdf: gpd.GeoDataFrame, config: dict) -> bool:
    """
    Creates a 256x256 TIFF containing the Euclidean distance map from the ignition point,
    spatially aligned with the existing pre-fire Sentinel image.
    Value 1.0 corresponds to the ignition point, and 0.0 to the farthest pixels.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    ignition_output_filename = f"fire_{fire_id}_ignition_map.tif"
    out_path_ignition_tif = os.path.join(FIRE_SAVE_FOLDER, ignition_output_filename)

    # Check if the file already exists to avoid unnecessary regeneration
    # This is essential for future runs, but not for the first run after manual deletion
    if os.path.exists(out_path_ignition_tif):
        print(f"✅ Ignition point raster (distance map) already present for fire ID: {fire_id}. Skipping.")
        return True

    print(f"\n--- Starting ignition point raster (distance map) generation for fire ID: {fire_id} ---")

    # 1. Find the existing pre-fire Sentinel image.
    # We use its spatial profile (CRS, transform, dimensions) to guarantee alignment.
    best_image_info = find_best_image_in_folder(FIRE_SAVE_FOLDER)
    if best_image_info is None or "sentinel_path" not in best_image_info:
        print(
            f"❌ No valid Sentinel image found in '{FIRE_SAVE_FOLDER}'. "
            f"Cannot align the ignition map."
        )
        return False

    pre_sentinel_image_path = best_image_info["sentinel_path"]
    print(f"   Found Sentinel image for alignment: {os.path.basename(pre_sentinel_image_path)}")

    # 2. Read spatial profile directly from the Sentinel image
    try:
        with rasterio.open(pre_sentinel_image_path) as src_sentinel:
            # These will be the spatial parameters for our output image.
            # Output matches Sentinel size, CRS, and transform.
            output_transform = src_sentinel.transform
            output_crs = src_sentinel.crs
            patch_size_pixels_h = src_sentinel.height
            patch_size_pixels_w = src_sentinel.width

            # Consistency check vs config size
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

    # 3. Extract the ignition point from the GeoJSON
    try:
        # Find the matching fire row in the GeoJSON using its ID
        fire_feature = geojson_gdf[geojson_gdf["id"] == int(fire_id)]

        if fire_feature.empty:
            print(f"❌ Fire ID {fire_id} not found in GeoJSON. Cannot create ignition point raster.")
            return False

        # Extract 'point_x' and 'point_y' coordinates from GeoJSON
        ignition_x_geojson = fire_feature["point_x"].iloc[0]
        ignition_y_geojson = fire_feature["point_y"].iloc[0]

        # Create a GeoDataFrame for the ignition point using the GeoJSON CRS (EPSG:3857)
        gdf_ignition_geojson_crs = gpd.GeoDataFrame(
            geometry=[Point(ignition_x_geojson, ignition_y_geojson)],
            crs="EPSG:3857"  # Explicit GeoJSON CRS
        )

        # Convert ignition point to the target CRS (same as Sentinel/output)
        gdf_ignition_target_crs = gdf_ignition_geojson_crs.to_crs(output_crs)
        ignition_x_target_crs = gdf_ignition_target_crs.geometry.x.iloc[0]
        ignition_y_target_crs = gdf_ignition_target_crs.geometry.y.iloc[0]

        print(
            f"   Ignition point (converted to Sentinel CRS): "
            f"({ignition_x_target_crs:.2f}, {ignition_y_target_crs:.2f})"
        )

        # Convert ignition point coordinates to pixel indices (row, col)
        # within the 256x256 frame defined by the Sentinel transform.
        row, col = rasterio.transform.rowcol(output_transform, ignition_x_target_crs, ignition_y_target_crs)

        # --- Create a temporary binary image with a single pixel set to 1 ---
        # This is the input for the distance transform.
        single_pixel_ignition_raster = np.zeros((patch_size_pixels_h, patch_size_pixels_w), dtype=np.uint8)

        # If ignition point lies inside the patch, set that pixel to 1
        if 0 <= row < patch_size_pixels_h and 0 <= col < patch_size_pixels_w:
            single_pixel_ignition_raster[row, col] = 1
            print(f"   Ignition point placed at pixel ({row}, {col}) within the aligned patch.")
        else:
            print(
                f"⚠️ Ignition point for fire ID {fire_id} "
                f"(x:{ignition_x_target_crs:.2f}, y:{ignition_y_target_crs:.2f}) "
                f"IS OUT OF BOUNDS of the Sentinel-derived raster. "
                f"The ignition raster will be all zeros."
            )
            # Important case: if ignition point is outside the 256x256,
            # the distance map will be all zeros, which is acceptable.

        # --- Compute the Distance Transform ---
        # distance_transform_edt(input) computes distance from each non-zero pixel to nearest zero pixel.
        # We want distances from background pixels to the ignition point (which is 1).
        # So we invert: ignition point becomes 0 and background becomes 1 (1 - single_pixel_ignition_raster).
        # This yields 0 at the ignition point and increasing values farther away.
        distance_map = distance_transform_edt(1 - single_pixel_ignition_raster).astype(np.float32)

        # Normalize distance map to [0, 1], where 1.0 is ignition point.
        max_dist_value = np.max(distance_map)
        if max_dist_value > 0:
            ignition_raster_final = (max_dist_value - distance_map) / max_dist_value
        else:
            # If no ignition point exists (all zeros), normalized map stays all zeros.
            ignition_raster_final = np.zeros_like(distance_map)

        # 4. Define output TIFF profile using Sentinel spatial info
        output_profile = {
            "height": patch_size_pixels_h,
            "width": patch_size_pixels_w,
            "count": 1,  # Single band
            "dtype": ignition_raster_final.dtype,  # float32
            "crs": output_crs,  # Sentinel CRS
            "transform": output_transform,  # Sentinel transform
            "nodata": 0.0,  # NoData value for float
            "driver": "GTiff"
        }

        # 5. Save ignition raster as TIFF
        with rasterio.open(out_path_ignition_tif, "w", **output_profile) as dst:
            dst.write(ignition_raster_final, 1)

        print(f"🎉 Ignition point (distance map) TIFF saved: {os.path.basename(out_path_ignition_tif)}")
        return True

    except Exception as e:
        print(f"❌ Error while creating ignition point (distance map) raster for fire ID {fire_id}: {e}")
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    # General dataset/process configuration
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson",
        "root_dataset_folder": "piedmont_new",
        "target_crs": "EPSG:32632",  # Used only to convert ignition point from GeoJSON
                                    # Final ignition TIFF CRS will match the reference Sentinel CRS.
        "patch_size_pixels": 256,    # Used only as a check; actual dimensions come from Sentinel.
        "target_resolution_m": 10,   # Used only as a check; actual resolution comes from Sentinel.
    }

    # Validate dataset root folder
    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERROR: Root dataset folder '{config['root_dataset_folder']}' does not exist.")
        print(
            "Ensure that 'root_dataset_folder' in the configuration points to the directory "
            "containing the fire folders (e.g., 'piedmont_new')."
        )
        exit("Cannot proceed.")

    # Load GeoJSON containing ignition points
    print(f"Loading GeoJSON from: {config['geojson_path']}")
    try:
        main_geojson_gdf = gpd.read_file(config["geojson_path"])

        # Verify required columns exist
        if "id" not in main_geojson_gdf.columns:
            print("ERROR: Column 'id' not found in GeoJSON. Required to match with 'fire_XXX' folders.")
            exit("Cannot proceed.")

        if "point_x" not in main_geojson_gdf.columns or "point_y" not in main_geojson_gdf.columns:
            print("ERROR: Columns 'point_x' and/or 'point_y' not found in GeoJSON.")
            print("Ensure the GeoJSON contains these columns for the ignition point.")
            exit("Cannot proceed.")

    except Exception as e:
        print(f"ERROR: Unable to load GeoJSON from '{config['geojson_path']}': {e}")
        exit("Cannot proceed.")

    print("GeoJSON loaded successfully.")

    # Log file handling for processed IDs
    log_file_path = os.path.join(config["root_dataset_folder"], "processed_fire_ids_ignition_points.log")

    # --- Section to test a single fire_id (NOW COMMENTED OUT) ---
    # chosen_fire_id = "7074"  # <--- INSERT FIRE_ID TO TEST HERE
    # print(f"\n--- TEST: Generating ignition point raster for single fire ID: {chosen_fire_id} ---")
    # if chosen_fire_id in processed_ids and \
    #    os.path.exists(os.path.join(config["root_dataset_folder"], f"fire_{chosen_fire_id}", f"fire_{chosen_fire_id}_ignition_point.tif")):
    #     print(f"ℹ️ Fire ID: {chosen_fire_id} already processed (in log and file exists). Skipping regeneration.")
    # else:
    #     success = generate_ignition_point_raster(chosen_fire_id, main_geojson_gdf, config)
    #     if success:
    #         log_processed_id(log_file_path, chosen_fire_id)
    #         print(f"🎉 Ignition point for fire ID {chosen_fire_id} generated and logged.")
    #     else:
    #         print(f"❌ Ignition point process failed for fire ID {chosen_fire_id}. Not added to log.")
    # print("\nTest script finished.")

    # --- Section to run generation for ALL fire_id (NOW UNCOMMENTED) ---
    all_fire_ids = []
    # Rebuild list of all fire IDs present in folders
    for item in sorted(os.listdir(config["root_dataset_folder"])):
        full_path = os.path.join(config["root_dataset_folder"], item)
        if os.path.isdir(full_path) and item.startswith("fire_"):
            fire_id_str = item.replace("fire_", "")
            all_fire_ids.append(fire_id_str)

    print(f"Found {len(all_fire_ids)} fire folders in: {config['root_dataset_folder']}")

    # Even if you delete the log manually, this will be empty on the first run.
    # Still read it for safety in later runs where you don't want to regenerate everything.
    processed_ids = get_processed_ids(log_file_path)

    print(f"Fires with ignition points already processed (from log): {len(processed_ids)}")

    processed_count_current_run = 0
    for fire_id_str in all_fire_ids:
        # Skip if already processed AND output file exists.
        # After manual deletion, this will be false for most fires, forcing regeneration.
        if (fire_id_str in processed_ids and
                os.path.exists(os.path.join(
                    config["root_dataset_folder"],
                    f"fire_{fire_id_str}",
                    f"fire_{fire_id_str}_ignition_point.tif"
                ))):
            print(f"ℹ️ Fire ID: {fire_id_str} already processed and output file exists. Skipping.")
            continue

        success = generate_ignition_point_raster(fire_id_str, main_geojson_gdf, config)
        if success:
            processed_count_current_run += 1
            log_processed_id(log_file_path, fire_id_str)
        else:
            print(f"Ignition point process failed for fire ID {fire_id_str}. Not added to log.")

    print(f"\nProcessing completed. Generated {processed_count_current_run} new ignition point rasters.")
    print(f"Successfully processed fire IDs were recorded in: {log_file_path}")
    print("\nScript finished.")
