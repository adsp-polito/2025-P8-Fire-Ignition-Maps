import os
import rasterio
import numpy as np
import cv2
from rasterio.warp import Resampling
# Make sure the path is correct for your utils functions
from modelWithLandsat.utils import resample_image, read_image_and_metadata

# --- General Configuration ---
BASE_DATA_DIR = "piedmont_new"

# Suffix to append to resampled DEM filenames.
RESAMPLED_SUFFIX = "_10m.tif"


def find_images_by_criteria(directory, file_type="sentinel"):
    """
    Finds image files based on specific criteria.

    Args:
        directory (str): Directory path to search in.
        file_type (str): 'sentinel' for pre_sentinel.tif (non-CM),
                         'dem' for dem.tif (non-CM, not already resampled).

    Returns:
        list: A list of full paths for the files found.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if file_type == "sentinel":
                # Sentinel criteria: contains "pre_sentinel", ends with ".tif",
                # does NOT contain "_CM", does NOT contain the resampling suffix
                if "pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f and RESAMPLED_SUFFIX not in f:
                    found_files.append(os.path.join(root, f))
                    # For Sentinel we only need one reference, so we can stop at the first match
                    return found_files

            elif file_type == "dem":
                # DEM criteria: contains "dem", ends with ".tif",
                # does NOT contain "_CM", does NOT contain the resampling suffix
                if "dem" in f and f.endswith(".tif") and RESAMPLED_SUFFIX not in f:
                    found_files.append(os.path.join(root, f))

    return found_files


def process_fire_folder(fire_dir_path):
    """
    Processes a single fire folder: finds the Sentinel reference and all DEMs,
    then resamples each DEM to match the Sentinel grid.
    """
    print(f"\nProcessing folder: {fire_dir_path}")

    # Find the Sentinel reference image path (we take only one)
    sentinel_paths = find_images_by_criteria(fire_dir_path, file_type="sentinel")
    if not sentinel_paths:
        print(
            f"  WARNING: No 'pre_sentinel' file (non-CM) found in {fire_dir_path}. "
            f"Skipping this folder."
        )
        return

    # Use the first Sentinel found as reference for metadata
    sentinel_reference_path = sentinel_paths[0]

    # Find all original DEM image paths
    dem_paths = find_images_by_criteria(fire_dir_path, file_type="dem")
    if not dem_paths:
        print(
            f"  WARNING: No 'dem' file (non-CM, not resampled) found in {fire_dir_path}. "
            f"Skipping this folder or not resampling DEM."
        )
        return

    try:
        # Load Sentinel metadata (our 10 m reference)
        _, sentinel_transform, sentinel_crs, sentinel_shape, sentinel_res, _ = read_image_and_metadata(
            sentinel_reference_path
        )
        print(
            f"  Sentinel reference: shape={sentinel_shape}, res={sentinel_res}m, "
            f"path={os.path.basename(sentinel_reference_path)}"
        )

        # Process each DEM found
        for dem_path in dem_paths:
            original_dem_filename = os.path.basename(dem_path)
            dem_output_filename = original_dem_filename.replace(".tif", RESAMPLED_SUFFIX)
            output_dem_10m_path = os.path.join(os.path.dirname(dem_path), dem_output_filename)

            # Skip if already resampled
            if os.path.exists(output_dem_10m_path):
                print(
                    f"  Resampled DEM file '{os.path.basename(output_dem_10m_path)}' already exists. "
                    f"Skipping resampling for this file."
                )
                continue

            # Load DEM metadata (image to resample)
            _, dem_transform, dem_crs, dem_shape, dem_res, _ = read_image_and_metadata(dem_path)
            print(f"  DEM original: shape={dem_shape}, res={dem_res}m, path={original_dem_filename}")

            # Upsample DEM to 10 m
            print(f"  Upsampling DEM to 10 m: {original_dem_filename} -> {dem_output_filename}")
            resample_image(
                input_tif_path=dem_path,
                output_tif_path=output_dem_10m_path,
                target_transform=sentinel_transform,
                target_crs=sentinel_crs,
                target_shape=sentinel_shape  # (Height, Width)
            )
            print(f"  Resampled DEM saved to: {output_dem_10m_path}")

    except Exception as e:
        print(f"  Error while processing folder {fire_dir_path}: {e}")


# --- Main Execution ---
print(f"Starting pre-resampling process for folders in '{BASE_DATA_DIR}'...")

# Iterate through all subfolders in BASE_DATA_DIR
for item_name in os.listdir(BASE_DATA_DIR):
    full_path = os.path.join(BASE_DATA_DIR, item_name)
    if os.path.isdir(full_path) and item_name.startswith("fire_"):  # Only "fire_" folders
        process_fire_folder(full_path)

print("\nPre-resampling process completed for all fire folders.")
