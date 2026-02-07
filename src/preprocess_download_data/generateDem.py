import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import box, shape
import pandas as pd
from datetime import timedelta


def are_images_similar(img1, img2, tolerance_percentage=0.005):
    """Compare two NumPy image arrays for similarity."""
    if img1.shape != img2.shape or img1.dtype != img2.dtype:
        return False

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))

    if np.issubdtype(img1.dtype, np.integer):
        max_val = np.iinfo(img1.dtype).max
        normalized_diff = diff / max_val
    else:
        normalized_diff = diff

    mean_diff = np.mean(normalized_diff)
    return mean_diff < tolerance_percentage


def process_single_dem_for_fire(fire_id, config):
    """
    Downloads the Copernicus GLO-30 DEM (30 m) for a 256x256 pixel area
    centered on the fire geometry and saves it as a TIFF file.
    Returns True on success, False otherwise.
    """

    main_geojson_path = config["geojson_path"]
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]

    # Check whether the fire folder already exists (user requirement)
    FIRE_SAVE_FOLDER = os.path.join(ROOT_DATASET_FOLDER, f"fire_{fire_id}")
    if not os.path.isdir(FIRE_SAVE_FOLDER):
        print(f"ℹ️ Folder '{FIRE_SAVE_FOLDER}' not found. No action taken for fire ID: {fire_id}.")
        return False

    print(f"Loading GeoJSON from: {main_geojson_path}")
    gdf_all_fires = gpd.read_file(main_geojson_path)

    if "id" in gdf_all_fires.columns:
        print(f"DEBUG: Data type of 'id' column in GeoJSON: {gdf_all_fires['id'].dtype}")
        print(f"DEBUG: First 5 IDs in GeoJSON: {gdf_all_fires['id'].head().tolist()}")
        print(f"DEBUG: Type of searched fire ID: {type(fire_id)} (value: {fire_id})")
    else:
        print("DEBUG: WARNING: 'id' column not found in GeoJSON.")

    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632")
    if str(gdf_all_fires.crs) != TARGET_CRS_FOR_FIRES:
        print(f"Reprojecting GeoJSON from {gdf_all_fires.crs} to {TARGET_CRS_FOR_FIRES} for internal computations.")
        gdf_all_fires = gdf_all_fires.to_crs(TARGET_CRS_FOR_FIRES)

    fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
    if fire_data.empty:
        print(f"❌ Fire with ID {fire_id} not found in GeoJSON.")
        return False

    fire = fire_data.iloc[0]
    gt_geometry_utm = shape(fire["geometry"])

    min_fire_area_sq_m = config.get("min_fire_area_sq_m", 200)
    if gt_geometry_utm.area < min_fire_area_sq_m:
        print(
            f"❌ Fire ID {fire_id} has an area too small "
            f"({gt_geometry_utm.area:.2f} m²) to be useful at 30 m resolution. "
            f"Minimum required: {min_fire_area_sq_m} m². Skipped."
        )
        print(f"\n--- END process for fire ID: {fire_id} ---")
        return False

    TARGET_RESOLUTION_M = 30
    patch_size_pixels = config.get("patch_size_pixels", 256)
    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M

    print(f"\n--- Processing DEM for Fire ID: {fire_id} ---")
    print(
        f"Target patch size (DEM): {patch_size_pixels}x{patch_size_pixels} pixels "
        f"({fixed_patch_size_meters / 1000:.2f} km per side) at {TARGET_RESOLUTION_M} m."
    )
    print(f"Output folder: {FIRE_SAVE_FOLDER}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search_bbox_wgs84 = (
        gpd.GeoSeries([gt_geometry_utm.buffer(fixed_patch_size_meters / 2)],
                      crs=TARGET_CRS_FOR_FIRES)
        .to_crs(epsg=4326)
        .iloc[0]
        .bounds
    )
    print(f"STAC search with bbox (WGS84): {search_bbox_wgs84}")

    stac_collections = ["cop-dem-glo-30"]
    search = catalog.search(
        collections=stac_collections,
        bbox=search_bbox_wgs84,
        limit=50
    )

    items = list(search.items())
    if not items:
        print(f"❌ No DEM items found for fire ID {fire_id}.")
        print(f"\n--- END DEM process for fire ID: {fire_id} ---")
        return False

    print(f"✅ Found {len(items)} DEM items.")

    # Select the first DEM item intersecting the fire geometry
    dem_item_to_process = None
    for item_candidate in items:
        asset_name = "data"
        if asset_name not in item_candidate.assets:
            print(f"  Skipping item {item_candidate.id} - asset 'data' not available.")
            continue

        signed_href_check = pc.sign(item_candidate.assets[asset_name]).href
        try:
            with rasterio.open(signed_href_check) as src_check:
                item_bounds_polygon = gpd.GeoSeries(
                    [box(*src_check.bounds)],
                    crs=src_check.crs
                ).iloc[0]

                if str(src_check.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                    gt_geometry_in_item_crs_check = (
                        gpd.GeoSeries([gt_geometry_utm],
                                      crs=TARGET_CRS_FOR_FIRES)
                        .to_crs(src_check.crs)
                        .iloc[0]
                    )
                else:
                    gt_geometry_in_item_crs_check = gt_geometry_utm

                if gt_geometry_in_item_crs_check.intersects(item_bounds_polygon):
                    print(f"  Item {item_candidate.id} intersects fire geometry. Selected for processing.")
                    dem_item_to_process = item_candidate
                    break
                else:
                    print(f"  ❌ Item {item_candidate.id} does not intersect fire geometry.")
        except Exception as e:
            print(f"  Error while checking item {item_candidate.id}: {e}")

    if not dem_item_to_process:
        print("🔴 None of the found DEM images intersect the fire geometry. Cannot proceed with clipping.")
        print(f"\n--- END DEM process for fire ID: {fire_id} ---")
        return False

    print(f"\n📸 Processing DEM: {dem_item_to_process.id}.")

    dem_asset_key = "data"
    signed_href = pc.sign(dem_item_to_process.assets[dem_asset_key]).href

    try:
        with rasterio.open(signed_href) as src_dem:
            print(
                f"DEBUG: Source DEM CRS: {src_dem.crs}, "
                f"Bounds: {src_dem.bounds}, NoData value: {src_dem.nodata}"
            )

            if str(src_dem.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                gt_geometry_in_dem_crs = (
                    gpd.GeoSeries([gt_geometry_utm],
                                  crs=TARGET_CRS_FOR_FIRES)
                    .to_crs(src_dem.crs)
                    .iloc[0]
                )
            else:
                gt_geometry_in_dem_crs = gt_geometry_utm

            centroid_x = gt_geometry_in_dem_crs.centroid.x
            centroid_y = gt_geometry_in_dem_crs.centroid.y

            # Compute target patch bounds in source DEM CRS
            minx = centroid_x - fixed_patch_size_meters / 2
            miny = centroid_y - fixed_patch_size_meters / 2
            maxx = centroid_x + fixed_patch_size_meters / 2
            maxy = centroid_y + fixed_patch_size_meters / 2

            patch_bounds_in_src_crs = (minx, miny, maxx, maxy)

            # Compute window to read from source DEM
            window_to_read = from_bounds(
                *patch_bounds_in_src_crs,
                transform=src_dem.transform
            )

            # Ensure the window lies within DEM bounds
            window_to_read = window_to_read.intersection(
                rasterio.windows.Window(0, 0, src_dem.width, src_dem.height)
            )
            window_to_read = (
                window_to_read
                .round_offsets(op="floor")
                .round_lengths(op="ceil")
            )

            print(f"DEBUG: Calculated window to read from source DEM: {window_to_read}")
            print(
                f"DEBUG: Window bounds (source DEM CRS): "
                f"{rasterio.windows.bounds(window_to_read, src_dem.transform)}"
            )

            if window_to_read.width == 0 or window_to_read.height == 0:
                print(
                    "🔴 ERROR: Computed DEM read window is empty or invalid. "
                    "Fire may be too close to tile edge or data missing. Skipping."
                )
                print(f"\n--- END DEM process for fire ID: {fire_id} ---")
                return False

            raw_window_data = src_dem.read(
                1,
                window=window_to_read,
                masked=True
            ).squeeze()

            fill_value = src_dem.nodata if src_dem.nodata is not None else -9999.0
            dem_array_for_reproject = (
                raw_window_data.filled(fill_value)
                if isinstance(raw_window_data, np.ma.MaskedArray)
                else raw_window_data
            )

            print(
                f"DEBUG: Raw window data shape: {dem_array_for_reproject.shape}, "
                f"Dtype: {dem_array_for_reproject.dtype}"
            )
            print(
                f"DEBUG: Raw data stats — Min: {dem_array_for_reproject.min()}, "
                f"Max: {dem_array_for_reproject.max()}"
            )

            src_window_transform = rasterio.windows.transform(
                window_to_read,
                src_dem.transform
            )

            # Final transform in target CRS with fixed resolution and alignment
            final_transform = rasterio.transform.from_bounds(
                gt_geometry_utm.centroid.x - fixed_patch_size_meters / 2,
                gt_geometry_utm.centroid.y - fixed_patch_size_meters / 2,
                gt_geometry_utm.centroid.x + fixed_patch_size_meters / 2,
                gt_geometry_utm.centroid.y + fixed_patch_size_meters / 2,
                width=patch_size_pixels,
                height=patch_size_pixels
            )

            dem_data_reprojected = np.zeros(
                (patch_size_pixels, patch_size_pixels),
                dtype=np.float32
            )

            reproject(
                source=dem_array_for_reproject,
                destination=dem_data_reprojected,
                src_transform=src_window_transform,
                src_crs=src_dem.crs,
                dst_transform=final_transform,
                dst_crs=TARGET_CRS_FOR_FIRES,
                resampling=Resampling.bilinear,
                src_nodata=fill_value
            )

            dem_filename = f"fire_{fire_id}_dem.tif"
            out_path_dem_tif = os.path.join(FIRE_SAVE_FOLDER, dem_filename)

            output_profile = src_dem.profile.copy()
            output_profile.update({
                "height": patch_size_pixels,
                "width": patch_size_pixels,
                "count": 1,
                "dtype": dem_data_reprojected.dtype,
                "crs": TARGET_CRS_FOR_FIRES,
                "transform": final_transform,
                "nodata": -9999.0
            })

            with rasterio.open(out_path_dem_tif, "w", **output_profile) as dst:
                dst.write(dem_data_reprojected, 1)

            print(f"💾 DEM saved (single-band TIFF at 30 m): {out_path_dem_tif}")
            return True

    except Exception as e:
        print(f"❌ Critical error during DEM download/processing for fire ID {fire_id}: {e}")
        return False


def generate_dem_for_specific_fire(single_fire_id: str, config: dict):
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")

    print(f"\n--- Attempting DEM generation for fire ID: {single_fire_id} ---")
    print(f"Looking for folder: {fire_folder_path}")

    if os.path.isdir(fire_folder_path):
        print(f"✅ Folder '{fire_folder_path}' found. Starting DEM generation.")
        success = process_single_dem_for_fire(single_fire_id, config)
        if success:
            print(f"DEM process completed successfully for fire ID: {single_fire_id}.")
        else:
            print(f"DEM process failed for fire ID: {single_fire_id}. Check error logs.")
    else:
        print(f"ℹ️ Folder '{fire_folder_path}' not found. No action taken for fire ID: {single_fire_id}.")

    print(f"--- End attempt for fire ID: {single_fire_id} ---\n")


if __name__ == "__main__":
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson",
        "root_dataset_folder": "piedmont_new",
        "target_crs": "EPSG:32632",
        "min_fire_area_sq_m": 200,
        "patch_size_pixels": 256,
    }

    if not os.path.exists(config["root_dataset_folder"]):
        print(
            f"ERROR: Root dataset folder '{config['root_dataset_folder']}' does not exist."
        )
        print(
            "Ensure that 'root_dataset_folder' in the configuration points to the directory "
            "containing the fire folders (e.g., 'piedmont_new')."
        )
        exit("Cannot proceed.")

    print(
        f"Starting DEM raster processing for all folders in: "
        f"{config['root_dataset_folder']}"
    )

    processed_count = 0

    for item in os.listdir(config["root_dataset_folder"]):
        full_path = os.path.join(config["root_dataset_folder"], item)

        # Process only subfolders starting with "fire_"
        if os.path.isdir(full_path) and item.startswith("fire_"):
            fire_id = int(item.replace("fire_", ""))
            success = process_single_dem_for_fire(fire_id, config)
            if success:
                processed_count += 1

    print(f"\nProcessing completed. DEM rasters generated for {processed_count} fire folders.")
    print("\nScript finished.")
