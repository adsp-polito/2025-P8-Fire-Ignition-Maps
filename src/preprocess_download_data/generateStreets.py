import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import numpy as np
import os
import re

# --- Single Global Configuration (EDIT THIS PATH) ---
GLOBAL_ROADS_GEOJSON_PATH = "data/strade.geojson"  # <-- UPDATE THIS TO YOUR REAL PATH!

# Target CRS for all spatial operations (your EPSG:32632)
TARGET_CRS = "EPSG:32632"

# Load and reproject the roads GeoJSON once at the start of the script
strade_gdf = None  # Initialize as None; will be loaded if the path is valid
try:
    if os.path.exists(GLOBAL_ROADS_GEOJSON_PATH):
        print(f"Loading and reprojecting {GLOBAL_ROADS_GEOJSON_PATH} to {TARGET_CRS}...")
        strade_gdf = gpd.read_file(GLOBAL_ROADS_GEOJSON_PATH)
        strade_gdf = strade_gdf.to_crs(TARGET_CRS)
        # Create a spatial index for fast queries (essential for large datasets)
        strade_gdf.sindex
        print("Roads GeoDataFrame loaded and reprojected successfully, and spatial index created.")
    else:
        print(f"Error: roads.geojson file not found at path: {GLOBAL_ROADS_GEOJSON_PATH}")
        print("Make sure 'GLOBAL_ROADS_GEOJSON_PATH' is set correctly.")
except Exception as e:
    print(f"Error while loading or reprojecting the roads GeoJSON: {e}")
    strade_gdf = None


def generate_streets_raster_for_fire_folder(fire_folder_path: str):
    """
    Generates a roads raster (fire_id_streets.tif) for a given fire folder.

    Args:
        fire_folder_path (str): Full path to a single fire folder (e.g., 'data/fire_5411').
    """
    if strade_gdf is None:
        print(
            f"  Cannot process '{os.path.basename(fire_folder_path)}': "
            f"roads GeoJSON was not loaded correctly or not found."
        )
        return

    # Extract fire_ID from folder name
    fire_id = os.path.basename(fire_folder_path)
    print(f"\nProcessing fire: {fire_id}")

    # --- 1. Find a reference Sentinel image inside the folder ---
    # Any Sentinel .tif that is not a mask is fine to obtain spatial references.
    sentinel_ref_path = None
    for fname in os.listdir(fire_folder_path):
        if ("sentinel" in fname.lower()
                and fname.lower().endswith(".tif")
                and "_gt" not in fname.lower()
                and "_cm" not in fname.lower()
                and "_temp" not in fname.lower()):
            sentinel_ref_path = os.path.join(fire_folder_path, fname)
            break

    if sentinel_ref_path is None:
        print(f"  Error: No reference Sentinel .tif found in folder '{fire_folder_path}'. Skipping.")
        return

    print(f"  Found reference Sentinel image: {os.path.basename(sentinel_ref_path)}")

    # --- 2. Read spatial properties from the Sentinel image ---
    try:
        with rasterio.open(sentinel_ref_path) as src:
            img_bounds = src.bounds
            img_transform = src.transform
            img_width = src.width
            img_height = src.height
            img_crs = src.crs
            print(f"  Reference image dimensions: {img_width}x{img_height} pixels")
            print(f"  Reference image CRS: {img_crs}")

            if str(img_crs) != TARGET_CRS:
                print(
                    f"  Warning: Sentinel CRS ({img_crs}) does not match target CRS ({TARGET_CRS})."
                )
                print(
                    "  Ensure all images and spatial data use the correct CRS for precise alignment."
                )

    except Exception as e:
        print(f"  Error opening/reading reference image properties: {e}. Skipping.")
        return

    # --- 3. Filter roads intersecting the image bounding box ---
    bbox_polygon = Polygon([
        (img_bounds.left, img_bounds.bottom),
        (img_bounds.left, img_bounds.top),
        (img_bounds.right, img_bounds.top),
        (img_bounds.right, img_bounds.bottom)
    ])

    # Use spatial index for efficient querying
    possible_matches_index = list(strade_gdf.sindex.intersection(bbox_polygon.bounds))
    filtered_strade = strade_gdf.iloc[possible_matches_index].cx[
        img_bounds.left:img_bounds.right, img_bounds.bottom:img_bounds.top
    ]

    print(f"  Found {len(filtered_strade)} road geometries intersecting the image area.")

    # --- 4. Rasterize roads ---
    if filtered_strade.empty:
        print(f"  No valid roads found in the image area for {fire_id}.")
        roads_raster = np.zeros((img_height, img_width), dtype=np.uint8)
    else:
        # Prepare geometries with GP_RTP values to rasterize
        shapes_to_rasterize = [(row.geometry, row["GP_RTP"]) for idx, row in filtered_strade.iterrows()]

        roads_raster = rasterize(
            shapes=shapes_to_rasterize,
            out_shape=(img_height, img_width),
            transform=img_transform,
            all_touched=True,  # Capture all pixels touched by geometries
            fill=0,            # Background value for pixels without roads
            dtype=np.uint8     # Data type (0 for background, 1-5 for GP_RTP)
        )

    # --- 5. Save the TIFF raster ---
    output_tif_path = os.path.join(fire_folder_path, f"{fire_id}_streets.tif")

    try:
        with rasterio.open(
            output_tif_path,
            "w",
            driver="GTiff",
            height=img_height,
            width=img_width,
            count=1,  # Single band for road type
            dtype=roads_raster.dtype,
            crs=img_crs,            # Use reference image CRS
            transform=img_transform # Use reference image transform
        ) as dst:
            dst.write(roads_raster, 1)  # Write band 1 (roads_raster)
        print(f"  TIFF raster '{os.path.basename(output_tif_path)}' generated and saved successfully.")
    except Exception as e:
        print(
            f"  Error while saving roads TIFF raster: {e}. "
            f"The file may not have been created."
        )
        # Do not return here; allow loop to continue for other folders


# --- Block to process ALL "fire_*" folders ---
if __name__ == "__main__":
    # --- 1. SET THE PATH TO YOUR MAIN DATA DIRECTORY HERE ---
    main_data_root = "piedmont_new"  # <-- UPDATE THIS!

    if os.path.exists(main_data_root):
        print(f"Starting roads raster processing for all folders in: {main_data_root}")
        processed_count = 0
        for item in os.listdir(main_data_root):
            full_path = os.path.join(main_data_root, item)
            # Process only subfolders starting with "fire_"
            if os.path.isdir(full_path) and item.startswith("fire_"):
                generate_streets_raster_for_fire_folder(full_path)
                processed_count += 1
        print(f"\nProcessing completed. Generated rasters for {processed_count} fire folders.")
    else:
        print(f"Error: Main directory '{main_data_root}' does not exist. Check the path.")

    print("\nScript finished.")
