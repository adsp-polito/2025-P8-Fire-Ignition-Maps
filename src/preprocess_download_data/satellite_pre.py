import planetary_computer
import pystac_client
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os
import yaml
from dateutil import parser


def load_config(config_path="src/project_name/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_sentinel_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval=7):
    fire_geometry_buffered = fire_geometry.buffer(250)  # buffer in meters
    fire_geometry_buffered = gpd.GeoSeries([fire_geometry_buffered], crs=3857).to_crs(epsg=4326).iloc[0]
    fire_geometry = gpd.GeoSeries([fire_geometry], crs=3857).to_crs(epsg=4326).iloc[0]

    start_date = fire_date - timedelta(days=interval)
    end_date = fire_date
    time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    print("🔥 Searching for Sentinel-2 PRE-fire images")
    print("Fire date:", fire_date.strftime("%Y-%m-%d"))
    print("Search interval:", time_range)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=fire_geometry_buffered.bounds,
        datetime=time_range,
    )

    items = list(search.items())
    if not items:
        print("❌ No images found.")
        return

    print(f"✅ Found {len(items)} images.")

    sentinel2_bands = {
        "B02": 10, "B03": 10, "B04": 10, "B08": 10,
        "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
        "B01": 60, "B09": 60
    }

    date_counts = {}
    for i, item in enumerate(items):
        base_date = item.datetime.strftime("%Y-%m-%d")
        date_counts[base_date] = date_counts.get(base_date, 0) + 1
        suffix = f"_{date_counts[base_date]}" if date_counts[base_date] > 1 else ""
        full_date_str = f"{base_date}{suffix}"
        print(f"\n📸 Image {i+1} from {full_date_str}")

        band_arrays_resampled = []
        band_names_resampled = []
        profile = None
        window = None

        # Reference band
        reference_band = next((b for b, r in sentinel2_bands.items() if r == 10 and b in item.assets), None)
        if reference_band is None:
            print("⚠️ No 10m band found. Skipping.")
            continue

        signed_href = planetary_computer.sign(item.assets[reference_band]).href
        with rasterio.open(signed_href) as ref_src:
            bbox_transformed = gpd.GeoSeries([fire_geometry_buffered], crs=4326).to_crs(ref_src.crs).iloc[0].bounds
            window = rasterio.windows.from_bounds(*bbox_transformed, transform=ref_src.transform)
            profile = ref_src.profile.copy()
            profile.update({
                "height": int(window.height),
                "width": int(window.width),
                "transform": rasterio.windows.transform(window, ref_src.transform),
                "count": len(sentinel2_bands)
            })
        target_height = int(window.height)
        target_width = int(window.width)

        for band, native_resolution in sentinel2_bands.items():
            if band not in item.assets:
                continue
            signed_href = planetary_computer.sign(item.assets[band]).href
            with rasterio.open(signed_href) as src:
                try:
                    if native_resolution != 10:
                        scale = native_resolution / 10
                        scaled_window = rasterio.windows.Window(
                            col_off=window.col_off / scale,
                            row_off=window.row_off / scale,
                            width=window.width / scale,
                            height=window.height / scale
                        )
                        band_data = src.read(
                            1,
                            window=scaled_window,
                            out_shape=(target_height, target_width),
                            resampling=Resampling.bilinear
                        )
                    else:
                        band_data = src.read(
                            1,
                            window=window,
                            out_shape=(target_height, target_width),
                            resampling=Resampling.bilinear
                        )
                    if band_data.shape != (target_height, target_width):
                        print(f"⚠️ Band {band} has shape {band_data.shape}, expected {(target_height, target_width)}. Skipping.")
                        continue
                    band_arrays_resampled.append(band_data)
                    band_names_resampled.append(band)
                    print(f"  ✔️ Band {band} loaded.")
                except Exception as e:
                    print(f"  ❌ Error on band {band}: {e}")

        if not band_arrays_resampled:
            print("⚠️ No usable bands found. Skipping.")
            continue

        data_cube = np.stack(band_arrays_resampled, axis=0)
        profile.update(count=len(band_arrays_resampled))

        out_path = os.path.join(SAVE_FOLDER, f"fire_{fire_id}_{full_date_str}_pre_sentinel.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data_cube)
            dst.descriptions = band_names_resampled

        print(f"💾 Saved: {out_path}")


def process_modis_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval):
    # Step 1: Buffer and CRS transformation
    fire_geometry_buffered = gpd.GeoSeries([fire_geometry.buffer(250)], crs=3857).to_crs(epsg=4326).iloc[0]

    start_date = fire_date - timedelta(days=interval)
    end_date = fire_date
    time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    print("🔥 Searching for MODIS PRE-fire images")
    print("Fire date:", fire_date.strftime("%Y-%m-%d"))
    print("Search interval:", time_range)

    # Step 3: Search MODIS images
    search = catalog.search(
        collections=["modis-11A2-061"],  # or 11A1 for daily
        bbox=fire_geometry_buffered.bounds,
        datetime=time_range,
        query={"platform": {"eq": "terra"}}
    )
    items = list(search.items())
    print(f"Found {len(items)} images.")

    for i, item in enumerate(items):
        date_str = parser.isoparse(item.properties["start_datetime"]).strftime("%Y-%m-%d")
        print(f"[{i+1}/{len(items)}] Date: {date_str}")

        emis_31_asset = item.assets.get("Emis_31")
        emis_32_asset = item.assets.get("Emis_32")

        if not emis_31_asset or not emis_32_asset:
            print(f"  → Missing bands for {date_str}")
            continue

        emis_31_href = planetary_computer.sign(emis_31_asset.href)
        emis_32_href = planetary_computer.sign(emis_32_asset.href)

        try:
            with rasterio.open(emis_31_href) as src_31, rasterio.open(emis_32_href) as src_32:
                print(f"  Raster CRS: {src_31.crs}")

                bbox_m = gpd.GeoSeries([fire_geometry_buffered], crs=4326).to_crs(src_31.crs).iloc[0].bounds
                window = rasterio.windows.from_bounds(*bbox_m, transform=src_31.transform)

                data_31 = src_31.read(1, window=window)
                data_32 = src_32.read(1, window=window)

                if data_31.size == 0 or data_32.size == 0:
                    print(f"  ⚠️ Empty image for {date_str}, skipping.")
                    continue

                out_transform = src_31.window_transform(window)
                profile = src_31.profile.copy()
                profile.update({
                    "count": 2,
                    "height": data_31.shape[0],
                    "width": data_31.shape[1],
                    "transform": out_transform
                })

                out_path = os.path.join(SAVE_FOLDER, f"fire_{fire_id}_{date_str}_pre_modis.tif")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data_31, 1)
                    dst.write(data_32, 2)
                    dst.descriptions = ("Emis_31", "Emis_32")

                print(f"  ✅ Saved: {out_path}")
        except Exception as e:
            print(f"  ❌ Error for {date_str}: {e}")
            continue


def process_landsat_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval):
    fire_geometry_buffered = fire_geometry.buffer(250)  # 100 meters
    fire_geometry_buffered = gpd.GeoSeries([fire_geometry_buffered], crs=3857).to_crs(epsg=4326).iloc[0]
    fire_geometry = gpd.GeoSeries([fire_geometry], crs=3857).to_crs(epsg=4326).iloc[0]

    # ---------------------
    # Date range
    # ---------------------
    start_date = fire_date - timedelta(days=interval)
    end_date = fire_date
    time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    # print("Fire bounding box:", fire_geometry.bounds)
    # print("Approx centroid:", fire_geometry.centroid)
    print("Fire date:", fire_date.strftime("%Y-%m-%d"))
    print("Search interval:", time_range)

    # ---------------------
    # Search Landsat C2 L2
    # ---------------------
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=fire_geometry_buffered.bounds,
        datetime=time_range
    )

    items = list(search.items())

    if not items:
        print("No Landsat images found for this fire.")
        return

    print(f"Found {len(items)} Landsat C2 L2 images.")

    # Bands of interest and their resolutions
    bands_of_interest = [
        "coastal", "blue", "green", "red", "nir08",
        "swir16", "swir22", "lwir11", "lwir", "pan",
        "atran", "cdist", "drad", "emis", "emsd", "trad", "urad"
    ]

    landsat_band_resolutions = {
        "blue": 30, "green": 30, "red": 30, "nir08": 30, "swir16": 30, "swir22": 30,
        "lwir11": 30, "lwir": 30, "pan": 15, "coastal": 30,
        "atran": 30, "cdist": 30, "drad": 30, "emis": 30, "emsd": 30, "trad": 30, "urad": 30
    }

    date_counts = {}
    for i, item in enumerate(items):
        base_date = item.datetime.strftime("%Y-%m-%d")
        date_counts[base_date] = date_counts.get(base_date, 0) + 1
        suffix = f"_{date_counts[base_date]}" if date_counts[base_date] > 1 else ""
        full_date_str = f"{base_date}{suffix}"

        print(f"\nImage {i+1}: acquired on {full_date_str}")

        band_arrays_resampled = []
        band_names_resampled = []
        profile = None
        window = None

        # Use "red" as reference band
        if "red" not in item.assets:
            print("⚠️ No 'red' band available as reference. Skipping.")
            continue

        signed_href = planetary_computer.sign(item.assets["red"]).href
        with rasterio.open(signed_href) as ref_src:
            xres, yres = abs(ref_src.res[0]), abs(ref_src.res[1])
            print(f"🛰️ Raster resolution for 'red': {xres:.2f}m x {yres:.2f}m")

            bbox_transformed = gpd.GeoSeries([fire_geometry_buffered], crs=4326).to_crs(ref_src.crs).iloc[0].bounds
            window = rasterio.windows.from_bounds(*bbox_transformed, transform=ref_src.transform)
            profile = ref_src.profile.copy()

            profile.update({
                "height": int(window.height),
                "width": int(window.width),
                "transform": rasterio.windows.transform(window, ref_src.transform),
                "count": len(bands_of_interest)
            })

        target_height = int(window.height)
        target_width = int(window.width)

        for band in bands_of_interest:
            if band in item.assets:
                signed_href = planetary_computer.sign(item.assets[band]).href
                with rasterio.open(signed_href) as src:
                    try:
                        native_res = landsat_band_resolutions.get(band, 30)

                        if native_res != 30:
                            # Scale the window based on resolution
                            scale = native_res / 30
                            scaled_window = rasterio.windows.Window(
                                col_off=window.col_off / scale,
                                row_off=window.row_off / scale,
                                width=window.width / scale,
                                height=window.height / scale
                            )

                            band_data = src.read(
                                1,
                                window=scaled_window,
                                out_shape=(target_height, target_width),
                                resampling=Resampling.bilinear
                            )
                        else:
                            band_data = src.read(
                                1,
                                window=window,
                                out_shape=(target_height, target_width),
                                resampling=Resampling.bilinear
                            )

                        if band_data.shape != (target_height, target_width):
                            print(f"⚠️ Band {band} has shape {band_data.shape}, expected {(target_height, target_width)}. Skipping.")
                            continue

                        band_arrays_resampled.append(band_data)
                        band_names_resampled.append(band)
                        print(f"  -> Band {band} read and cropped.")

                    except Exception as e:
                        print(f"  -> Error loading band {band}: {e}")

        if not band_arrays_resampled:
            print("No valid bands after cropping. Skipping.")
            continue

        data_cube = np.stack(band_arrays_resampled, axis=0)
        profile.update(count=len(band_arrays_resampled))

        cube_path = os.path.join(SAVE_FOLDER, f"fire_{fire_id}_{full_date_str}_pre_landsat.tif")
        with rasterio.open(cube_path, "w", **profile) as dst:
            dst.write(data_cube)
            dst.descriptions = band_names_resampled

        print(f"✅ Cropped Landsat TIFF saved to: {cube_path}")


def main():
    config = load_config()
    fire_id = config["fire_id"]
    satellite = config["satellite"].lower()
    geojson_path = config["geojson_path"]
    interval = config["interval"]

    SAVE_FOLDER = os.path.join("piedmont", f"fire_{fire_id}")
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # ----------------------------
    # Read fire geometry
    # ----------------------------
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=3857)
    fire = gdf[gdf["id"] == fire_id].iloc[0]
    fire_date = pd.to_datetime(fire["initialdate"])
    fire_geometry = shape(fire["geometry"])

    # ---------------------
    # Open the catalog
    # ---------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # ----------------------------
    # Run per satellite
    # ----------------------------
    if satellite == "sentinel" or satellite == "all":
        process_sentinel_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval)
    if satellite == "modis" or satellite == "all":
        process_modis_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval)
    if satellite == "landsat" or satellite == "all":
        process_landsat_pre(fire_geometry, fire_date, fire_id, SAVE_FOLDER, catalog, interval)


if __name__ == "__main__":
    main()
