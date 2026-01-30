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
# Non usiamo più distance_transform_edt, ma usiamo la dilatazione per creare l'area localizzata
from scipy.ndimage import generate_binary_structure, binary_dilation 

# IMPORTANTE: Assicurati che questa importazione sia corretta per il tuo progetto
from modelWithLandsat.utils import find_best_image_in_folder

# --- Funzioni di Utilità ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Estrae la data da un nome file Sentinel specifico per gli incendi.
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
    Estrae il centroide (lon, lat in WGS84) da un'immagine raster (es. Sentinel).
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
            
            print(f"   Info Spaziali (per log): Centroide immagine (WGS84) ({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f})")
            
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,
                centroid_lon_wgs84 - 0.5,
                centroid_lat_wgs84 - 0.5,
                centroid_lon_wgs84 + 0.5
            ]
            
            return centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds
    except Exception as e:
        print(f"❌ Errore estrazione info spaziali da '{os.path.basename(image_path)}': {e}")
        return None, None, None

def log_processed_id(log_file: str, fire_id: str):
    """
    Registra l'ID di un incendio processato in un file di log.
    """
    with open(log_file, 'a') as f:
        f.write(f"{fire_id}\n")

def get_processed_ids(log_file: str) -> set:
    """
    Legge gli ID degli incendi già processati dal file di log.
    """
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}

# --- Funzione per Generare il Raster del Punto di Accensione (NUOVA VERSIONE) ---
def generate_ignition_point_raster(fire_id: str, geojson_gdf: gpd.GeoDataFrame, config: dict, spread_radius: int = 3) -> bool:
    """
    Crea un TIFF 256x256 con un punto di accensione e un'area circostante.
    Il valore 1.0 corrisponde all'area di ignizione, 0.0 ai pixel di background.
    
    Argomenti:
        fire_id (str): L'ID dell'incendio.
        geojson_gdf (gpd.GeoDataFrame): Il GeoDataFrame contenente i punti di ignizione.
        config (dict): Il dizionario di configurazione.
        spread_radius (int): La dimensione dell'area attorno al punto di ignizione. 
                             Un valore di 1 crea un'area di 3x3 pixel.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    ignition_output_filename = f"fire_{fire_id}_ignition_pt.tif"
    out_path_ignition_tif = os.path.join(FIRE_SAVE_FOLDER, ignition_output_filename)

    # Controlla se il file esiste già per evitare rigenerazioni inutili
    if os.path.exists(out_path_ignition_tif):
        print(f"✅ Ignition point raster (punto rosso) già presente per incendio ID: {fire_id}. Skippo.")
        return True

    print(f"\n--- Inizio generazione ignition point raster (punto rosso) per incendio ID: {fire_id} ---")

    # 1. Trova l'immagine Sentinel pre-incendio esistente.
    best_image_info = find_best_image_in_folder(FIRE_SAVE_FOLDER)
    if best_image_info is None or 'sentinel_path' not in best_image_info:
        print(f"❌ Nessuna immagine Sentinel valida trovata in '{FIRE_SAVE_FOLDER}'. Impossibile allineare l'ignition map.")
        return False
    
    pre_sentinel_image_path = best_image_info['sentinel_path']
    print(f"   Trovata immagine Sentinel per allineamento: {os.path.basename(pre_sentinel_image_path)}")

    # 2. Ottieni il profilo spaziale direttamente dall'immagine Sentinel
    try:
        with rasterio.open(pre_sentinel_image_path) as src_sentinel:
            output_transform = src_sentinel.transform
            output_crs = src_sentinel.crs
            patch_size_pixels_h = src_sentinel.height
            patch_size_pixels_w = src_sentinel.width
            
            if patch_size_pixels_h != config["patch_size_pixels"] or patch_size_pixels_w != config["patch_size_pixels"]:
                print(f"⚠️ Attenzione: Dimensioni immagine Sentinel ({patch_size_pixels_h}x{patch_size_pixels_w}) non corrispondono a config['patch_size_pixels'] ({config['patch_size_pixels']}x{config['patch_size_pixels']}). Userò le dimensioni della Sentinel.")
            
            print(f"   Profilo di output derivato da Sentinel: Dims {patch_size_pixels_h}x{patch_size_pixels_w}, CRS: {output_crs}")
    except Exception as e:
        print(f"❌ Errore apertura Sentinel per info spaziali '{os.path.basename(pre_sentinel_image_path)}': {e}")
        return False

    # 3. Estrai il punto di accensione dal GeoJSON
    try:
        fire_feature = geojson_gdf[geojson_gdf['id'] == int(fire_id)]
        
        if fire_feature.empty:
            print(f"❌ Incendio ID {fire_id} non trovato nel GeoJSON. Impossibile creare ignition point raster.")
            return False
        
        ignition_x_geojson = fire_feature['point_x'].iloc[0]
        ignition_y_geojson = fire_feature['point_y'].iloc[0]
        
        gdf_ignition_geojson_crs = gpd.GeoDataFrame(
            geometry=[Point(ignition_x_geojson, ignition_y_geojson)],
            crs="EPSG:3857"
        )
        
        gdf_ignition_target_crs = gdf_ignition_geojson_crs.to_crs(output_crs)
        ignition_x_target_crs = gdf_ignition_target_crs.geometry.x.iloc[0]
        ignition_y_target_crs = gdf_ignition_target_crs.geometry.y.iloc[0]
        
        row, col = rasterio.transform.rowcol(output_transform, ignition_x_target_crs, ignition_y_target_crs)

        # --- Crea l'immagine binaria con un'area localizzata attorno al punto ---
        ignition_raster_final = np.zeros((patch_size_pixels_h, patch_size_pixels_w), dtype=np.float32)

        if 0 <= row < patch_size_pixels_h and 0 <= col < patch_size_pixels_w:
            # Crea la maschera di dilatazione con la forma a croce
            kernel = generate_binary_structure(2, 1)
            
            # Crea un raster temporaneo con il solo pixel di ignizione a 1
            temp_ignition_pixel = np.zeros_like(ignition_raster_final, dtype=np.uint8)
            temp_ignition_pixel[row, col] = 1

            # Applica la dilatazione per creare un'area più ampia
            dilated_area = binary_dilation(temp_ignition_pixel, structure=kernel, iterations=spread_radius)
            ignition_raster_final[dilated_area] = 1.0
            
            print(f"   Punto di accensione e intorno (raggio {spread_radius}) posizionato a pixel ({row}, {col}).")
        else:
            print(f"⚠️ Punto di accensione per incendio ID {fire_id} FUORI DAI LIMITI del raster. Il raster dell'ignition point sarà tutto zero.")
        
        # 4. Definisci il profilo per il TIFF in output
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

        # 5. Salva il raster dell'ignition point
        with rasterio.open(out_path_ignition_tif, "w", **output_profile) as dst:
            dst.write(ignition_raster_final, 1)
        
        print(f"🎉 Ignition point (punto rosso) TIFF salvato: {os.path.basename(out_path_ignition_tif)}")
        return True

    except Exception as e:
        print(f"❌ Errore durante la creazione del raster ignition point per incendio ID {fire_id}: {e}")
        return False

# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # Configurazione generale del dataset e del processo
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson",
        "root_dataset_folder": "piedmont_new",
        "target_crs": "EPSG:32632",
        "patch_size_pixels": 256,
        "target_resolution_m": 10,
    }

    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERRORE: La cartella radice del dataset '{config['root_dataset_folder']}' non esiste.")
        print("Assicurati che 'root_dataset_folder' nella configurazione punti alla directory dove si trovano le cartelle degli incendi (es. 'piedmont_new').")
        exit("Impossibile procedere.")
    
    print(f"Caricamento GeoJSON da: {config['geojson_path']}")
    try:
        main_geojson_gdf = gpd.read_file(config["geojson_path"])
        if 'id' not in main_geojson_gdf.columns:
            print("ERRORE: Colonna 'id' non trovata nel GeoJSON. Necessaria per il matching con le cartelle 'fire_XXX'.")
            exit("Impossibile procedere.")
        
        if 'point_x' not in main_geojson_gdf.columns or 'point_y' not in main_geojson_gdf.columns:
            print("ERRORE: Colonne 'point_x' o 'point_y' non trovate nel GeoJSON.")
            print("Assicurati che il GeoJSON contenga queste colonne per il punto di accensione.")
            exit("Impossibile procedere.")

    except Exception as e:
        print(f"ERRORE: Impossibile caricare il GeoJSON da '{config['geojson_path']}': {e}")
        exit("Impossibile procedere.")
    print("GeoJSON caricato con successo.")

    # Gestione del file di log per gli ID già processati
    log_file_path = os.path.join(config["root_dataset_folder"], "processed_fire_ids_ignition_points.log")
    processed_ids = get_processed_ids(log_file_path)

    print(f'Trovate {len(os.listdir(config["root_dataset_folder"]))} cartelle di incendi in: {config["root_dataset_folder"]}')
    print(f'Incendi con ignition point già processato (da log): {len(processed_ids)}')

    all_fire_ids = [item.replace("fire_", "") for item in sorted(os.listdir(config["root_dataset_folder"])) if os.path.isdir(os.path.join(config["root_dataset_folder"], item)) and item.startswith("fire_")]

    processed_count_current_run = 0
    for fire_id_str in all_fire_ids:
        if fire_id_str in processed_ids and os.path.exists(os.path.join(config["root_dataset_folder"], f"fire_{fire_id_str}", f"fire_{fire_id_str}_ignition_pt.tif")):
            print(f"ℹ️ Incendio ID: {fire_id_str} già processato e file esistente. Skippo.")
            continue
        
        success = generate_ignition_point_raster(fire_id_str, main_geojson_gdf, config) 
        if success:
            processed_count_current_run += 1
            log_processed_id(log_file_path, fire_id_str)
        else:
            print(f"Processo ignition point per incendio ID {fire_id_str} fallito. Non aggiunto al log.")
    
    print(f"\nElaborazione completata. Generati {processed_count_current_run} nuovi ignition point raster.")
    print(f"Gli ID degli incendi con ignition point generato sono stati registrati in: {log_file_path}")
    print("\nScript terminato.")