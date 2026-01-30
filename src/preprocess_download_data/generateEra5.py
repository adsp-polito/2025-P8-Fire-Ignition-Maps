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
import sys # Importa sys per accedere agli argomenti da riga di comando

# IMPORTANTE: Assicurati che questa importazione sia corretta per il tuo progetto
from modelWithLandsat.utils import find_best_image_in_folder

# --- Funzioni di Utilità (rimangono le stesse) ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Estrae la data di acquisizione da un nome di file Sentinel con il formato:
    'fire_xxxx_YYYY-MM-DD_pre_sentinel_N.tif' (dove xxxx è l'ID dell'incendio).

    Args:
        filepath (str): Il percorso completo o il nome del file Sentinel.

    Returns:
        datetime.datetime: L'oggetto datetime parsato (solo data), o None se il formato non corrisponde.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'fire_\d+_(\d{4}-\d{2}-\d{2})_pre_sentinel_\d+\.tif', filename)
    
    if match:
        datetime_str = match.group(1)
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d')
        except ValueError:
            print(f"ATTENZIONE: Impossibile parsare la data '{datetime_str}' dal file: {filename}")
            return None
    else:
        print(f"ATTENZIONE: Nessun pattern di data trovato nel nome del file: {filename}")
        return None

def get_image_spatial_info(image_path: str, target_crs: str):
    """
    Estrae il centroide e la bounding box di un'immagine TIFF nel CRS target.

    Args:
        image_path (str): Percorso al file TIFF (es. Sentinel/Landsat).
        target_crs (str): Il CRS target (es. "EPSG:32632").

    Returns:
        tuple: (centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds) in WGS84 per la ricerca CDS.
               Restituisce (None, None, None) in caso di errore.
    """
    try:
        with rasterio.open(image_path) as src:
            src_bounds = src.bounds
            src_crs = src.crs

            # Calcola il centroide nel CRS sorgente
            centroid_x_src = (src_bounds.left + src_bounds.right) / 2
            centroid_y_src = (src_bounds.bottom + src_bounds.top) / 2
            
            # Crea un GeoDataFrame con il centroide nel CRS sorgente
            gdf_centroid_src = gpd.GeoDataFrame(
                geometry=[gpd.points_from_xy([centroid_x_src], [centroid_y_src])[0]],
                crs=src_crs
            )
            
            # Converte il centroide in WGS84 (EPSG:4326)
            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326)
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]
            
            # Calcola una bounding box in WGS84 basata sul centroide e una dimensione fissa.
            # Questo è per la richiesta CDS API. ERA5 ha una risoluzione di 0.1 gradi (~10km).
            # Una bbox di +/- 0.5 gradi (~50km) attorno al centroide dovrebbe coprire ampiamente
            # la patch di 2.56km x 2.56km e fornire dati sufficienti per l'interpolazione.
            # L'ordine per CDS API è [north, west, south, east]
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,  # north
                centroid_lon_wgs84 - 0.5,  # west
                centroid_lat_wgs84 - 0.5,  # south
                centroid_lon_wgs84 + 0.5   # east
            ]
            
            print(f"  Info Spaziali: Centroide (WGS84) ({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f}), BBox CDS {bbox_wgs84_for_cds}")
            
            return centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds
    except Exception as e:
        print(f"❌ Errore estrazione info spaziali da '{os.path.basename(image_path)}': {e}")
        return None, None, None

def download_and_process_era5_land(fire_id: str, image_datetime: datetime, centroid_lon_wgs84: float, centroid_lat_wgs84: float, bbox_cds: list, config: dict) -> bool:
    """
    Scarica i dati ERA5-Land "derived-era5-land-daily-statistics" dal CDS API per la data e l'area specificate,
    li ritaglia e li riesampiona alla dimensione della patch desiderata (256x256 a 10m) in un unico TIFF multi-banda.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    TARGET_RESOLUTION_M = config.get("target_resolution_m", 10) # Risoluzione target per i dati ERA5 (10m)
    patch_size_pixels = config.get("patch_size_pixels", 256)
    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632")

    os.makedirs(FIRE_SAVE_FOLDER, exist_ok=True)

    client = cdsapi.Client()
    dataset = "derived-era5-land-daily-statistics" # Dataset di statistiche giornaliere
    
    era5_variables_config = [
        {"cds_name": "2m_temperature", "xr_name": "t2m"},
        {"cds_name": "10m_u_component_of_wind", "xr_name": "u10"},
        {"cds_name": "10m_v_component_of_wind", "xr_name": "v10"}
    ]

    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M # 256 * 10 = 2560 metri

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
        
        temp_nc_file = os.path.join(FIRE_SAVE_FOLDER, f"temp_era5_land_{var_cds_name.replace(' ', '_')}_{image_datetime.strftime('%Y%m%d')}.nc")

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
                print(f"  Download '{var_cds_name}' (tentativo {attempt + 1}/{MAX_RETRIES})...")
                client.retrieve(dataset, request_params, temp_nc_file)
                download_success = True
                print(f"  Download '{var_cds_name}' completato.")
                break
            except Exception as e:
                print(f"  ❌ Errore download '{var_cds_name}': {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Ritento tra {RETRY_DELAY_SEC} secondi...")
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    print(f"  ❌ Fallito download '{var_cds_name}' dopo {MAX_RETRIES} tentativi.")
                    success_all_vars = False
                    break
        
        if not download_success:
            continue 

        try:
            with xr.open_dataset(temp_nc_file, engine='netcdf4') as ds_era5:
                era5_data_array_slice = ds_era5[var_xr_name].isel(valid_time=0).squeeze()

                if len(era5_data_array_slice.dims) != 2:
                    print(f"  ❌ Variabile '{var_xr_name}' non 2D ({era5_data_array_slice.dims}). Skip.")
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
                src_crs_era5 = 'EPSG:4326' 

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
                print(f"  Elaborazione '{var_cds_name}' completata.")

        except Exception as e:
            print(f"  ❌ Errore elaborazione '{var_cds_name}' per incendio ID {fire_id}: {e}")
            success_all_vars = False
        finally: 
            if os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                print(f"  File temporaneo '{os.path.basename(temp_nc_file)}' rimosso.")
            
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

            print(f"🎉 ERA5 multi-band TIFF salvato: {os.path.basename(out_path_era5_tif)}")
            print(f"  Bande contenute: {', '.join(band_names)}")
        except Exception as e:
            print(f"❌ Errore scrittura TIFF multi-banda: {e}")
            success_all_vars = False
    elif not reprojected_data_bands:
        print("⚠️ Nessun dato ERA5 riproiettato disponibile per TIFF multi-banda.")
        success_all_vars = False
            
    return success_all_vars

# --- Funzione per il logging degli ID processati (rimane la stessa) ---
def log_processed_id(log_file: str, fire_id: str):
    with open(log_file, 'a') as f:
        f.write(f"{fire_id}\n")

def get_processed_ids(log_file: str) -> set:
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}

# --- Funzione Wrapper Principale (rimane la stessa) ---
def generate_era5_for_specific_fire(single_fire_id: str, config: dict):
    """
    Funzione wrapper per avviare il processo di generazione ERA5 per un singolo incendio.

    Args:
        single_fire_id (str): L'ID dell'incendio da processare.
        config (dict): Dizionario di configurazione.
    """
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")
    era5_output_filename_check = f"fire_{single_fire_id}_era5_multi_band_" # Prefisso per controllo esistenza file ERA5

    if not os.path.isdir(fire_folder_path):
        print(f"ℹ️ Cartella '{fire_folder_path}' non trovata per ID: {single_fire_id}. Skippo.")
        return False

    era5_file_exists = any(f.startswith(era5_output_filename_check) and f.endswith(".tif") 
                           for f in os.listdir(fire_folder_path))

    if era5_file_exists:
        print(f"✅ ERA5 raster già presente per incendio ID: {single_fire_id}. Skippo.")
        return True # Considera già processato con successo

    print(f"\n--- Inizio elaborazione ERA5 per incendio ID: {single_fire_id} ---")
    print(f"  Cercando Sentinel in: {fire_folder_path}")

    best_image_info = find_best_image_in_folder(fire_folder_path)
    if best_image_info is None or 'sentinel_path' not in best_image_info:
        print(f"❌ Nessuna immagine Sentinel valida trovata per ID: {single_fire_id}. Impossibile cercare ERA5.")
        return False
    
    pre_sentinel_image_path = best_image_info['sentinel_path']
    image_datetime = parse_datetime_from_sentinel_filename(pre_sentinel_image_path)

    if not image_datetime:
        print(f"❌ Impossibile determinare data immagine Sentinel da '{os.path.basename(pre_sentinel_image_path)}'.")
        return False
    
    print(f"  Data immagine Sentinel: {image_datetime.isoformat()} da: {os.path.basename(pre_sentinel_image_path)}")

    centroid_lon_wgs84, centroid_lat_wgs84, bbox_cds = get_image_spatial_info(pre_sentinel_image_path, config["target_crs"])
    if centroid_lon_wgs84 is None:
        print(f"❌ Impossibile ottenere info spaziali da Sentinel per ID: {single_fire_id}.")
        return False

    success = download_and_process_era5_land(fire_id=single_fire_id, 
                                             image_datetime=image_datetime, 
                                             centroid_lon_wgs84=centroid_lon_wgs84, 
                                             centroid_lat_wgs84=centroid_lat_wgs84, 
                                             bbox_cds=bbox_cds, 
                                             config=config) 
    
    if success:
        print(f"✅ Processo ERA5 completato per incendio ID: {single_fire_id}.")
    else:
        print(f"❌ Processo ERA5 fallito per incendio ID: {single_fire_id}. Controllare i log.")
    print(f"--- Fine elaborazione per incendio ID: {single_fire_id} ---\n")
    return success


# --- Blocco Principale di Esecuzione (MODIFICATO per array jobs) ---
if __name__ == "__main__":
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
    
    log_file_path = os.path.join(config["root_dataset_folder"], "processed_fire_ids_era5.log")
    processed_ids = get_processed_ids(log_file_path)

    # Recupera gli ID dei folder fire_XXX
    all_fire_ids = []
    for item in sorted(os.listdir(config["root_dataset_folder"])):
        full_path = os.path.join(config["root_dataset_folder"], item)
        if os.path.isdir(full_path) and item.startswith("fire_"):
            fire_id_str = item.replace("fire_", "")
            all_fire_ids.append(fire_id_str)

    print(f'Trovate {len(all_fire_ids)} cartelle di incendi in: {config["root_dataset_folder"]}')
    print(f'Incendi già processati (da log): {processed_ids}')

    # Gestione degli argomenti del job array
    task_id = 0
    num_tasks = 1
    # sys.argv[0] è il nome dello script stesso
    if len(sys.argv) > 2: # Se ci sono almeno 2 argomenti (task_id e num_tasks)
        try:
            task_id = int(sys.argv[1])
            num_tasks = int(sys.argv[2])
            print(f"Esecuzione come task array: Task ID {task_id} di {num_tasks} totali.")
        except ValueError:
            print("AVVISO: Argomenti SLURM_ARRAY_TASK_ID/COUNT non validi. Eseguo in modalità sequenziale.")
    else:
        print("Esecuzione in modalità sequenziale (non come job array Slurm).")


    # Filtra i fire_ids da processare per questo specifico task array
    fire_ids_for_this_task = []
    for i, fire_id_str in enumerate(all_fire_ids):
        # La logica modulo distribuisce gli ID in modo circolare tra i task
        if i % num_tasks == task_id: 
            fire_ids_for_this_task.append(fire_id_str)
    
    print(f"Questo task ({task_id}/{num_tasks}) processerà {len(fire_ids_for_this_task)} incendi.")

    processed_count_current_run = 0
    for fire_id_str in fire_ids_for_this_task:
        # Questo controllo è fondamentale per evitare di riprocessare ciò che è già stato fatto
        # sia da questo job in un run precedente, sia da altri job nel run corrente
        if fire_id_str in processed_ids:
            print(f"ℹ️ Incendio ID: {fire_id_str} già presente nel log o file ERA5 esistente. Skippo.")
            continue 

        success = generate_era5_for_specific_fire(fire_id_str, config)
        if success:
            processed_count_current_run += 1
            # Logga l'ID solo se il processo ha successo e non era già presente nel log all'inizio del run
            # (il controllo `fire_id_str in processed_ids` sopra gestisce questo)
            log_processed_id(log_file_path, fire_id_str) 
        else:
            print(f"Processo ERA5 per incendio ID {fire_id_str} fallito. Non aggiunto al log.")
    
    print(f"\nElaborazione completata per task {task_id}. Processati {processed_count_current_run} nuovi incendi.")
    print(f"Gli ID degli incendi processati con successo sono stati registrati in: {log_file_path}")
    print("\nScript terminato.")