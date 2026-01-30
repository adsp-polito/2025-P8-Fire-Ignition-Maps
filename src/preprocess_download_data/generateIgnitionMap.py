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
from scipy.ndimage import distance_transform_edt # Importazione per la distanza euclidea
# import cv2 # Non usato in questo script, può essere rimosso o mantenuto se usato altrove

# IMPORTANTE: Assicurati che questa importazione sia corretta per il tuo progetto
# Questa funzione è necessaria per trovare l'immagine Sentinel da cui derivare la georeferenziazione
# Assicurati che il percorso 'modelWithLandsat.utils' sia corretto per la tua struttura di progetto
from modelWithLandsat.utils import find_best_image_in_folder

# --- Funzioni di Utilità ---
def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Estrae la data da un nome file Sentinel specifico per gli incendi.
    """
    filename = os.path.basename(filepath)
    # Regex per catturare la data nel formato YYYY-MM-DD
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
    Questa funzione è usata per debug o log, non più per definire il bounds della patch di output
    dell'ignition point.
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
            
            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326) # Converte sempre in WGS84
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]
            
            print(f"   Info Spaziali (per log): Centroide immagine (WGS84) ({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f})")
            
            # Il bbox_wgs84_for_cds non è strettamente necessario per questo script, ma mantenuto per compatibilità di firma
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.5,   # north
                centroid_lon_wgs84 - 0.5,   # west
                centroid_lat_wgs84 - 0.5,   # south
                centroid_lon_wgs84 + 0.5    # east
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

# --- Funzione per Generare il Raster del Punto di Accensione (MODIFICATA) ---
def generate_ignition_point_raster(fire_id: str, geojson_gdf: gpd.GeoDataFrame, config: dict) -> bool:
    """
    Crea un TIFF 256x256 con la mappa di distanza euclidea dal punto di accensione,
    allineato spazialmente con l'immagine Sentinel pre-incendio esistente.
    Il valore 1.0 corrisponde al punto di ignizione, 0.0 ai pixel più lontani.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    ignition_output_filename = f"fire_{fire_id}_ignition_map.tif"
    out_path_ignition_tif = os.path.join(FIRE_SAVE_FOLDER, ignition_output_filename)

    # Controlla se il file esiste già per evitare rigenerazioni inutili
    # Questo controllo è essenziale per le run future, ma non per la prima run dopo la cancellazione
    if os.path.exists(out_path_ignition_tif):
        print(f"✅ Ignition point raster (distance map) già presente per incendio ID: {fire_id}. Skippo.")
        return True

    print(f"\n--- Inizio generazione ignition point raster (distance map) per incendio ID: {fire_id} ---")

    # 1. Trova l'immagine Sentinel pre-incendio esistente.
    # Useremo il suo profilo spaziale (CRS, trasformazione, dimensioni) per garantire l'allineamento.
    best_image_info = find_best_image_in_folder(FIRE_SAVE_FOLDER)
    if best_image_info is None or 'sentinel_path' not in best_image_info:
        print(f"❌ Nessuna immagine Sentinel valida trovata in '{FIRE_SAVE_FOLDER}'. Impossibile allineare l'ignition map.")
        return False
    
    pre_sentinel_image_path = best_image_info['sentinel_path']
    print(f"   Trovata immagine Sentinel per allineamento: {os.path.basename(pre_sentinel_image_path)}")

    # 2. Ottieni il profilo spaziale direttamente dall'immagine Sentinel
    try:
        with rasterio.open(pre_sentinel_image_path) as src_sentinel:
            # Questi saranno i parametri spaziali per la nostra immagine di output.
            # Sarà della stessa dimensione, CRS e con la stessa trasformazione della Sentinel.
            output_transform = src_sentinel.transform
            output_crs = src_sentinel.crs
            patch_size_pixels_h = src_sentinel.height
            patch_size_pixels_w = src_sentinel.width
            
            # Controllo di coerenza con le dimensioni configurate
            if patch_size_pixels_h != config["patch_size_pixels"] or patch_size_pixels_w != config["patch_size_pixels"]:
                print(f"⚠️ Attenzione: Dimensioni immagine Sentinel ({patch_size_pixels_h}x{patch_size_pixels_w}) non corrispondono a config['patch_size_pixels'] ({config['patch_size_pixels']}x{config['patch_size_pixels']}). Userò le dimensioni della Sentinel.")
                
            print(f"   Profilo di output derivato da Sentinel: Dims {patch_size_pixels_h}x{patch_size_pixels_w}, CRS: {output_crs}")

    except Exception as e:
        print(f"❌ Errore apertura Sentinel per info spaziali '{os.path.basename(pre_sentinel_image_path)}': {e}")
        return False

    # 3. Estrai il punto di accensione dal GeoJSON
    try:
        # Trova la riga corrispondente all'incendio nel GeoJSON usando l'ID
        fire_feature = geojson_gdf[geojson_gdf['id'] == int(fire_id)]
        
        if fire_feature.empty:
            print(f"❌ Incendio ID {fire_id} non trovato nel GeoJSON. Impossibile creare ignition point raster.")
            return False
        
        # Estrai le coordinate 'point_x' e 'point_y' dal GeoJSON
        ignition_x_geojson = fire_feature['point_x'].iloc[0]
        ignition_y_geojson = fire_feature['point_y'].iloc[0]

        # Crea un GeoDataFrame per il punto di accensione usando il CRS del GeoJSON (EPSG:3857)
        gdf_ignition_geojson_crs = gpd.GeoDataFrame(
            geometry=[Point(ignition_x_geojson, ignition_y_geojson)],
            crs="EPSG:3857" # CRS esplicito del GeoJSON
        )
        
        # Converte il punto di accensione nel CRS target (lo stesso della Sentinel e dell'output)
        gdf_ignition_target_crs = gdf_ignition_geojson_crs.to_crs(output_crs)
        ignition_x_target_crs = gdf_ignition_target_crs.geometry.x.iloc[0]
        ignition_y_target_crs = gdf_ignition_target_crs.geometry.y.iloc[0]
        
        print(f"   Punto di ignizione (convertito al CRS della Sentinel): ({ignition_x_target_crs:.2f}, {ignition_y_target_crs:.2f})")

        # Converti le coordinate spaziali (geografiche) del punto di accensione
        # in coordinate pixel (riga, colonna) all'interno del frame 256x256
        # definito dalla trasformazione della Sentinel.
        row, col = rasterio.transform.rowcol(output_transform, ignition_x_target_crs, ignition_y_target_crs)
        
        # --- Crea l'immagine binaria temporanea con un singolo pixel a 1 ---
        # Questa sarà l'input per la distance transform.
        # Le dimensioni sono quelle lette dalla Sentinel.
        single_pixel_ignition_raster = np.zeros((patch_size_pixels_h, patch_size_pixels_w), dtype=np.uint8)

        # Se il punto di accensione cade all'interno del riquadro 256x256, impostalo a 1
        if 0 <= row < patch_size_pixels_h and 0 <= col < patch_size_pixels_w:
            single_pixel_ignition_raster[row, col] = 1
            print(f"   Punto di accensione posizionato a pixel ({row}, {col}) all'interno della patch allineata.")
        else:
            print(f"⚠️ Punto di accensione per incendio ID {fire_id} (x:{ignition_x_target_crs:.2f}, y:{ignition_y_target_crs:.2f}) FUORI DAI LIMITI del raster derivato dalla Sentinel. Il raster dell'ignition point sarà tutto zero.")
            # Questo caso è importante: se il punto di ignizione è fuori dal 256x256,
            # la mappa di distanza sarà tutta zero, il che è corretto per il modello.
        
        # --- Calcola la Distance Transform ---
        # distance_transform_edt(input) calcola la distanza di ogni pixel non-zero dai pixel zero.
        # Vogliamo la distanza dai pixel di background al punto di ignizione (che è 1 in single_pixel_ignition_raster).
        # Quindi, creiamo una maschera dove il punto di ignizione è 0 e il resto è 1 (1 - single_pixel_ignition_raster).
        # Questo darà un valore di 0 all'ignition point e valori crescenti allontanandosi.
        distance_map = distance_transform_edt(1 - single_pixel_ignition_raster).astype(np.float32)

        # Normalizza la mappa di distanza per avere valori tra 0 e 1, dove 1.0 è il punto di ignizione.
        max_dist_value = np.max(distance_map)
        if max_dist_value > 0:
            # Inverti la scala: 0 diventa max_dist_value, max_dist_value diventa 0.
            # Poi normalizza dividendo per max_dist_value.
            ignition_raster_final = (max_dist_value - distance_map) / max_dist_value
        else:
            # Se non c'era alcun punto di ignizione (single_pixel_ignition_raster era tutto zero),
            # la mappa di distanza normalizzata sarà anch'essa tutta zero.
            ignition_raster_final = np.zeros_like(distance_map)

        # 4. Definisci il profilo per il TIFF in output usando le info della Sentinel
        output_profile = {
            "height": patch_size_pixels_h,
            "width": patch_size_pixels_w,
            "count": 1, # Singola banda
            "dtype": ignition_raster_final.dtype, # Tipo di dato float32
            "crs": output_crs, # CRS della Sentinel
            "transform": output_transform, # Trasformazione della Sentinel
            "nodata": 0.0, # Valore di nodata per float
            "driver": "GTiff"
        }

        # 5. Salva il raster dell'ignition point come TIFF
        with rasterio.open(out_path_ignition_tif, "w", **output_profile) as dst:
            dst.write(ignition_raster_final, 1) # Scrive la banda 1
        
        print(f"🎉 Ignition point (distance map) TIFF salvato: {os.path.basename(out_path_ignition_tif)}")
        return True

    except Exception as e:
        print(f"❌ Errore durante la creazione del raster ignition point (distance map) per incendio ID {fire_id}: {e}")
        return False

# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # Configurazione generale del dataset e del processo
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson", 
        "root_dataset_folder": "piedmont_new", 
        "target_crs": "EPSG:32632", # Questo CRS sarà usato solo per la conversione del punto ignizione dal GeoJSON
                                    # Il CRS finale del TIFF ignition_point sarà quello della Sentinel di riferimento.
        "patch_size_pixels": 256,   # Usato come controllo, le dimensioni effettive derivano dalla Sentinel.
        "target_resolution_m": 10,  # Usato come controllo, la risoluzione effettiva deriva dalla Sentinel.
    }

    # Validazione della cartella radice del dataset
    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERRORE: La cartella radice del dataset '{config['root_dataset_folder']}' non esiste.")
        print("Assicurati che 'root_dataset_folder' nella configurazione punti alla directory dove si trovano le cartelle degli incendi (es. 'piedmont_new').")
        exit("Impossibile procedere.")
    
    # Caricamento del GeoJSON contenente i punti di ignizione
    print(f"Caricamento GeoJSON da: {config['geojson_path']}")
    try:
        main_geojson_gdf = gpd.read_file(config["geojson_path"])
        # Verifica la presenza delle colonne essenziali
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
    # Non chiamiamo get_processed_ids qui, perché il file log verrà eliminato prima.
    # processed_ids = get_processed_ids(log_file_path) # Questo non è più necessario all'inizio

    # --- Sezione per testare un singolo fire_id (ORA COMMENTATA) ---
    # chosen_fire_id = "7074" # <--- INSERISCI QUI IL FIRE_ID CHE VUOI TESTARE
    # print(f"\n--- TEST: Generazione ignition point raster per il singolo incendio ID: {chosen_fire_id} ---")
    # if chosen_fire_id in processed_ids and \
    #    os.path.exists(os.path.join(config["root_dataset_folder"], f"fire_{chosen_fire_id}", f"fire_{chosen_fire_id}_ignition_point.tif")):
    #     print(f"ℹ️ Incendio ID: {chosen_fire_id} già processato (nel log e file esistente). Skippo il re-generazione.")
    # else:
    #     success = generate_ignition_point_raster(chosen_fire_id, main_geojson_gdf, config) 
    #     if success:
    #         log_processed_id(log_file_path, chosen_fire_id) 
    #         print(f"🎉 Ignition point per incendio ID {chosen_fire_id} generato e registrato nel log.")
    #     else:
    #         print(f"❌ Processo ignition point per incendio ID {chosen_fire_id} fallito. Non aggiunto al log.")
    # print("\nScript di test terminato.")


    # --- Sezione per lanciare la generazione su TUTTI i fire_id (ORA DECOMMENTATA) ---
    all_fire_ids = []
    # Ricostruisce la lista di tutti gli ID degli incendi presenti nelle cartelle
    for item in sorted(os.listdir(config["root_dataset_folder"])):
        full_path = os.path.join(config["root_dataset_folder"], item)
        if os.path.isdir(full_path) and item.startswith("fire_"):
            fire_id_str = item.replace("fire_", "")
            all_fire_ids.append(fire_id_str)

    print(f'Trovate {len(all_fire_ids)} cartelle di incendi in: {config["root_dataset_folder"]}')
    
    # Dato che eliminerai il file di log manualmente, qui `processed_ids` sarà vuoto all'inizio di questa run.
    # Tuttavia, lo leggiamo comunque per il controllo interno del loop, nel caso tu non lo elimini
    # o nel caso di run future dove non vuoi rigenerare tutto.
    processed_ids = get_processed_ids(log_file_path) # Leggi qui gli ID processati

    print(f'Incendi con ignition point già processato (da log): {len(processed_ids)}')

    processed_count_current_run = 0
    # Loop su tutti gli ID degli incendi
    for fire_id_str in all_fire_ids:
        # Salta se già processato E il file di output esiste.
        # Dopo la cancellazione manuale, questo if sarà false per quasi tutti gli incendi,
        # forzando la rigenerazione.
        if fire_id_str in processed_ids and \
           os.path.exists(os.path.join(config["root_dataset_folder"], f"fire_{fire_id_str}", f"fire_{fire_id_str}_ignition_point.tif")):
            print(f"ℹ️ Incendio ID: {fire_id_str} già processato e file esistente. Skippo.")
            continue
        
        # Chiama la funzione per generare il raster dell'ignition point
        success = generate_ignition_point_raster(fire_id_str, main_geojson_gdf, config) 
        if success:
            processed_count_current_run += 1
            log_processed_id(log_file_path, fire_id_str) # Registra l'ID nel log
        else:
            print(f"Processo ignition point per incendio ID {fire_id_str} fallito. Non aggiunto al log.")
    
    print(f"\nElaborazione completata. Generati {processed_count_current_run} nuovi ignition point raster.")
    print(f"Gli ID degli incendi con ignition point generato sono stati registrati in: {log_file_path}")
    print("\nScript terminato.")