import os
import rasterio
import numpy as np
import cv2
from rasterio.warp import Resampling
# Assicurati che il percorso sia corretto per le tue funzioni utils
from modelWithLandsat.utils import resample_image, read_image_and_metadata

# --- Configurazione Generale ---
BASE_DATA_DIR = 'piedmont_new' 

# Suffisso da aggiungere ai nomi dei file dem risampled.

RESAMPLED_SUFFIX = '_10m.tif'

def find_images_by_criteria(directory, file_type='sentinel'):
    """
    Trova i file immagine basandosi su criteri specifici.
    Args:
        directory (str): Il percorso della directory in cui cercare.
        file_type (str): 'sentinel' per pre_sentinel.tif (non CM), 'dem' per pre_dem.tif (non CM, non risampled).
    Returns:
        list: Una lista di percorsi completi dei file trovati.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if file_type == 'sentinel':
                # Criteri per Sentinel: contiene "pre_sentinel", termina con ".tif", NON contiene "_CM", NON contiene il suffisso di resampling
                if "pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f and RESAMPLED_SUFFIX not in f:
                    found_files.append(os.path.join(root, f))
                    # Per Sentinel, ne vogliamo solo una come riferimento, quindi possiamo fermarci al primo match
                    return found_files 
            elif file_type == 'dem':
                # Criteri per dem: contiene "dem", termina con ".tif", NON contiene "_CM", NON contiene il suffisso di resampling
                if "dem" in f and f.endswith(".tif") and RESAMPLED_SUFFIX not in f:
                    found_files.append(os.path.join(root, f))
    return found_files

def process_fire_folder(fire_dir_path):
    """
    Processa una singola cartella di fuoco: trova Sentinel e tutte le dem,
    e risample ogni dem a Sentinel.
    """
    print(f"\nProcessing folder: {fire_dir_path}")

    # Trova il percorso dell'immagine Sentinel di riferimento (ne prendiamo solo una)
    sentinel_paths = find_images_by_criteria(fire_dir_path, file_type='sentinel')
    if not sentinel_paths:
        print(f"  ATTENZIONE: Nessun file 'pre_sentinel' (non CM) trovato in {fire_dir_path}. Salto questa cartella.")
        return
    
    # Prendi la prima Sentinel trovata come riferimento per i metadati
    sentinel_reference_path = sentinel_paths[0]

    # Trova tutti i percorsi delle immagini dem originali
    dem_paths = find_images_by_criteria(fire_dir_path, file_type='dem')
    if not dem_paths:
        print(f"  ATTENZIONE: Nessun file 'dem' (non CM, non risampled) trovato in {fire_dir_path}. Salto questa cartella o non eseguo il resampling dem.")
        return

    try:
        # Carica i metadati di Sentinel (il nostro riferimento a 10m)
        _, sentinel_transform, sentinel_crs, sentinel_shape, sentinel_res, _ = read_image_and_metadata(sentinel_reference_path)
        print(f"  Sentinel reference: shape={sentinel_shape}, res={sentinel_res}m, path={os.path.basename(sentinel_reference_path)}")

        # Processa ogni immagine dem trovata
        for dem_path in dem_paths:
            original_dem_filename = os.path.basename(dem_path)
            dem_output_filename = original_dem_filename.replace('.tif', RESAMPLED_SUFFIX)
            output_dem_10m_path = os.path.join(os.path.dirname(dem_path), dem_output_filename)

            # Controlla se il file risampled esiste già per evitare di rifare il lavoro
            if os.path.exists(output_dem_10m_path):
                print(f"  File dem risampled '{os.path.basename(output_dem_10m_path)}' esiste già. Salto il resampling per questo file.")
                continue # Passa al prossimo file dem

            # Carica i metadati di dem (l'immagine da risample)
            _, dem_transform, dem_crs, dem_shape, dem_res, _ = read_image_and_metadata(dem_path)
            print(f" Dem original: shape={dem_shape}, res={dem_res}m, path={original_dem_filename}")

            # Esegui l'UPsampling di dem a 10m
            print(f"  Eseguo l'UPsampling di dem a 10m per {original_dem_filename} -> {dem_output_filename}")
            resample_image(
                input_tif_path=dem_path,
                output_tif_path=output_dem_10m_path,
                target_transform=sentinel_transform,
                target_crs=sentinel_crs,
                target_shape=sentinel_shape # Passa (Height, Width)
            )
            print(f"  dem risampled salvato in: {output_dem_10m_path}")

    except Exception as e:
        print(f"  Errore durante il processamento della cartella {fire_dir_path}: {e}")

# --- Main Execution ---
print(f"Inizio il processo di pre-resampling per le cartelle in '{BASE_DATA_DIR}'...")

# Itera su tutte le sottocartelle in BASE_DATA_DIR
for item_name in os.listdir(BASE_DATA_DIR):
    full_path = os.path.join(BASE_DATA_DIR, item_name)
    if os.path.isdir(full_path) and item_name.startswith('fire_'): # Filtra solo le cartelle "fire_"
        process_fire_folder(full_path)

print("\nProcesso di pre-resampling completato per tutte le cartelle dei fuochi.")