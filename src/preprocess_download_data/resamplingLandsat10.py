import os
import rasterio
import numpy as np
import cv2
from rasterio.warp import Resampling
# Assicurati che il percorso sia corretto per le tue funzioni utils
from modelWithLandsat.utils import resample_image, read_image_and_metadata

# --- Configurazione Generale ---
# Questa è la directory radice che contiene tutte le tue cartelle di fuoco (es. fire_6578, fire_xyz, etc.)
BASE_DATA_DIR = 'piedmont_new' 

# Suffisso da aggiungere ai nomi dei file Landsat risampled.
# Esempio: 'fire_6578_2022-02-06_pre_landsat_2.tif' diventerà 'fire_6578_2022-02-06_pre_landsat_2_10m.tif'
RESAMPLED_SUFFIX = '_10m.tif'

def find_images_by_criteria(directory, file_type='sentinel'):
    """
    Trova i file immagine basandosi su criteri specifici.
    Args:
        directory (str): Il percorso della directory in cui cercare.
        file_type (str): 'sentinel' per pre_sentinel.tif (non CM), 'landsat' per pre_landsat.tif (non CM, non risampled).
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
            elif file_type == 'landsat':
                # Criteri per Landsat: contiene "pre_landsat", termina con ".tif", NON contiene "_CM", NON contiene il suffisso di resampling
                if "pre_landsat" in f and f.endswith(".tif") and "_CM" not in f and RESAMPLED_SUFFIX not in f:
                    found_files.append(os.path.join(root, f))
    return found_files

def process_fire_folder(fire_dir_path):
    """
    Processa una singola cartella di fuoco: trova Sentinel e tutte le Landsat,
    e risample ogni Landsat a Sentinel.
    """
    print(f"\nProcessing folder: {fire_dir_path}")

    # Trova il percorso dell'immagine Sentinel di riferimento (ne prendiamo solo una)
    sentinel_paths = find_images_by_criteria(fire_dir_path, file_type='sentinel')
    if not sentinel_paths:
        print(f"  ATTENZIONE: Nessun file 'pre_sentinel' (non CM) trovato in {fire_dir_path}. Salto questa cartella.")
        return
    
    # Prendi la prima Sentinel trovata come riferimento per i metadati
    sentinel_reference_path = sentinel_paths[0]

    # Trova tutti i percorsi delle immagini Landsat originali
    landsat_paths = find_images_by_criteria(fire_dir_path, file_type='landsat')
    if not landsat_paths:
        print(f"  ATTENZIONE: Nessun file 'pre_landsat' (non CM, non risampled) trovato in {fire_dir_path}. Salto questa cartella o non eseguo il resampling Landsat.")
        return

    try:
        # Carica i metadati di Sentinel (il nostro riferimento a 10m)
        _, sentinel_transform, sentinel_crs, sentinel_shape, sentinel_res, _ = read_image_and_metadata(sentinel_reference_path)
        print(f"  Sentinel reference: shape={sentinel_shape}, res={sentinel_res}m, path={os.path.basename(sentinel_reference_path)}")

        # Processa ogni immagine Landsat trovata
        for landsat_path in landsat_paths:
            original_landsat_filename = os.path.basename(landsat_path)
            landsat_output_filename = original_landsat_filename.replace('.tif', RESAMPLED_SUFFIX)
            output_landsat_10m_path = os.path.join(os.path.dirname(landsat_path), landsat_output_filename)

            # Controlla se il file risampled esiste già per evitare di rifare il lavoro
            if os.path.exists(output_landsat_10m_path):
                print(f"  File Landsat risampled '{os.path.basename(output_landsat_10m_path)}' esiste già. Salto il resampling per questo file.")
                continue # Passa al prossimo file Landsat

            # Carica i metadati di Landsat (l'immagine da risample)
            _, landsat_transform, landsat_crs, landsat_shape, landsat_res, _ = read_image_and_metadata(landsat_path)
            print(f"  Landsat original: shape={landsat_shape}, res={landsat_res}m, path={original_landsat_filename}")

            # Esegui l'UPsampling di Landsat a 10m
            print(f"  Eseguo l'UPsampling di Landsat a 10m per {original_landsat_filename} -> {landsat_output_filename}")
            resample_image(
                input_tif_path=landsat_path,
                output_tif_path=output_landsat_10m_path,
                target_transform=sentinel_transform,
                target_crs=sentinel_crs,
                target_shape=sentinel_shape # Passa (Height, Width)
            )
            print(f"  Landsat risampled salvato in: {output_landsat_10m_path}")

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