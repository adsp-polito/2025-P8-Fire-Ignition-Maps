import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import numpy as np
import os
import re

# --- Configurazione Globale UNICA (MODIFICA QUESTO PERCORSO) ---
GLOBAL_ROADS_GEOJSON_PATH = "data/strade.geojson" # <-- MODIFICA QUESTO CON IL TUO PERCORSO REALE!

# CRS target per tutte le operazioni spaziali (il tuo EPSG:32632)
TARGET_CRS = "EPSG:32632"

# Carica e riproietta il GeoJSON delle strade una sola volta all'inizio dello script
strade_gdf = None # Inizializza a None, verrà caricato se il percorso è valido
try:
    if os.path.exists(GLOBAL_ROADS_GEOJSON_PATH):
        print(f"Caricamento e riproiezione di {GLOBAL_ROADS_GEOJSON_PATH} a {TARGET_CRS}...")
        strade_gdf = gpd.read_file(GLOBAL_ROADS_GEOJSON_PATH)
        strade_gdf = strade_gdf.to_crs(TARGET_CRS)
        # Crea un indice spaziale per query veloci (essenziale per grandi dataset)
        strade_gdf.sindex
        print("Strade GeoDataFrame caricato e riproiettato con successo e indice creato.")
    else:
        print(f"Errore: Il file strade.geojson non trovato al percorso: {GLOBAL_ROADS_GEOJSON_PATH}")
        print("Assicurati che 'GLOBAL_ROADS_GEOJSON_PATH' sia impostato correttamente.")
except Exception as e:
    print(f"Errore durante il caricamento o riproiezione del GeoJSON delle strade: {e}")
    strade_gdf = None

def generate_streets_raster_for_fire_folder(fire_folder_path: str):
    """
    Genera un raster delle strade (fire_id_streets.tif) per una data cartella di incendio.

    Args:
        fire_folder_path (str): Il percorso completo alla cartella di un singolo incendio (es. 'data/fire_5411').
    """
    if strade_gdf is None:
        print(f"  Impossibile elaborare '{os.path.basename(fire_folder_path)}': il GeoJSON delle strade non è stato caricato correttamente o non trovato.")
        return

    # Estrai il fire_ID dal nome della cartella
    fire_id = os.path.basename(fire_folder_path)
    print(f"\nElaborazione dell'incendio: {fire_id}")

    # --- 1. Trova un'immagine Sentinel di riferimento nella cartella ---
    # Qualsiasi file Sentinel .tif che non sia una maschera va bene per ottenere i riferimenti spaziali.
    sentinel_ref_path = None
    for fname in os.listdir(fire_folder_path):
        if "sentinel" in fname.lower() and fname.lower().endswith(".tif") and "_gt" not in fname.lower() and "_cm" not in fname.lower() and "_temp" not in fname.lower():
            sentinel_ref_path = os.path.join(fire_folder_path, fname)
            break
    
    if sentinel_ref_path is None:
        print(f"  Errore: Nessun file Sentinel .tif di riferimento trovato nella cartella '{fire_folder_path}'. Skipping.")
        return

    print(f"  Trovata immagine di riferimento Sentinel: {os.path.basename(sentinel_ref_path)}")

    # --- 2. Ottieni le proprietà spaziali dall'immagine Sentinel ---
    try:
        with rasterio.open(sentinel_ref_path) as src:
            img_bounds = src.bounds
            img_transform = src.transform
            img_width = src.width
            img_height = src.height
            img_crs = src.crs
            print(f"  Dimensioni immagine di riferimento: {img_width}x{img_height} pixel")
            print(f"  CRS immagine di riferimento: {img_crs}")

            if str(img_crs) != TARGET_CRS:
                print(f"  Attenzione: Il CRS dell'immagine Sentinel ({img_crs}) non corrisponde al CRS target ({TARGET_CRS}).")
                print("  Assicurati che tutte le immagini e i dati spaziali siano nel CRS corretto per un allineamento preciso.")

    except Exception as e:
        print(f"  Errore durante l'apertura o lettura delle proprietà dell'immagine di riferimento: {e}. Skipping.")
        return

    # --- 3. Filtra le strade che intersecano la bounding box dell'immagine ---
    bbox_polygon = Polygon([
        (img_bounds.left, img_bounds.bottom),
        (img_bounds.left, img_bounds.top),
        (img_bounds.right, img_bounds.top),
        (img_bounds.right, img_bounds.bottom)
    ])
    
    # Usa l'indice spaziale per una query efficiente
    possible_matches_index = list(strade_gdf.sindex.intersection(bbox_polygon.bounds))
    filtered_strade = strade_gdf.iloc[possible_matches_index].cx[img_bounds.left:img_bounds.right, img_bounds.bottom:img_bounds.top]
    
    print(f"  Trovate {len(filtered_strade)} geometrie stradali che intersecano l'area dell'immagine.")

    # --- 4. Rasterizzazione delle Strade ---
    if filtered_strade.empty:
        print(f"  Nessuna strada valida trovata nell'area dell'immagine per {fire_id}.")
        roads_raster = np.zeros((img_height, img_width), dtype=np.uint8)
    else:
        # Prepara le geometrie con i valori GP_RTP da rasterizzare
        shapes_to_rasterize = [(row.geometry, row['GP_RTP']) for idx, row in filtered_strade.iterrows()]

        roads_raster = rasterize(
            shapes=shapes_to_rasterize,
            out_shape=(img_height, img_width),
            transform=img_transform,
            all_touched=True, # Cattura tutti i pixel toccati dalle geometrie
            fill=0, # Valore di background per i pixel senza strade
            dtype=np.uint8 # Tipo di dato per il raster (0 per background, 1-5 per GP_RTP)
        )
    
    # --- 5. Salvataggio del Raster TIFF ---
    output_tif_path = os.path.join(fire_folder_path, f"{fire_id}_streets.tif")

    try:
        with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=img_height,
            width=img_width,
            count=1, # Una singola banda per il tipo di strada
            dtype=roads_raster.dtype,
            crs=img_crs, # Usa il CRS dell'immagine di riferimento
            transform=img_transform, # Usa la trasformazione dell'immagine di riferimento
        ) as dst:
            dst.write(roads_raster, 1) # Scrivi la banda 1 (il nostro array roads_raster)
        print(f"  Raster TIFF '{os.path.basename(output_tif_path)}' generato e salvato con successo.")
    except Exception as e:
        print(f"  Errore durante il salvataggio del raster TIFF delle strade: {e}. Il file potrebbe non essere stato creato.")
        # Non si esce con return qui, l'errore è stampato ma il loop può continuare per altri folder

# --- Blocco per il processing di TUTTI i folder "fire_*" ---
if __name__ == "__main__":
    # --- 1. IMPOSTA QUI IL PERCORSO ALLA DIRECTORY PRINCIPALE DEI TUOI DATI ---
    main_data_root = "piedmont_new" # <-- MODIFICA QUESTO!

    if os.path.exists(main_data_root):
        print(f"Avvio elaborazione dei raster delle strade per tutti i folder in: {main_data_root}")
        processed_count = 0
        for item in os.listdir(main_data_root):
            full_path = os.path.join(main_data_root, item)
            # Processa solo le sottocartelle che iniziano con "fire_"
            if os.path.isdir(full_path) and item.startswith("fire_"):
                generate_streets_raster_for_fire_folder(full_path)
                processed_count += 1
        print(f"\nElaborazione completata. Generati raster per {processed_count} cartelle di incendio.")
    else:
        print(f"Errore: La directory principale '{main_data_root}' non esiste. Controlla il percorso.")

    print("\nScript terminato.")