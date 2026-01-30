import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from cloudsen12_models.cloudsen12 import load_model_by_name, COLORS_CLOUDSEN12
from PIL import Image

# === Dati utente: Ordine esatto delle tue 15 bande Landsat nel file TIFF ===
# QUESTO È FONDAMENTALE! Deve corrispondere all'ordine reale delle bande nel tuo TIFF.
user_landsat_bands_in_order = [
    'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11', 'coastal',
    'atran', 'cdist', 'drad', 'emis', 'emsd', 'trad', 'urad'
]

# Crea una mappatura dal nome della banda dell'utente all'indice 0-based nel file sorgente TIFF
user_band_name_to_src_index = {
    name: idx for idx, name in enumerate(user_landsat_bands_in_order)
}

# === Configurazioni dei Modelli CloudSEN12 ===
model_configs = {
    "dtacs4bands": {
        "expected_bands": ["B08", "B04", "B03", "B02"], # Ordine specifico per il modello
        "band_mapping": { # Nome_banda_modello: Nome_banda_Landsat_utente_da_prelevare
            'B08': 'nir08', # S2 B8 NIR -> L8 B5 NIR
            'B04': 'red',   # S2 B4 Red -> L8 B4 Red
            'B03': 'green', # S2 B3 Green -> L8 B3 Green
            'B02': 'blue'   # S2 B2 Blue -> L8 B2 Blue
        }
    }
}

# === Funzione per preparare l'input Landsat per un dato modello ===
def prepare_landsat_input(src_dataset, model_config, user_band_name_to_src_index):
    """
    Prepara l'array di input Landsat per un modello specifico di rilevamento nuvole,
    selezionando e riordinando le bande e gestendo le mancanti.

    Args:
        src_dataset (rasterio.DatasetReader): Il dataset rasterio aperto dell'immagine Landsat.
        model_config (dict): Configurazione per il modello (expected_bands, band_mapping).
        user_band_name_to_src_index (dict): Mappa i nomi delle bande dell'utente ai loro indici
                                             0-based nel dataset rasterio.

    Returns:
        np.ndarray: Array preparato (C, H, W) pronto per la predizione del modello.
        dict: Metadati aggiornati per la scrittura dell'output.
    """
    expected_bands_model = model_config["expected_bands"]
    band_mapping_model = model_config["band_mapping"]

    height = src_dataset.height
    width = src_dataset.width
    
    bands_for_model = []

    print(f"DEBUG: Preparing input for model expecting {len(expected_bands_model)} bands.")

    for model_band_name in expected_bands_model:
        landsat_user_band_name = band_mapping_model.get(model_band_name)
        
        print(f"DEBUG: Processing model band '{model_band_name}'...")
        if landsat_user_band_name is None: 
            print(f"ATTENZIONE: Banda '{model_band_name}' richiesta dal modello ma non ha un match diretto in Landsat. Creazione banda di zeri.")
            band_data = np.zeros((height, width), dtype=np.float32)
            bands_for_model.append(band_data)
        elif landsat_user_band_name in user_band_name_to_src_index:
            src_band_idx = user_band_name_to_src_index[landsat_user_band_name]
            print(f"DEBUG: Mapping model band '{model_band_name}' to user Landsat band '{landsat_user_band_name}' at source index {src_band_idx + 1}.")
            band_data = src_dataset.read(src_band_idx + 1).astype(np.float32)
            bands_for_model.append(band_data)
        else:
            raise ValueError(f"ERRORE: La banda Landsat '{landsat_user_band_name}' (per il modello '{model_band_name}') non è stata trovata nella lista delle bande fornite dall'utente. Verifica 'user_landsat_bands_in_order'.")
        print(f"DEBUG: Current band_data shape: {band_data.shape}")
    
    print(f"DEBUG: Total bands collected in bands_for_model: {len(bands_for_model)}")
    if not bands_for_model:
        raise ValueError("No bands were collected for the model input. This should not happen.")

    arr = np.stack(bands_for_model)
    print(f"DEBUG: Shape of stacked array (C, H, W) before normalization: {arr.shape}")
    
    arr = arr / 10_000.0
    print(f"DEBUG: Shape of array after normalization: {arr.shape}")

    meta = src_dataset.meta.copy()
    meta.update({
        'count': arr.shape[0], # Ensure meta count matches actual prepared bands count
        'dtype': np.float32 
    })
    return arr, meta

# === Esecuzione principale ===
# Percorsi di esempio per i tuoi file TIFF Landsat. AGGIORNA CON I TUOI PERCORSI REALI!
landsat_tif_paths = [
    # Esempio per un file Landsat a 30m (se lo hai)
    "piedmont_new/fire_6806/fire_6806_2022-07-17_pre_landsat_1.tif",  
]

output_dir = "la_cloud"
os.makedirs(output_dir, exist_ok=True) # Crea la cartella "landsat_cloud" se non esiste

for tif_path in landsat_tif_paths:
    if not os.path.exists(tif_path):
        print(f"Skipping {tif_path}: File non trovato. Aggiorna i percorsi reali dei tuoi TIFF Landsat.")
        continue

    print(f"\n--- Elaborazione del file: {os.path.basename(tif_path)} ---")

    for model_name, config in model_configs.items():
        print(f"** Tentativo con il modello: {model_name} **")
        
        out_tif_base = os.path.basename(tif_path).replace(".tif", f"_{model_name}_CM.tif")
        out_tif_path = os.path.join(output_dir, out_tif_base)
        
        try:
            with rasterio.open(tif_path) as src:
                # Debugging: Controlla il numero di bande nel TIFF sorgente
                print(f"DEBUG: Source TIFF '{os.path.basename(tif_path)}' has {src.count} bands.")
                print(f"DEBUG: Source TIFF band descriptions: {src.descriptions}")

                # 1. Prepara l'array di input per il modello
                prepared_arr, meta_out = prepare_landsat_input(src, config, user_band_name_to_src_index)
                
                # Debugging: Controlla la forma prima di passare al modello
                print(f"DEBUG: Shape of prepared_arr before passing to model: {prepared_arr.shape}")
                
                # 2. Carica il modello CloudSEN12
                print(f"Caricamento del modello '{model_name}'...")
                model = load_model_by_name(model_name)
                print("Modello caricato.")

                # 3. Esegui la predizione della maschera nuvole
                print("Esecuzione della predizione della maschera nuvole...")
                # CORREZIONE CRUCIALE: Passa l'array (C, H, W) direttamente al modello, senza aggiungere la dimensione del batch
                mask = model.predict(prepared_arr) 
                print("Predizione completata.")
                
                # Il risultato della predizione potrebbe essere (H, W) o (1, H, W). Assicuriamoci sia (H, W).
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0) 
                elif mask.ndim != 2:
                    raise ValueError(f"Formato inatteso della maschera di output: {mask.shape}. Previsto (H, W) o (1, H, W).")


                # 4. Scrivi la maschera in formato TIFF
                mask_tif = mask.astype(rasterio.uint8)

                meta_out.update({
                    'count': 1,                 # Una singola banda per la maschera
                    'dtype': rasterio.uint8,    # Tipo di dato uint8
                    'nodata': 255           # Valore NoData (opzionale, ma utile)
                })
                with rasterio.open(out_tif_path, 'w', **meta_out) as dst:
                    dst.write(mask_tif, 1) 
                print(f"✔ Maschera nuvole TIFF salvata in: {out_tif_path}")

                # === Generazione PNG della maschera a colori (Opzionale) ===
                # Decommenta il blocco seguente se vuoi generare anche l'immagine PNG colorata
                '''
                colored_mask = (COLORS_CLOUDSEN12[mask] * 255).astype(np.uint8)
                img = Image.fromarray(colored_mask, 'RGB')
                img.save(out_png_path)
                print(f"✔ Immagine PNG della maschera nuvole a colori salvata in: {out_png_path}")
                '''

        except Exception as e:
            print(f"ERRORE durante l'elaborazione del file {os.path.basename(tif_path)} con il modello {model_name}: {e}")
            continue # Continua con il prossimo modello/file anche se uno fallisce