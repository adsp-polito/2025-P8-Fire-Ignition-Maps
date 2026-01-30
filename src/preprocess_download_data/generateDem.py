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
    """Compara due immagini NumPy array per somiglianza."""
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
    Scarica il DEM di Copernicus GLO-30 (30m) per un'area 256x256 pixel
    centrata sulla geometria dell'incendio e lo salva come file TIFF.
    Restituisce True in caso di successo, False altrimenti.
    """
    
    main_geojson_path = config["geojson_path"]
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    
    # Verifica se la cartella dell'incendio esiste già (requisito dell'utente)
    FIRE_SAVE_FOLDER = os.path.join(ROOT_DATASET_FOLDER, f"fire_{fire_id}")
    if not os.path.isdir(FIRE_SAVE_FOLDER):
        print(f"ℹ️ Cartella '{FIRE_SAVE_FOLDER}' non trovata. Nessuna azione intrapresa per incendio ID: {fire_id}.")
        return False 

    print(f"Caricamento GeoJSON da: {main_geojson_path}")
    gdf_all_fires = gpd.read_file(main_geojson_path)
    
    if "id" in gdf_all_fires.columns:
        print(f"DEBUG: Tipo di dato della colonna 'id' nel GeoJSON: {gdf_all_fires['id'].dtype}")
        print(f"DEBUG: Primi 5 ID nel GeoJSON: {gdf_all_fires['id'].head().tolist()}")
        print(f"DEBUG: Tipo di dato dell'ID cercato: {type(fire_id)} (valore: {fire_id})")
    else:
        print("DEBUG: AVVISO: Colonna 'id' non trovata nel GeoJSON.")

    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632") 
    if str(gdf_all_fires.crs) != TARGET_CRS_FOR_FIRES:
        print(f"Convertendo GeoJSON da {gdf_all_fires.crs} a {TARGET_CRS_FOR_FIRES} per i calcoli interni.")
        gdf_all_fires = gdf_all_fires.to_crs(TARGET_CRS_FOR_FIRES)

    fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
    
    if fire_data.empty:
        print(f"❌ Incendio con ID {fire_id} non trovato nel GeoJSON.")
        return False 

    fire = fire_data.iloc[0]
    gt_geometry_utm = shape(fire["geometry"])
    
    min_fire_area_sq_m = config.get("min_fire_area_sq_m", 200) 
    if gt_geometry_utm.area < min_fire_area_sq_m:
        print(f"❌ Incendio ID {fire_id} ha un'area troppo piccola ({gt_geometry_utm.area:.2f} mq) per essere utile a 30m di risoluzione. Minimo richiesto: {min_fire_area_sq_m} mq. Saltato.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return False

    TARGET_RESOLUTION_M = 30
    patch_size_pixels = config.get("patch_size_pixels", 256)
    
    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M 

    print(f"\n--- Processando DEM per Incendio ID: {fire_id} ---")
    print(f"Dimensione patch target (DEM): {patch_size_pixels}x{patch_size_pixels} pixel ({fixed_patch_size_meters/1000:.2f} km per lato) a {TARGET_RESOLUTION_M}m.")
    print(f"Cartella di output: {FIRE_SAVE_FOLDER}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search_bbox_wgs84 = gpd.GeoSeries([gt_geometry_utm.buffer(fixed_patch_size_meters/2)], crs=TARGET_CRS_FOR_FIRES).to_crs(epsg=4326).iloc[0].bounds
    print(f"Ricerca STAC con bbox (WGS84): {search_bbox_wgs84}")
    
    stac_collections = ["cop-dem-glo-30"]

    search = catalog.search(
        collections=stac_collections,
        bbox=search_bbox_wgs84,
        limit=50
    )

    items = list(search.items())
    if not items:
        print(f"❌ Nessun item DEM trovato per la regione {fire_id}.")
        print(f"\n--- FINE Processo DEM per incendio ID: {fire_id} ---")
        return False

    print(f"✅ Trovati {len(items)} item DEM.")
    
    # Seleziona il primo item DEM valido che interseca la geometria del fuoco
    dem_item_to_process = None
    for item_candidate in items:
        asset_name = "data"
        if asset_name not in item_candidate.assets:
            print(f"  Saltato item {item_candidate.id} - Asset 'data' non disponibile.")
            continue
        
        signed_href_check = pc.sign(item_candidate.assets[asset_name]).href
        try:
            with rasterio.open(signed_href_check) as src_check:
                item_bounds_polygon = gpd.GeoSeries([box(*src_check.bounds)], crs=src_check.crs).iloc[0]
                
                if str(src_check.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                    gt_geometry_in_item_crs_check = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(src_check.crs).iloc[0]
                else:
                    gt_geometry_in_item_crs_check = gt_geometry_utm

                if gt_geometry_in_item_crs_check.intersects(item_bounds_polygon):
                    print(f"  Item {item_candidate.id} interseca la geometria del fuoco. Verrà processato questo.")
                    dem_item_to_process = item_candidate
                    break # Trovato un item adatto, esci dal loop
                else:
                    print(f"  ❌ Item {item_candidate.id} non interseca la geometria del fuoco.")
        except Exception as e:
            print(f"  Errore durante il controllo dell'item {item_candidate.id}: {e}")
            
    if not dem_item_to_process:
        print("🔴 Nessuna delle immagini DEM trovate interseca la geometria dell'incendio. Impossibile procedere con il ritaglio.")
        print(f"\n--- FINE Processo DEM per incendio ID: {fire_id} ---")
        return False

    print(f"\n📸 Elaborazione DEM: {dem_item_to_process.id}.")
    
    dem_asset_key = 'data' 
    signed_href = pc.sign(dem_item_to_process.assets[dem_asset_key]).href
    try:
        with rasterio.open(signed_href) as src_dem:
            print(f"DEBUG: Source DEM CRS: {src_dem.crs}, Bounds: {src_dem.bounds}, NoData Value: {src_dem.nodata}")

            if str(src_dem.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                gt_geometry_in_dem_crs = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(src_dem.crs).iloc[0]
            else:
                gt_geometry_in_dem_crs = gt_geometry_utm
            
            centroid_x, centroid_y = gt_geometry_in_dem_crs.centroid.x, gt_geometry_in_dem_crs.centroid.y
            
            # Calcola i limiti della patch target nel CRS del DEM sorgente
            minx_patch_in_src_crs = centroid_x - (fixed_patch_size_meters / 2)
            miny_patch_in_src_crs = centroid_y - (fixed_patch_size_meters / 2)
            maxx_patch_in_src_crs = centroid_x + (fixed_patch_size_meters / 2)
            maxy_patch_in_src_crs = centroid_y + (fixed_patch_size_meters / 2)
            
            patch_bounds_in_src_crs = (minx_patch_in_src_crs, miny_patch_in_src_crs, maxx_patch_in_src_crs, maxy_patch_in_src_crs)

            # Calcola la finestra da leggere dal DEM sorgente usando i limiti nel suo CRS
            window_to_read = from_bounds(*patch_bounds_in_src_crs, transform=src_dem.transform)

            # Assicurati che la finestra sia all'interno dei limiti effettivi del DEM sorgente
            window_to_read = window_to_read.intersection(rasterio.windows.Window(0, 0, src_dem.width, src_dem.height))
            window_to_read = window_to_read.round_offsets(op='floor').round_lengths(op='ceil')

            print(f"DEBUG: Calculated window to read from source DEM: {window_to_read}")
            src_window_bounds_in_src_crs = rasterio.windows.bounds(window_to_read, src_dem.transform)
            print(f"DEBUG: Bounds of calculated window (in source DEM CRS): {src_window_bounds_in_src_crs}")

            if window_to_read.width == 0 or window_to_read.height == 0:
                print("🔴 ERRORE: La finestra di lettura calcolata per il DEM è vuota o invalida. Incendio forse troppo vicino al bordo della tile DEM o dati mancanti. Skip.")
                print(f"\n--- FINE Processo DEM per incendio ID: {fire_id} ---")
                return False

            raw_window_data = src_dem.read(1, window=window_to_read, masked=True).squeeze()
            
            fill_value = src_dem.nodata if src_dem.nodata is not None else -9999.0 
            
            if isinstance(raw_window_data, np.ma.MaskedArray):
                dem_array_for_reproject = raw_window_data.filled(fill_value=fill_value)
            else:
                dem_array_for_reproject = raw_window_data 
            
            print(f"DEBUG: Dati raw dalla finestra di lettura - Shape (after squeeze and fill): {dem_array_for_reproject.shape}, Dtype: {dem_array_for_reproject.dtype}")
            print(f"DEBUG: Data for reproject - Min: {dem_array_for_reproject.min()}, Max: {dem_array_for_reproject.max()}")

            src_window_transform = rasterio.windows.transform(window_to_read, src_dem.transform)
            print(f"DEBUG: Transform of raw_window_data: {src_window_transform}")

            # La trasformazione finale è basata sui limiti desiderati nel CRS di destinazione,
            # garantendo la risoluzione e l'allineamento corretti.
            final_transform = rasterio.transform.from_bounds(
                gt_geometry_utm.centroid.x - (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.y - (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.x + (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.y + (fixed_patch_size_meters / 2),
                width=patch_size_pixels, 
                height=patch_size_pixels
            )

            print(f"DEBUG: Final output transform: {final_transform}")
            print(f"DEBUG: Final output bounds (in target CRS {TARGET_CRS_FOR_FIRES}): {rasterio.transform.array_bounds(patch_size_pixels, patch_size_pixels, final_transform)}")

            dem_data_reprojected = np.zeros((patch_size_pixels, patch_size_pixels), dtype=np.float32)

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
            
            dem_filename = f"fire_{fire_id}_dem.tif" # Nome file fisso come richiesto
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
            print(f"💾 Salvato DEM (TIFF mono-banda a 30m): {out_path_dem_tif}")
            return True
    except Exception as e:
        print(f"❌ Errore critico durante la fase di download/elaborazione DEM per incendio ID {fire_id}: {e}")
        return False

def generate_dem_for_specific_fire(single_fire_id: str, config: dict):
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")

    print(f"\n--- Tentativo di generazione DEM per incendio ID: {single_fire_id} ---")
    print(f"Cercando la cartella: {fire_folder_path}")

    if os.path.isdir(fire_folder_path):
        print(f"✅ Cartella '{fire_folder_path}' trovata. Inizio generazione DEM.")
        success = process_single_dem_for_fire(single_fire_id, config) 
        if success:
            print(f"Processo DEM completato con successo per incendio ID: {single_fire_id}.")
        else:
            print(f"Processo DEM fallito per incendio ID: {single_fire_id}. Controllare i log di errore.")
    else:
        print(f"ℹ️ Cartella '{fire_folder_path}' non trovata. Nessuna azione intrapresa per incendio ID: {single_fire_id}.")
    print(f"--- Fine tentativo per incendio ID: {single_fire_id} ---\n")


if __name__ == '__main__':
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson",
        "root_dataset_folder": "piedmont_new", 
        "target_crs": "EPSG:32632", 
        "min_fire_area_sq_m": 200,
        "patch_size_pixels": 256,
    }

    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERRORE: La cartella radice del dataset '{config['root_dataset_folder']}' non esiste.")
        print("Assicurati che 'root_dataset_folder' nella configurazione punti alla directory dove si trovano le cartelle degli incendi (es. 'piedmont_new').")
        exit("Impossibile procedere.")
    
    # Correzione del SyntaxError: usate virgolette singole per l'f-string esterna
    print(f'Avvio elaborazione dei raster dem per tutti i folder in: {config["root_dataset_folder"]}')
    processed_count = 0
    
    # Il controllo os.path.exists(config["root_dataset_folder"]) è già stato fatto sopra,
    # quindi possiamo procedere direttamente con il listing dei contenuti.
    for item in os.listdir(config["root_dataset_folder"]):
        full_path = os.path.join(config["root_dataset_folder"], item)
        
        # Processa solo le sottocartelle che iniziano con "fire_"
        if os.path.isdir(full_path) and item.startswith("fire_"):
            # Estrai l'ID dell'incendio dal nome della cartella (es. "fire_6550" -> "6550")
            fire_id = int(item.replace("fire_", ""))
            
            # Chiamata corretta alla funzione process_single_dem_for_fire
            # Passa l'ID dell'incendio estratto e l'intero dizionario config
            success = process_single_dem_for_fire(fire_id, config)
            if success:
                processed_count += 1
    
    print(f"\nElaborazione completata. Generati raster per {processed_count} cartelle di incendio.")
    print("\nScript terminato.")