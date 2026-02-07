import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from cloudsen12_models.cloudsen12 import load_model_by_name, COLORS_CLOUDSEN12
from PIL import Image

# data: exact order of the 15 Landsat bands in the TIFF file ===
# CRITICAL! It must match the actual band order in the TIFF.
user_landsat_bands_in_order = [
    'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11', 'coastal',
    'atran', 'cdist', 'drad', 'emis', 'emsd', 'trad', 'urad'
]

# Create a mapping from the band name to the 0-based index in the source TIFF file
user_band_name_to_src_index = {
    name: idx for idx, name in enumerate(user_landsat_bands_in_order)
}

# CloudSEN12 Model Configurations
model_configs = {
    "dtacs4bands": {
        "expected_bands": ["B08", "B04", "B03", "B02"], # Specific order required by the model
        "band_mapping": { # model_band_name: id_landsat_band_name_to_extract
            'B08': 'nir08', # S2 B8 NIR -> L8 B5 NIR
            'B04': 'red',   # S2 B4 Red -> L8 B4 Red
            'B03': 'green', # S2 B3 Green -> L8 B3 Green
            'B02': 'blue'   # S2 B2 Blue -> L8 B2 Blue
        }
    }
}

# Function to prepare Landsat input for a given model
def prepare_landsat_input(src_dataset, model_config, user_band_name_to_src_index):
    """
        Prepares the Landsat input array for a specific cloud-detection model by
        selecting and reordering bands and handling missing ones.

        Args:
            src_dataset (rasterio.DatasetReader): The opened rasterio dataset for the Landsat image.
            model_config (dict): Model configuration (expected_bands, band_mapping).
            user_band_name_to_src_index (dict): Maps user band names to their 0-based indices
                in the rasterio dataset.

        Returns:
            np.ndarray: Prepared array (C, H, W) ready for model prediction.
            dict: Updated metadata for writing the output.
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
            print(f"WARNING: Banda '{model_band_name}' Required by the model but has no direct match in Landsat. Creating a zero-filled band.")
            band_data = np.zeros((height, width), dtype=np.float32)
            bands_for_model.append(band_data)
        elif landsat_user_band_name in user_band_name_to_src_index:
            src_band_idx = user_band_name_to_src_index[landsat_user_band_name]
            print(f"DEBUG: Mapping model band '{model_band_name}' to user Landsat band '{landsat_user_band_name}' at source index {src_band_idx + 1}.")
            band_data = src_dataset.read(src_band_idx + 1).astype(np.float32)
            bands_for_model.append(band_data)
        else:
            raise ValueError(f"ERROR: The Landsat band '{landsat_user_band_name}' (required for model '{model_band_name}') was not found in the list of bands provided by the user. Check 'user_landsat_bands_in_order'.")
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

# === Main execution ===
# Example paths for your Landsat TIFF files. UPDATE THESE WITH YOUR REAL PATHS!
landsat_tif_paths = [
    # Example for a 30 m Landsat file (if available)
    "piedmont_new/fire_6806/fire_6806_2022-07-17_pre_landsat_1.tif",  
]

output_dir = "la_cloud"
os.makedirs(output_dir, exist_ok=True) # Create the "landsat_cloud" directory if it does not exist

for tif_path in landsat_tif_paths:
    if not os.path.exists(tif_path):
        print(f"Skipping {tif_path}: File not found. Update the actual paths to your Landsat TIFF files.")
        continue

    print(f"\n--- Processing file: {os.path.basename(tif_path)} ---")

    for model_name, config in model_configs.items():
        print(f"** Attempting with model: {model_name} **")

        out_tif_base = os.path.basename(tif_path).replace(".tif", f"_{model_name}_CM.tif")
        out_tif_path = os.path.join(output_dir, out_tif_base)
        
        try:
            with rasterio.open(tif_path) as src:
                # Debugging
                print(f"DEBUG: Source TIFF '{os.path.basename(tif_path)}' has {src.count} bands.")
                print(f"DEBUG: Source TIFF band descriptions: {src.descriptions}")

                # 1. Prepare the input array for the model
                prepared_arr, meta_out = prepare_landsat_input(src, config, user_band_name_to_src_index)
                
                # Debugging: Check the shape before passing to the model
                print(f"DEBUG: Shape of prepared_arr before passing to model: {prepared_arr.shape}")

                # 2. Load the CloudSEN12 model
                print(f"Loading model '{model_name}'...")
                model = load_model_by_name(model_name)
                print("Model loaded.")

                # 3. Run cloud mask prediction
                print("Running cloud mask prediction...")
                # CRITICAL FIX: Pass the array (C, H, W) directly to the model, without adding a batch dimension
                mask = model.predict(prepared_arr)
                print("Prediction completed.")

                # The prediction output may be (H, W) or (1, H, W). Ensure it is (H, W).
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.ndim != 2:
                    raise ValueError(
                        f"Unexpected output mask format: {mask.shape}. Expected (H, W) or (1, H, W)."
                    )

                # 4. Write the mask to TIFF format
                mask_tif = mask.astype(rasterio.uint8)

                meta_out.update({
                    'count': 1,  # Single band for the mask
                    'dtype': rasterio.uint8,  # uint8 data type
                    'nodata': 255  # NoData value (optional but useful)
                })
                with rasterio.open(out_tif_path, 'w', **meta_out) as dst:
                    dst.write(mask_tif, 1)
                print(f"✔ Cloud mask TIFF saved to: {out_tif_path}")

                # === Generate colored PNG mask (Optional) ===
                # Uncomment the block below if you also want to generate a colored PNG image
                '''
                colored_mask = (COLORS_CLOUDSEN12[mask] * 255).astype(np.uint8)
                img = Image.fromarray(colored_mask, 'RGB')
                img.save(out_png_path)
                print(f"✔ Colored cloud mask PNG saved to: {out_png_path}")
                '''

        except Exception as e:
            print(
                f"ERROR while processing file {os.path.basename(tif_path)} "
                f"with model {model_name}: {e}"
            )
            continue  # Continue with the next model/file even if one fails