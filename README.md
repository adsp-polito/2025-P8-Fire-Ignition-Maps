# 2025-P8-Fire-Ignition-Maps


# 🔥 Multimodal Wildfire Burned Area Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Abstract:** This project implements a comparative deep learning framework to evaluate whether **pre-fire multimodal data** (satellite imagery, terrain, weather, and infrastructure) can predict final wildfire burned areas more accurately than traditional Sentinel-only baselines. The study highlights the impact of **encoder design** on segmentation performance and is optimized for efficiency on **CPU-constrained environments**.

---

## 📌 Project Objectives

The primary goals of this research are:

- **Predict** final burned areas using *only* **pre-fire information**.
- **Compare** a standard **Sentinel-only baseline** with a novel **multimodal deep learning model**.
- **Analyze** the effect of **encoder design choices** (e.g., ResNet vs. EfficientNet).
- **Estimate** the specific contribution of each **input modality** to prediction performance.
- **Produce** reproducible, slide-ready results for academic presentation.

---

## ❓ Research Questions

1.  **Feasibility:** Can pre-fire multimodal data predict final burned area with sufficient accuracy to support early emergency decisions?
2.  **Optimization:** Do auxiliary objectives (e.g., land-cover segmentation) improve burned-area prediction?
3.  **Feature Importance:** Which input data modalities contribute most to predictive performance?

---

## 🧠 Methodology Overview

This project contrasts two modeling approaches to isolate the value of data fusion.

### 1. Baseline Model
- **Architecture:** U-Net / FPN-style segmentation
- **Input:** Sentinel-2 imagery only (12 spectral bands)
- **Encoder:** **ResNet-34** (ImageNet pretrained)
- **Purpose:** Provide a strong, interpretable reference baseline.

### 2. Multimodal Model (Final Proposed Solution)
- **Architecture:** MultiModalFPN
- **Inputs:**
  - 🛰️ Sentinel-2 imagery
  - 🛰️ Landsat imagery
  - 🏔️ DEM + Road network rasters
  - ☁️ ERA5 Weather (Raster + Tabular)
  - 🔥 Ignition point map
- **Encoder:** **EfficientNet-B4** (ImageNet pretrained)
- **Fusion:** Attention-based feature fusion blocks
- **Purpose:** Leverage complementary pre-fire signals to improve segmentation accuracy.

---

## 🧩 Encoder Strategy

Two encoder strategies were strictly investigated:

1.  **Multiple encoders** with different capacities for each modality.
2.  **Single unified encoder** shared across all modalities.

**Decision:** The final model adopts a **unified EfficientNet-B4 encoder**, which achieved:
- Higher **IoU** and **F1 scores**.
- More **stable training** dynamics.
- Reduced architectural complexity.

---

## 📊 Results Summary

- The multimodal model **consistently outperforms** the Sentinel-only baseline.
- Encoder choice has a **significant impact** on segmentation quality.
- Multimodal inputs provide complementary information beyond optical imagery.
- Strong performance is achieved even under **CPU-only constraints** (MacBook Air).

> *Detailed quantitative comparisons and plots are stored in the `docs/` directory.*

---


## 🧪 Experimental Notes

To ensure transparency and reproducibility:

- **Hardware:** All experiments were conducted on **CPU (MacBook Air)**.
- **Optimization:** Auxiliary land-cover loss was disabled to avoid negative transfer.
- **Evaluation:** Some evaluations use reduced validation subsets for efficiency.
- **Reproducibility:** All results are clearly labeled in the `docs/` folder.

---

## ✅ Conclusion

This project shows that:

1. **Pre-fire multimodal data** can meaningfully improve burned-area prediction.
2. **Encoder design** is a critical performance factor.
3. A **unified high-capacity encoder** outperforms more complex multi-encoder setups.
4. Careful architectural choices enable strong results under **limited resources**.

---

## 👤 Authors

**Yousef Fayyaz**
**Parastoo Hashemi Alvar**

---

## 🗂️ Project Structure

```text
WildFire/
│
├── data/                       # Raw and processed datasets
├── geojson/                    # Vector data
│
├── docs/                       # Documentation & Analysis
│   ├── Maps_Graphs/            # Generated inference maps
│   ├── modality_ablation/      # Ablation study results
│   ├── model_comparison/       # Baseline vs Multimodal metrics
│   ├── slide_figures/          # Figures for presentation
│   └── output_result_paper_comparison.txt
│
├── inference/                  # Inference Scripts
│   ├── compare_baseline_vs_multimodal.py
│   ├── deploy_inference.py
│   ├── inference_2.py
│   └── inference_map.py
│
├── src/                        # Source Code
│   ├── Baseline_model/         # Baseline Implementation
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   ├── train.py
│   │   ├── main.py
│   │   └── unet_sentinel_best.pth
│   │
│   ├── checkpoint_2/           # Saved Models
│   │   └── best_model_3.pth
│   │
│   ├── figures_tables/         # Visualization Scripts
│   │   ├── export_slide_table.py
│   │   ├── make_qualitative_panels.py
│   │   ├── modality_ablation_quick.py
│   │   ├── paper_comparison.py
│   │   └── plot_threshold_sweep.py
│   │
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── main.py                 # Main Training Entry Point
│
├── inference_output/
├── runs/                       # TensorBoard Logs
│
├── .gitignore
├── .gitattributes
└── readme.md



-----







