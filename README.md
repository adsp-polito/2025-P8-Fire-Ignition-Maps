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

## Methodology

### Model Architecture

The proposed architecture follows a **feature-level multimodal fusion strategy** designed to preserve high-resolution spatial detail while incorporating complementary environmental context.

- **Primary backbone:** Sentinel-2 imagery encoded using **EfficientNet-B4**
- **Auxiliary inputs:**  
  - Landsat-8 imagery  
  - Digital Elevation Model (DEM)  
  - Road / human infrastructure density maps  
  - ERA5 meteorological variables (temperature, wind)
- **Fusion strategy:** Auxiliary modalities are projected and fused at **multiple encoder scales**
- **Decoder:** Feature Pyramid Network (FPN)
- **Outputs:**
  - Main head: burned-area probability map  
  - Auxiliary head: land-cover classification (training only)

The auxiliary task improves spatial coherence and reduces physically implausible predictions.

---

### Architecture Diagram

<p align="center">
  <img src="documents/model_architecture.png" width="800"/>
</p>

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
├── data/
│   └── geojson/
│
├── documents/
│   ├── model_architecture.png
│   ├── inference_checkpoint_3_results/
│   ├── Maps_Graphs_checkpoint_1/
│   ├── modality_ablation_checkpoint_3/
│   ├── model_comparison_baseline_multimodal_checkpoint_3/
│   ├── multimodal_auxiliary_final_checkpoint/
│   └── output_result_paper_comparison.txt
│
├── inference/
│
├── report/
│   ├── Checkpoint_1/
│   ├── Checkpoint_2/
│   ├── Checkpoint_3/
│   ├── checkpoint_final/
│   └── paper_final/
│
├── src/
│   ├── baseline_singlemodal/
│   ├── multimodal/
│   ├── multimodal_auxiliary/
│   ├── preprocess_download_data/
│   ├── inference_output/
│   └── figures_tables/
│
├── requirements.txt
├── README.md



-----







