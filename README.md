# MTFINet: Multi-Task Fluorescence Intelligence Network for Heavy-Metal Determination

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: IEEE Submission](https://img.shields.io/badge/Submission-IEEE%20Sensors%20Journal-brightgreen.svg)]()

Official PyTorch implementation of the **Multi-Task Fluorescence Intelligence Network (MTFINet)** for simultaneous heavy-metal identification and concentration-class determination from paper-based fluorescent carbon-dot images.

---

## 📌 Overview

Paper-based analytical devices ($\mu\text{PADs}$) using fluorescent carbon-dot probes provide low-cost sensing platforms for water contamination screening. However, traditional scalar optical detectors (cuvette fluorometers, single-channel photodiodes) discard 2D spatial quenching heterogeneity (coffee-ring patterns, radial gradients, paper fiber textures). 

**MTFINet** introduces an intelligent image-based readout architecture based on a shared **ResNet-18** convolutional backbone and dual specialized linear heads:
1. **Metal Identification Head:** Categorical classification across 8 toxic heavy-metal species (**Pb, Hg, Cd, Cr, Cu, Zn, Ag, Pd**).
2. **Concentration Estimation Head:** Discrete ordinal classification across 7 nominal concentration tiers (**0, 10, 50, 100, 150, 200, 250 µM**).

By learning shared latent representations from spatial fluorescence response patterns, MTFINet eliminates redundant single-task architectures while improving prediction stability and classification accuracy.

---

## 📊 Performance Summary on Held-Out Test Set ($N = 818$ Independent Samples)

The dataset was partitioned using a strict **leakage-safe grouped split** (`GroupShuffleSplit`, random seed 42) based on unique physical sensor spots before any augmentation, ensuring zero data leakage across partitions.

| Evaluation Task | Metric | Test Score | Description |
| :--- | :--- | :---: | :--- |
| **Metal Identification** | **Classification Accuracy** | **92.67%** | Exact species match across 8 classes |
| | **Total Error of Classification (TEC)** | **7.33%** | $100\% - \text{Accuracy}$ |
| | **Macro Precision** | **94.56%** | Unweighted mean precision across metals |
| | **Macro Recall** | **93.42%** | Unweighted mean recall across metals |
| | **Macro F1-Score** | **93.61%** | Harmonic mean of precision and recall |
| **Concentration Ranking** | **Classification Accuracy** | **97.43%** | Exact class match across 7 nominal tiers |
| | **Total Error of Classification (TEC)** | **2.57%** | $100\% - \text{Accuracy}$ |
| | **Mapped-Class MAE (QER)** | **2.00 µM** | $\frac{1}{N}\sum \|C(y_i) - C(\hat{y}_i)\|$ across 0–250 µM range |
| | **Quadratic Weighted Kappa (QWK)** | **0.9850** | Inter-rater ordinal agreement metric |

### Per-Metal Test Performance:
* **Chromium (Cr):** 100% Precision | 100% Recall | 1.0000 F1
* **Lead (Pb):** 100% Precision | 100% Recall | 1.0000 F1
* **Cadmium (Cd):** 99.04% Precision | 96.26% Recall | 0.9763 F1
* **Silver (Ag):** 95.18% Precision | 98.75% Recall | 0.9693 F1
* **Palladium (Pd):** 100% Precision | 87.96% Recall | 0.9360 F1
* **Mercury (Hg):** 95.45% Precision | 84.68% Recall | 0.8974 F1
* **Zinc (Zn):** 94.90% Precision | 83.04% Recall | 0.8857 F1
* **Copper (Cu):** 71.88% Precision | 96.64% Recall | 0.8244 F1

---

## 📁 Repository Structure

```
MTFINet/
├── README.md                          <- Project overview, documentation & results
├── LICENSE                            <- MIT Open Source License
├── requirements.txt                   <- Environment dependencies
├── .gitignore                         <- Excludes raw image datasets from version control
├── models/
│   ├── __init__.py
│   ├── mtfinet.py                     <- Core MTFINet architecture (MultiTaskResNet)
│   └── baselines.py                   <- Single-task and mobile baseline architectures
├── dataset/
│   ├── __init__.py
│   ├── dataloader.py                  <- Leakage-safe DataLoader with train-only transforms
│   └── dataset_manifest.csv           <- Master manifest (5,296 images, split assignments)
├── scripts/
│   ├── train.py                       <- Training script with validation checkpointing
│   ├── evaluate.py                    <- Held-out test set evaluation & report generator
│   ├── plot_spectrometer.py           <- Plots JASCO FP-8550 emission spectra
│   └── plot_quenching_curves.py       <- Extracts image-derived intensity curves
├── notebooks/
│   └── MTFINet_Demo.ipynb             <- Step-by-step walkthrough & interactive evaluation
├── results/
│   ├── test_metrics_summary.json      <- Machine-readable test metric summary
│   ├── test_set_predictions.csv       <- Aligned test sample predictions
│   ├── figures/
│   │   ├── leakage_free_dual_confusion_matrix.png
│   │   ├── jasco_fp8550_emission_spectrum.png
│   │   └── image_optical_quenching_curves.png
│   └── spectrometer_data/
│       └── jasco_fp8550_data.txt      <- Raw spectrofluorometer measurements (VIT Vellore)
└── checkpoints/
    ├── README.md                      <- Model loading guide & architecture details
    └── best_multitask_model_leakage_free.pth  <- Trained PyTorch weights (42.7 MB)
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/<your-username>/MTFINet.git
cd MTFINet
pip install -r requirements.txt
```

### 2. Evaluate Pretrained Model
To evaluate the saved model on the test split and reproduce the confusion matrices and metrics:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_multitask_model_leakage_free.pth \
    --manifest dataset/dataset_manifest.csv
```

### 3. Re-Train MTFINet from Scratch
```bash
python scripts/train.py \
    --manifest dataset/dataset_manifest.csv \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0001 \
    --save-path checkpoints/best_multitask_model_leakage_free.pth
```

### 4. Plot Spectrofluorometer Reference Data
To generate the calibrated JASCO FP-8550 photoluminescence emission plot:
```bash
python scripts/plot_spectrometer.py
```

---

## 🔬 Spectrofluorometric Reference Characterization

To substantiate the optical properties of the carbon dots, benchtop spectrofluorometry was performed using a research-grade **JASCO FP-8550 Spectrofluorometer** (S/N: C023162067, 150 W Xenon arc lamp) at **Vellore Institute of Technology (VIT)**:
* **Excitation Wavelength:** $\lambda_{ex} = 360.0\text{ nm}$ (matching 365 nm UV dark-box illumination)
* **Emission Range:** $370\text{--}710\text{ nm}$ across 5 independent scans ($N=5$)
* **Optical Window:** Intense photoluminescence across the blue-to-cyan region ($440\text{--}520\text{ nm}$), confirming optical alignment with standard RGB camera channels.

The raw instrumental data is provided in `results/spectrometer_data/jasco_fp8550_data.txt`.

---

## 🔒 Data Availability Statement

In accordance with institutional guidelines and journal reproducibility policies:
* All source code, training manifests, split assignments, model checkpoints, and instrumental spectrometer data are openly provided in this repository.
* Raw uncompressed image datasets are available from the authors upon reasonable request, subject to institutional data-transfer agreements.

---

## 📖 Citation

If you use MTFINet in your research, please cite:

```bibtex
@article{kim2026mtfinet,
  title={Carbon-Dot Fluorescence Imaging for Heavy-Metal Determination Using Multi-Task Learning},
  author={Kim, Bubryur and Nath, Sourasish and Pranav and K.R., Sri Preethaa and Natarajan, Yuvaraj and Munisamy, Shyamala Devi},
  journal={IEEE Sensors Journal},
  year={2026}
}
```

---

## ✉️ Contact & Inquiries
For questions regarding the codebase or paper revision, please contact:
* **Sourasish Nath** (`sourasish.nath2025@vitstudent.ac.in`)
* **Yuvaraj Natarajan** (Corresponding Author, `yuvaraj.n@vit.ac.in`)
