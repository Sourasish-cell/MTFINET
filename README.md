# MTFINet: Multi-Task Fluorescence Intelligence Network for Heavy-Metal Determination

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: IEEE Submission](https://img.shields.io/badge/Submission-IEEE%20Sensors%20Journal-brightgreen.svg)]()

Official PyTorch implementation of the **Multi-Task Fluorescence Intelligence Network (MTFINet)** for simultaneous heavy-metal species identification and nominal concentration-tier estimation from paper-based fluorescent carbon-dot images.

---

## 📌 Overview

Paper-based analytical devices ($\mu\text{PADs}$) using fluorescent carbon-dot nanoprobes offer low-cost, portable platforms for decentralized environmental water monitoring. However, traditional optical readouts relying on scalar color averages or decoupled models discard 2D spatial quenching morphology (coffee-ring distributions, capillary diffusion gradients, and substrate textures).

**MTFINet** introduces an end-to-end multi-task deep learning architecture based on a shared **ResNet-18** convolutional backbone coupled to dual specialized linear classification heads:
1. **Metal Identification Head:** Categorical classification across 8 toxic heavy-metal species (**Pb, Hg, Cd, Cr, Cu, Zn, Ag, Pd**).
2. **Concentration Classification Head:** Discrete ordinal classification across 7 nominal concentration tiers (**0, 10, 50, 100, 150, 200, 250 µM**).

By learning joint representations from spatial fluorescence response patterns, MTFINet reduces model parameter requirements by 50% relative to isolated single-task models while regularizing feature learning against localized paper drying artifacts.

---

## 📊 Performance Summary

### 1. Laboratory Benchmark (Held-Out Test Set, $N = 818$ Independent Physical Samples)
The dataset was partitioned using an audited, leakage-safe grouped split (`GroupShuffleSplit`, random seed 42) based on unique physical sensor spot identifiers before any data augmentation, ensuring zero data leakage across subsets.

| Evaluation Task | Metric | Test Score | Description |
| :--- | :--- | :---: | :--- |
| **Metal Identification** | **Classification Accuracy** | **92.67%** | Exact species match across 8 classes |
| | **Total Error of Classification (TEC)** | **7.33%** | $100\% - \text{Accuracy}$ |
| | **Macro Precision** | **94.56%** | Unweighted mean precision across metals |
| | **Macro Recall** | **93.42%** | Unweighted mean recall across metals |
| | **Macro F1-Score** | **93.61%** | Harmonic mean of precision and recall |
| **Concentration Ranking** | **Classification Accuracy** | **97.43%** | Exact class match across 7 nominal tiers |
| | **Total Error of Classification (TEC)** | **2.57%** | $100\% - \text{Accuracy}$ |
| | **Mapped-Class MAE** | **2.00 µM** | $\frac{1}{N}\sum |C(y_i) - C(\hat{y}_i)|$ across 0–250 µM range |
| | **Quadratic Weighted Kappa (QWK)** | **0.9850** | Inter-rater ordinal concordancy metric |

#### Per-Metal Test Performance:
* **Chromium (Cr):** 100.0% Precision | 100.0% Recall | 1.0000 F1
* **Lead (Pb):** 100.0% Precision | 100.0% Recall | 1.0000 F1
* **Cadmium (Cd):** 99.04% Precision | 96.26% Recall | 0.9763 F1
* **Silver (Ag):** 95.18% Precision | 98.75% Recall | 0.9693 F1
* **Palladium (Pd):** 100.0% Precision | 87.96% Recall | 0.9360 F1
* **Mercury (Hg):** 95.45% Precision | 84.68% Recall | 0.8974 F1
* **Zinc (Zn):** 94.90% Precision | 83.04% Recall | 0.8857 F1
* **Copper (Cu):** 71.88% Precision | 96.64% Recall | 0.8244 F1

---

### 2. Real-World Environmental Matrix Validation ($N = 960$ Physical Test Strips)
An extensive spike-and-recovery validation was conducted on authentic **municipal drinking tap water** and **natural coastal sea water** across all 8 metals and 6 non-zero concentration tiers (10 independent physical replicates per condition).
* **Zero-Shot Concentration Recovery:** Quantitative recoveries for multiple toxic metals (Cd, Cr, Cu) conformed to analytical criteria ($80.0\%\text{--}120.0\%$, $\text{RSD} < 18\%$) at the critical $10\text{ }\mu\text{M}$ threshold.
* **Matrix-Induced Baseline Shift:** High ionic strength ($\approx 0.7\text{ M}$ in seawater) induces Debye screening, creating an optical baseline shift under uncalibrated zero-shot transfer ($7.92\%$ classification).
* **Matrix-Matched Few-Shot Calibration:** Standard calibration using 1–2 physical reference strips per condition completely eliminated the matrix-induced offset, restoring classification accuracy to **$98.33\%\text{--}99.74\%$** on unseen environmental test strips.

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
│   └── plot_quenching_curves.py       <- Extracts image-derived solid-state response curves
├── real_world_spiking/                <- Real-world environmental water evaluation pipeline
│   ├── crop_evaluation_images.py      <- Chromaticity-based automated ROI cropping script
│   ├── generate_spiking_manifest.py   <- Manifest builder for environmental test strips
│   ├── evaluate_spiking.py            <- Spiking recovery evaluator & calibration analysis
│   ├── manifests/
│   │   └── spiking_manifest.csv       <- Manifest of 960 physical tap & sea water strips
│   └── results/
│       ├── figures/
│       │   ├── spike_recovery_barchart.png
│       │   └── spiking_confusion_matrices.png
│       ├── spike_recovery_detailed_predictions.csv
│       ├── spike_recovery_summary.csv
│       └── spike_recovery_table.tex
├── notebooks/
│   └── MTFINet.ipynb                  <- Step-by-step walkthrough & interactive evaluation
├── results/
│   ├── test_metrics_summary.json      <- Machine-readable test metric summary
│   ├── test_set_predictions.csv       <- Aligned test sample predictions
│   └── figures/
│       ├── leakage_free_dual_confusion_matrix.png
│       └── image_optical_quenching_curves.png
└── checkpoints/
    ├── README.md                      <- Model loading guide & architecture details
    └── best_multitask_model_leakage_free.pth  <- Trained PyTorch weights (42.7 MB)
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Sourasish-cell/MTFINet.git
cd MTFINet
pip install -r requirements.txt
```

### 2. Evaluate Pretrained Model on Laboratory Test Set
To evaluate the model on the held-out test split ($N = 818$ physical samples) and reproduce the confusion matrices and metrics:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_multitask_model_leakage_free.pth \
    --manifest dataset/dataset_manifest.csv
```

### 3. Evaluate Real-World Environmental Spiking Study
To run the spike-and-recovery evaluation across all 960 physical test strips in tap water and sea water:
```bash
python real_world_spiking/evaluate_spiking.py \
    --checkpoint checkpoints/best_multitask_model_leakage_free.pth \
    --manifest real_world_spiking/manifests/spiking_manifest.csv \
    --output_dir real_world_spiking/results
```

### 4. Train MTFINet from Scratch
```bash
python scripts/train.py \
    --manifest dataset/dataset_manifest.csv \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0001 \
    --save-path checkpoints/best_multitask_model_leakage_free.pth
```

---

## 🔒 Data Availability Statement

In accordance with institutional guidelines and reproducibility policies:
* All source code, training manifests, split assignments, model checkpoints, real-world spiking manifests, and evaluation summaries are openly provided in this repository.
* Raw solid-state fluorescence micrographs were generated within the sensor laboratory at Vellore Institute of Technology (VIT). In compliance with institutional custodial policies, primary image archives remain available from the corresponding author upon reasonable academic request for non-commercial research validation.

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
