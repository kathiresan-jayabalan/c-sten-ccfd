[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18529994.svg)](https://doi.org/10.5281/zenodo.18529994)

# C-STEN-CCFD  
Contrastive Spatio-Temporal Deep Learning for Credit Card Fraud Detection

This repository provides a **Jupyter Notebook** implementation of a Contrastive Spatio‑Temporal deep learning pipeline (C‑STEN) for **credit card fraud detection and evaluation** using the public Kaggle `creditcard.csv` dataset.

**Main artifact:** `notebooks/c-sten-ccfd.ipynb`

**End-to-end execution:** data loading → representation learning → fine‑tuning → evaluation.

---

## Overview

Credit card fraud detection is a strongly imbalanced classification problem where  temporal ordering and contextual patterns play a critical role.

This notebook implements a C-STEN-style workflow that combines:
- **Self‑supervised contrastive representation learning** to learn useful sequence representations.
- **Supervised fine‑tuning** for binary fraud classification (`Class` ∈ {0,1}).
 
All stages of the experiment are implemented inside a single notebook for clarity and reproducibility.

---

## Methodology

The notebook performs the following steps:

- Load and preprocess the dataset `creditcard.csv` (features, scaling/normalization as used in the notebook)  
  - Sort transactions by time  
  - Scale numeric features
- Build sequential samples (sliding windows) for spatio‑temporal / sequence modeling.
- Generate two augmented “views” per sequence using noise injection and feature masking for contrastive learning.
- Train a Transformer encoder backbone using:
  - **NT‑Xent contrastive loss** (representation learning).
  - **temporal prediction loss** (next-step / temporal target regression).
- Fine‑tune a classifier head for binary fraud detection (Class ∈ {0,1})
- Report validation metrics per epoch and final test metrics.

---

## Dataset

The experiments use the public **Credit Card Fraud Detection** dataset (European cardholders, anonymized features) available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (subject to Kaggle terms of use)

After downloading, place the file at:

data/creditcard.csv

If the full dataset is unavailable, a small synthetic or reduced subset may be used to verify execution. Such data will not reproduce reported performance metrics.

---

## Environment

Tested configuration:
- Python 3.x
- Jupyter Notebook or JupyterLab
- PyTorch
- GPU: NVIDIA T4 GPU (cloud or Colab)

---

## Dependencies

The notebook relies on the following Python packages. Install dependencies from `requirements.txt`:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch
- torchvision
- torchaudio
- tqdm
- ipywidgets
- jupyterlab
- pyyaml

---

## How to Run

1. Download the dataset from Kaggle.
2. Place `creditcard.csv` under the `data/` directory.
3. Launch Jupyter Notebook or JupyterLab.
4. Open `notebooks/c-sten-ccfd.ipynb`.
5. Run all cells from top to bottom.

The notebook will preprocess data, perform contrastive pretraining, fine-tune the classifier, and evaluate the model.

Runtime depends on hardware and dataset size.

---

## Outputs

All evaluation metrics and visualizations are generated directly within the notebook output cells. No result files are written to disk by default.

The notebook prints final and per-epoch metrics at the end of training and evaluation.

- ROC AUC
- Precision
- Recall
- F1-score
- Accuracy

Plots such as training loss curves and ROC curves may be displayed inline during execution depending on enabled cells.

Example performance values observed during testing include high accuracy and strong recall for the minority (fraud) class. Exact results depend on thresholds and experimental settings.

- Accuracy ≈ 0.9988
- Precision ≈ 0.6212
- Recall ≈ 0.8367
- F1 ≈ 0.7130
- ROC AUC ≈ 0.9473

---

## License

This project is licensed under the Apache License 2.0.

---

## Citation

If you use this repository in academic or research work, please cite:

Jayabalan, K., & Radhakrishnan, S. (2026). C-STEN-CCFD: Contrastive Spatio-Temporal Ensemble Network for Credit Card Fraud Detection (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.18529994
