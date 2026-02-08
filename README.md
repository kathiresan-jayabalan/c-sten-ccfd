# C-STEN-CCFD  
Contrastive Spatio-Temporal Deep Learning for Credit Card Fraud Detection

This repository contains a single Jupyter Notebook implementing a Contrastive Spatio-Temporal deep learning pipeline (C-STEN) for credit card fraud detection using the public Kaggle `creditcard.csv` dataset.

Primary artifact:  
notebooks/c-sten-ccfd.ipynb (end-to-end execution: data loading, training, evaluation)

---

## Overview

Credit card fraud detection is a highly imbalanced binary classification problem where temporal ordering and contextual patterns play a critical role.

This notebook implements a C-STEN-style workflow that combines:
- Self-supervised contrastive representation learning
- Supervised fine-tuning for fraud classification

All stages of the experiment are implemented inside a single notebook for clarity and reproducibility.

---

## Methodology

The notebook performs the following steps:

- Load and preprocess the dataset  
  - Sort transactions by time  
  - Scale numeric features
- Construct sliding temporal windows for sequence modeling
- Generate two augmented views per sequence using noise injection and feature masking
- Pretrain a Transformer encoder using:
  - NT-Xent contrastive loss
  - Temporal prediction loss
- Fine-tune a classification head for binary fraud detection (Class ∈ {0,1})
- Evaluate performance on a held-out test set

---

## Dataset

The experiments use the Credit Card Fraud Detection dataset (European cardholders, anonymized features) available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file at:

data/creditcard.csv

If the full dataset is unavailable, a small synthetic or reduced subset may be used to verify execution. Such data will not reproduce reported performance metrics.

---

## Environment

Tested configuration:
- Python 3.x
- Jupyter Notebook or JupyterLab
- PyTorch
- Recommended hardware: NVIDIA T4 GPU (cloud or Colab)

CPU execution is supported for small-scale tests only.

---

## Dependencies

The notebook relies on the following Python packages:

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

The notebook reports the following evaluation metrics:

- ROC AUC
- Precision
- Recall
- F1-score
- Accuracy

Optional plots such as loss curves and ROC curves may be generated depending on enabled cells.

Example performance values observed during testing include high accuracy and strong recall for the minority (fraud) class. Exact results depend on random seeds, thresholds, and experimental settings.

---

## License

This project is licensed under the Apache License 2.0.

---

## Citation

If you use this repository in academic or research work, please cite:

@misc{csten_ccfd_repo,
  author = {Kathiresan Jayabalan},
  title = {C-STEN-CCFD: Contrastive Spatio-Temporal Deep Learning for Credit Card Fraud Detection},
  year = {2026},
  howpublished = { https://github.com/kathiresan-jayabalan/c-sten-ccfd },
  note = {Apache-2.0 licensed code repository}
}
