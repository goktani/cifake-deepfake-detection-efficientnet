# 🤖 CIFAKE: AI-Generated Image Detection with EfficientNet-B0

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.50%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9984-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Can a neural network tell the difference between a real photograph and an AI-generated image? This project answers yes — with **98.5% accuracy**.

---

## 📌 Overview

As AI image generation (Stable Diffusion, DALL·E, Midjourney) becomes increasingly realistic, the ability to detect synthetic content is critical for digital forensics, journalism, and online trust systems.

This project fine-tunes a pretrained **EfficientNet-B0** on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — a benchmark containing 120,000 images split equally between real CIFAR-10 photographs and Stable Diffusion v1.4 synthetic counterparts.

---

## 🏆 Results

| Metric | Score |
|---|---|
| Test Accuracy | **98.50%** |
| ROC-AUC | **0.9984** |
| False Positives | 188 (REAL → FAKE) |
| False Negatives | 113 (FAKE → REAL) |
| Epochs Trained | 15 |

---

## 📂 Dataset Structure
```
cifake-real-and-ai-generated-synthetic-images/
├── train/
│   ├── REAL/      # 50,000 images (CIFAR-10)
│   └── FAKE/      # 50,000 images (Stable Diffusion v1.4)
└── test/
    ├── REAL/      # 10,000 images
    └── FAKE/      # 10,000 images
```

Download from Kaggle: [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

---

## 🧠 Model Architecture
```
EfficientNet-B0 (ImageNet pretrained)
└── features         ← frozen pretrained backbone
└── classifier
    ├── Dropout(0.4)
    ├── Linear(1280 → 256)
    ├── ReLU
    ├── Dropout(0.2)
    └── Linear(256 → 2)    ← REAL / FAKE
```

**Why EfficientNet-B0?**
- Compound scaling (depth + width + resolution simultaneously)
- Best accuracy/parameter trade-off in its class
- ImageNet pretrained weights provide strong low-level feature extraction

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Input Size | 224×224 |
| Batch Size | 64 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | Cosine Annealing |
| Loss | CrossEntropy + Label Smoothing (0.05) |
| Early Stopping | Patience = 5 |
| GPU | NVIDIA Tesla T4 |

---

## 📊 Notebook Contents

| Section | Description |
|---|---|
| 1. Setup | Imports, seeds, device, paths |
| 2. EDA | Class distribution, sample grid |
| 3. Data Pipeline | Augmentation, DataLoaders |
| 4. Model | EfficientNet-B0 architecture |
| 5. Training | Full training loop with early stopping |
| 6. Training Curves | Loss / Accuracy / LR plots |
| 7. Confusion Matrix & ROC | Evaluation on full test set |
| 8. Grad-CAM | Visual explainability heatmaps |
| 9. Wrong Predictions | FP/FN analysis, confidence distribution |
| 10. Model Saving | Checkpoint + final summary |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/goktani/cifake-deepfake-detection-efficientnet.git
cd cifake-deepfake-detection-efficientnet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Place the CIFAKE dataset under:
```
data/cifake-real-and-ai-generated-synthetic-images/
```
Or update `DATASET_DIR` in the notebook to your local path.

### 4. Run the notebook
```bash
jupyter lab cifake-efficientnet-b0-en.ipynb
```

> **Kaggle users:** The dataset path is pre-configured. Simply attach the dataset and run all cells.

---

## 🔍 Grad-CAM Explainability

Grad-CAM reveals which regions of an image the model focuses on when making its prediction.

- **REAL images** → model attends to natural textures, object edges, and structural features
- **FAKE images** → model detects artifacts in smooth regions and unnatural patterns introduced by the diffusion process

---

## 📁 Repository Structure
```
├── cifake-efficientnet-b0-en.ipynb   # Main notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── outputs/
    ├── training_curves.png
    ├── confusion_roc.png
    ├── gradcam_real.png
    ├── gradcam_fake.png
    └── wrong_predictions.png
```

---

## 📖 References

- Bird, J.J. and Lotfi, A., 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access.
- Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features from tiny images.*
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.

---

## 📜 License

This project is licensed under the MIT License. The CIFAKE dataset is also published under the MIT License.
