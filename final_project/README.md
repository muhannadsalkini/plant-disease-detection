# 🏥 Skin Lesion Classification & Segmentation — BIS539 Final Project

**Author:** Muhannad Salkini  
**Student ID:** 251129910  
**Course:** BIS539 — Biruni University, Department of Computer Engineering  
**Semester:** 2025-2026 Spring  

---

## 📋 Project Overview

This final project implements and compares **eight deep learning models** for automated skin lesion classification and segmentation using the **ISIC 2018 Challenge dataset** (10,015 dermoscopic images, 7 disease categories).

### Models Implemented

| # | Model | Type | Task | Params |
|---|-------|------|------|--------|
| 1 | **Custom CNN** | From scratch | Classification | ~0.13M |
| 2 | **EfficientNetB0** | Transfer Learning | Classification | ~4.4M |
| 3 | **MobileNetV2** | Transfer Learning | Classification | ~2.6M |
| 4 | **Vision Transformer (ViT)** | Transformer | Classification | ~1.5M |
| 5 | **CNN+Transformer Hybrid** | Hybrid | Classification | ~1.2M |
| 6 | **U-Net** | Encoder-Decoder | Segmentation | ~31M |
| 7 | **TransUNet** | CNN+Transformer | Segmentation | ~10M |
| 8 | **Instance Segmentation** | Multi-task | Clf + Seg | ~5M |

### Key Highlights
- **3 paradigms**: Classical CNN, Transformer/Hybrid, Segmentation
- **10-paper literature review** with detailed comparison tables
- **Transformer requirement**: ViT, CNN+Transformer Hybrid, TransUNet
- **Segmentation + instance-based methods**: U-Net, TransUNet, Instance Seg
- **Comprehensive metrics**: Accuracy, F1, Precision, Recall, Dice, IoU, Sensitivity, Specificity

---

## 📁 Project Structure

```
final_project/
├── config.py                         # All hyperparameters and paths
├── main.py                           # One-command full pipeline
├── download_dataset.py               # ISIC 2018 Kaggle downloader
├── requirements.txt                  # Python dependencies
├── BIS539_Final_Report.tex           # Full LaTeX report (IEEE format)
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Classification + segmentation data loading
│   ├── classification_models.py      # CNN, EfficientNet, MobileNet, ViT, Hybrid
│   ├── segmentation_models.py        # U-Net, TransUNet, Instance Seg
│   ├── train.py                      # Training pipeline (all models)
│   ├── evaluate.py                   # Test-set evaluation + metrics
│   └── visualize.py                  # Training curves, confusion matrices, comparisons
│
├── data/
│   └── ISIC2018/                     # Dataset (downloaded separately)
│       ├── images/                   # Dermoscopic images
│       ├── masks/                    # Segmentation masks
│       └── labels.csv                # Classification labels (7 classes)
│
└── results/
    ├── models/                       # Saved .keras model files
    ├── logs/                         # Training histories, metrics
    └── figures/                      # Generated plots (PNG)
```

---

## ⚙️ Setup

### 1. Navigate to the project
```bash
cd final_project
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download the dataset
```bash
python3 download_dataset.py
```

---

## 🚀 Running the Project

### Full pipeline (all stages)
```bash
python3 main.py
```

### Step by step
```bash
# Download dataset
python3 main.py --stage download

# Train classification models
python3 main.py --stage train_clf
python3 main.py --stage train_clf --model efficientnetb0
python3 main.py --stage train_clf --model vit_classifier

# Train segmentation models
python3 main.py --stage train_seg
python3 main.py --stage train_seg --model unet

# Evaluate all models
python3 main.py --stage evaluate

# Generate figures
python3 main.py --stage visualize
```

### Train individual models directly
```bash
python3 src/train.py --task classification --model custom_cnn
python3 src/train.py --task classification --model vit_classifier
python3 src/train.py --task segmentation --model transunet
python3 src/train.py --task all
```

---

## 📊 Dataset — ISIC 2018

- **Source:** [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/) / Kaggle
- **Images:** 10,015 dermoscopic images
- **Classes:** 7 (MEL, NV, BCC, AKIEC, BKL, DF, VASC)
- **Segmentation masks:** Binary lesion boundary masks
- **Splits:** Train 70% | Val 15% | Test 15% (stratified)

| Class | Disease | Count | % |
|-------|---------|------:|---:|
| NV | Melanocytic Nevus | 6,705 | 66.9% |
| MEL | Melanoma | 1,113 | 11.1% |
| BKL | Benign Keratosis | 1,099 | 11.0% |
| BCC | Basal Cell Carcinoma | 514 | 5.1% |
| AKIEC | Actinic Keratosis | 327 | 3.3% |
| VASC | Vascular Lesion | 142 | 1.4% |
| DF | Dermatofibroma | 115 | 1.1% |

---

## 🏗️ Model Architectures

### Classification Models

**Custom CNN** — 3×Conv blocks (32→64→128) + GAP + Dense + Softmax  
**EfficientNetB0** — ImageNet transfer learning, two-phase training  
**MobileNetV2** — Lightweight depthwise separable CNN  
**ViT** — 16×16 patches → embedding → 4 Transformer encoder blocks → [CLS] → MLP  
**CNN+Transformer Hybrid** — CNN feature extractor → Transformer encoder → GAP → MLP  

### Segmentation Models

**U-Net** — 4-level encoder-decoder with skip connections  
**TransUNet** — CNN encoder → Transformer bottleneck → CNN decoder  
**Instance Seg** — Shared backbone with dual heads (classification + segmentation)  

---

## 📈 Evaluation Metrics

### Classification
- Accuracy, Macro Precision, Recall, F1-Score
- Per-class classification reports
- Confusion matrices

### Segmentation
- Dice Score (F1 for segmentation)
- IoU (Jaccard Index)
- Pixel Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)

---

## 📄 Report

The full academic report is in **`BIS539_Final_Report.tex`** (IEEE two-column format) with:
- **10-paper literature review** (detailed analysis of each paper)
- Comprehensive comparison table with published benchmarks
- Research gap analysis

To compile:
```bash
pdflatex BIS539_Final_Report.tex
pdflatex BIS539_Final_Report.tex   # run twice for references
```

---

## 📦 Dependencies

```
tensorflow>=2.10
numpy
pandas
scikit-learn
matplotlib
seaborn
kaggle
Pillow
```

---

## 📜 License

This project is for academic purposes (BIS539 course, Biruni University).  
The ISIC 2018 dataset is publicly available under CC-BY-NC license.
