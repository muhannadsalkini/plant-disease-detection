# 🌿 Plant Disease Detection — BIL539 Term Project

**Author:** Muhannad Salkini  
**Student ID:** 251129910  
**Course:** BIL539 — Biruni University, Department of Computer Engineering  

---

## 📋 Project Overview

This project implements and compares **three deep learning models** for automated plant disease classification using the PlantVillage dataset:

| Model | Type | Test Accuracy | Macro F1 | Params | Time/Epoch |
|-------|------|:---:|:---:|:---:|:---:|
| **Custom CNN** | From scratch | 73.00% | 0.7008 | 0.13M | — |
| **EfficientNetB0** | Transfer Learning | **96.13%** | **0.9601** | 4.38M | 115.8s |
| **MobileNetV2** | Transfer Learning | 91.77% | 0.9105 | 2.59M | 72.2s |

**Key finding:** Transfer learning models outperform the scratch-trained baseline by ~23 percentage points, confirming the hypothesis that pre-trained ImageNet features transfer effectively to the plant disease domain.

---

## 📁 Project Structure

```
plant disease/
├── config.py                    # All hyperparameters and path constants
├── main.py                      # One-command full pipeline runner
├── download_dataset.py          # Kaggle dataset downloader
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Dataset loading, augmentation, generators
│   ├── models.py                # Model architectures (CNN, EfficientNetB0, MobileNetV2)
│   ├── train.py                 # Training loop (two-phase for transfer models)
│   ├── evaluate.py              # Test-set evaluation, metrics, confusion matrix
│   ├── visualize.py             # Training curves, confusion matrix plots
│   └── hyperparameter_sweep.py  # Systematic hyperparameter search
│
├── data/
│   └── plantvillage/            # Dataset (downloaded separately)
│       ├── Tomato___Early_blight/
│       ├── Tomato___healthy/
│       └── ...                  # 15 class folders
│
├── results/
│   ├── models/                  # Saved .keras model files
│   │   ├── custom_cnn_final.keras
│   │   ├── efficientnetb0_final.keras
│   │   └── mobilenetv2_final.keras
│   ├── logs/                    # Training histories, metrics JSON/CSV
│   │   ├── all_model_metrics.json
│   │   ├── efficientnetb0_summary.json
│   │   └── ...
│   └── figures/                 # Generated plots (PNG)
│
└── BIL539_Plant_Disease_Report.tex   # Full LaTeX report
```

---

## ⚙️ Setup

### 1. Clone / open the project
```bash
cd "plant disease"
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API (for dataset download)
- Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
- Place `kaggle.json` in `~/.kaggle/`
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download the dataset
```bash
python3 download_dataset.py
```
This downloads the PlantVillage dataset (~200MB) from Kaggle and places it in `data/plantvillage/`.

---

## 🚀 Running the Project

### Option A — Full pipeline (recommended)
```bash
python3 main.py
```
Runs: download → train all models → evaluate → generate figures.

### Option B — Step by step

**Train a specific model:**
```bash
python3 src/train.py --model custom_cnn
python3 src/train.py --model efficientnetb0
python3 src/train.py --model mobilenetv2
python3 src/train.py --model all          # train all three
```

**Evaluate on test set:**
```bash
python3 src/evaluate.py --model all
```

**Generate figures (training curves, confusion matrices):**
```bash
python3 src/visualize.py
```

**Hyperparameter sweep (EfficientNetB0):**
```bash
python3 src/hyperparameter_sweep.py
```

---

## 🏗️ Model Architectures

### Model 1 — Custom CNN (Baseline)
A lightweight CNN trained from scratch:
- 3 × convolutional blocks: `Conv2D(3×3) → BatchNorm → ReLU → MaxPool(2×2)`
- Filter sizes: 32 → 64 → 128
- Head: `GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Softmax(15)`
- **0.13M parameters**

### Model 2 — EfficientNetB0 (Transfer Learning)
- ImageNet pre-trained base (frozen in Phase 1, top 20 layers unfrozen in Phase 2)
- `Rescaling(255.0)` layer added before base model to convert `[0,1] → [0,255]` (critical — see note below)
- Head: `GAP → Dense(256, ReLU) → Dropout(0.4) → Softmax(15)`
- **4.38M parameters**
- Two-phase training: feature extraction (lr=5e-4) then fine-tuning (lr=1e-4)

### Model 3 — MobileNetV2 (Transfer Learning)
- ImageNet pre-trained base with depthwise separable convolutions
- Same two-phase strategy as EfficientNetB0
- Accepts `[0,1]` inputs natively (no rescaling needed)
- **2.59M parameters**
- Best accuracy-to-efficiency ratio for deployment

---

## ⚠️ Important: EfficientNetB0 Preprocessing Fix

A non-obvious bug was discovered and fixed during this project:

> **EfficientNetB0 was pre-trained on pixel values in `[0, 255]`**, but the data pipeline normalizes images to `[0, 1]`. Without correction, EfficientNetB0 receives near-zero inputs and produces random predictions (~16% accuracy — barely above chance for 15 classes).

**Fix applied in `src/models.py`:**
```python
inputs = layers.Input(shape=input_shape)
x = layers.Rescaling(scale=255.0)(inputs)   # [0,1] → [0,255]
x = base_model(x, training=False)
```

MobileNetV2 does not have this issue as it operates on `[0,1]` inputs natively.

---

## 📊 Dataset

- **Source:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Subset used:** 15 classes, **20,638 images**
- **Splits:** Train 14,446 | Val 3,095 | Test 3,097 (70/15/15 stratified)
- **Input size:** 224×224 RGB, normalized to [0,1]
- **Augmentation (train only):** horizontal/vertical flip, rotation ±20°, zoom 20%, shift 10%, brightness ±20%

**Classes covered:**
Tomato (Early Blight, Late Blight, Healthy, Leaf Mold, etc.), Potato (Early Blight, Late Blight, Healthy), Pepper (Bacterial Spot, Healthy), Corn (Common Rust, Northern Leaf Blight), Grape (Black Rot, Healthy), and others.

---

## 📈 Results Summary

```
══════════════════════════════════════════════════════════════════════
  FINAL TEST SET RESULTS
══════════════════════════════════════════════════════════════════════
  Model                  Acc(%)       F1  Precision   Recall  Params(M)  s/ep
  ────────────────────  ───────  ───────  ─────────  ───────  ─────────  ────
  custom_cnn             73.00   0.7008     0.7322   0.7434       0.13    --
  efficientnetb0         96.13   0.9601     0.9580   0.9637       4.38  115.8
  mobilenetv2            91.77   0.9105     0.9084   0.9197       2.59   72.2
```

### Hyperparameter Analysis (EfficientNetB0)

| Learning Rate | Val Acc | Converges (epochs) |
|:---:|:---:|:---:|
| 0.001  | 91.8% | 8  |
| **0.0005** | **95.9%** | **15** |
| 0.0001 | 93.4% | 22 |

| Dropout | Test Acc | Overfit Gap |
|:---:|:---:|:---:|
| 0.2 | 95.1% | 4.2% |
| **0.4** | **96.1%** | **1.8%** |
| 0.5 | 94.3% | 0.9% |

| Optimizer | Test Acc | Stability |
|:---:|:---:|:---:|
| **Adam** | **96.1%** | Stable |
| RMSprop | 93.2% | Stable |
| SGD | 88.6% | Moderate oscillation |

---

## 🔧 Configuration (`config.py`)

Key parameters you can tune:

```python
IMAGE_SIZE      = (224, 224)   # Input resolution
BATCH_SIZE      = 32
LEARNING_RATE   = 5e-4         # Phase 1 / CNN training LR
FINE_TUNE_LR    = 1e-4         # Phase 2 fine-tuning LR
DROPOUT_RATE    = 0.4
FINE_TUNE_LAYERS= 20           # Top N layers to unfreeze
OPTIMIZER       = "adam"
EPOCHS_PHASE1   = 10
EPOCHS_PHASE2   = 30
PATIENCE        = 10           # Early stopping patience
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
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Report

The full academic report is in **`BIL539_Plant_Disease_Report.tex`** (IEEE two-column format). To compile:

```bash
pdflatex BIL539_Plant_Disease_Report.tex
pdflatex BIL539_Plant_Disease_Report.tex   # run twice for references
```

Or open in [Overleaf](https://www.overleaf.com) by uploading the `.tex` file.

---

## 🔮 Future Work

1. **In-the-wild evaluation** — test models on [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) for real-field robustness
2. **Object detection** — extend to YOLOv8 for disease *localization* (bounding boxes)
3. **Model compression** — knowledge distillation / quantization for edge/IoT deployment
4. **Multi-label classification** — handle co-occurring diseases on the same leaf
5. **Grad-CAM visualizations** — explainability maps showing which leaf regions activate the model

---

## 📜 License

This project is for academic purposes (BIL539 course, Biruni University).  
The PlantVillage dataset is publicly available under CC0 license.
