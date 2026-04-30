"""
config.py
---------
Central configuration for all hyperparameters, paths, and experiment settings.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "PlantVillage")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
MODELS_DIR    = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR      = os.path.join(RESULTS_DIR, "logs")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 15   # This Kaggle subset has 15 classes (not the full 38)
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42

# ─────────────────────────────────────────────
# DEFAULT TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30          # Feature-extraction phase
FINE_TUNE_EPOCHS = 20       # Fine-tuning phase (transfer models)
LEARNING_RATE = 5e-4
FINE_TUNE_LR  = 1e-5
DROPOUT_RATE  = 0.4
OPTIMIZER     = "adam"      # "adam" | "sgd" | "rmsprop"
FINE_TUNE_LAYERS = 20       # Number of top base-model layers to unfreeze

# ─────────────────────────────────────────────
# HYPERPARAMETER SWEEP (EfficientNetB0)
# ─────────────────────────────────────────────
SWEEP_LR         = [1e-3, 5e-4, 1e-4]
SWEEP_DROPOUT    = [0.2, 0.4, 0.5]
SWEEP_OPTIMIZERS = ["adam", "sgd", "rmsprop"]
SWEEP_BATCH      = [16, 32, 64]

# Epochs per sweep run (shorter to keep compute feasible)
SWEEP_EPOCHS     = 15

# ─────────────────────────────────────────────
# MODEL NAMES
# ─────────────────────────────────────────────
MODEL_CUSTOM      = "custom_cnn"
MODEL_EFFICIENT   = "efficientnetb0"
MODEL_MOBILE      = "mobilenetv2"
