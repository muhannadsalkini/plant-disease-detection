"""
src/train.py
------------
Training pipeline for all three models with two-phase strategy for transfer models.

Usage:
    python src/train.py --model custom_cnn
    python src/train.py --model efficientnetb0
    python src/train.py --model mobilenetv2
    python src/train.py --model all            # trains all three sequentially
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_pipeline import load_data
from src.models import (
    build_custom_cnn,
    build_efficientnetb0,
    build_mobilenetv2,
    prepare_fine_tuning,
    print_model_summary,
)

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, phase: str = ""):
    """
    Returns a list of Keras callbacks:
      - ModelCheckpoint : saves the best val_accuracy weights
      - EarlyStopping   : stops training if val_accuracy stagnates
      - CSVLogger       : logs per-epoch metrics to a CSV file
      - ReduceLROnPlateau: lowers LR if val_loss stagnates (useful for scratch)
    """
    tag = f"{model_name}_{phase}" if phase else model_name
    ckpt_path = os.path.join(config.MODELS_DIR, f"{tag}_best.keras")
    log_path  = os.path.join(config.LOGS_DIR,   f"{tag}_history.csv")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR,   exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Train custom CNN
# ─────────────────────────────────────────────────────────────────────────────

def train_custom_cnn(train_gen, val_gen, class_weights):
    print("\n" + "═" * 60)
    print("  Training: Custom CNN (from scratch)")
    print("═" * 60)

    model = build_custom_cnn()
    print_model_summary(model)

    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(config.MODEL_CUSTOM),
        verbose=1,
    )
    elapsed = time.time() - t0

    # Save final model
    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_CUSTOM}_final.keras"))

    # Compute average time per epoch
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    print(f"\n✅ Custom CNN done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history, time_per_ep


# ─────────────────────────────────────────────────────────────────────────────
# Generic transfer learning trainer
# ─────────────────────────────────────────────────────────────────────────────

def _train_transfer(
    model_name: str,
    build_fn,
    train_gen,
    val_gen,
    class_weights,
):
    print("\n" + "═" * 60)
    print(f"  Training: {model_name.upper()} (Transfer Learning)")
    print("═" * 60)

    # ── Phase 1: Feature Extraction ──────────────────────────────────────────
    print("\n[Phase 1] Feature Extraction — base model frozen")
    model, base_model = build_fn()
    print_model_summary(model)

    t0 = time.time()
    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase1"),
        verbose=1,
    )
    ep1 = len(history_p1.history["loss"])

    # ── Phase 2: Fine-Tuning ─────────────────────────────────────────────────
    print("\n[Phase 2] Fine-Tuning — top layers unfrozen")
    model = prepare_fine_tuning(model, base_model)

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase2"),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep2 = len(history_p2.history["loss"])

    # Save final model
    model.save(os.path.join(config.MODELS_DIR, f"{model_name}_final.keras"))

    total_epochs = ep1 + ep2
    time_per_ep  = elapsed / total_epochs

    print(f"\n✅ {model_name} done — "
          f"phase1={ep1} + phase2={ep2} epochs, "
          f"{time_per_ep:.1f}s/epoch")

    # Merge histories for reporting
    merged = {
        k: history_p1.history[k] + history_p2.history[k]
        for k in history_p1.history
    }

    return model, merged, time_per_ep


# ─────────────────────────────────────────────────────────────────────────────
# Save training summary
# ─────────────────────────────────────────────────────────────────────────────

def save_training_summary(model_name: str, history, time_per_ep: float):
    """Save best val_accuracy and time/epoch to a JSON file."""
    best_val_acc = max(history["val_accuracy"])
    summary = {
        "model": model_name,
        "best_val_accuracy": float(best_val_acc),
        "time_per_epoch_s": float(time_per_ep),
        "total_epochs": len(history["loss"]),
    }
    out_path = os.path.join(config.LOGS_DIR, f"{model_name}_summary.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Train] Summary saved → {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train plant disease models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["custom_cnn", "efficientnetb0", "mobilenetv2", "all"],
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size (default: 32)",
    )
    args = parser.parse_args()

    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🖥️  GPU(s) found: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU found — training on CPU (will be slow)")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Train] Loading dataset …")
    train_gen, val_gen, test_gen, class_names, class_weights, _ = load_data(
        batch_size=args.batch_size
    )

    results = {}

    # ── Train requested model(s) ───────────────────────────────────────────────
    if args.model in ("custom_cnn", "all"):
        model, history, tpe = train_custom_cnn(train_gen, val_gen, class_weights)
        results[config.MODEL_CUSTOM] = save_training_summary(
            config.MODEL_CUSTOM, history, tpe
        )

    if args.model in ("efficientnetb0", "all"):
        model, history, tpe = _train_transfer(
            config.MODEL_EFFICIENT, build_efficientnetb0,
            train_gen, val_gen, class_weights
        )
        results[config.MODEL_EFFICIENT] = save_training_summary(
            config.MODEL_EFFICIENT, history, tpe
        )

    if args.model in ("mobilenetv2", "all"):
        model, history, tpe = _train_transfer(
            config.MODEL_MOBILE, build_mobilenetv2,
            train_gen, val_gen, class_weights
        )
        results[config.MODEL_MOBILE] = save_training_summary(
            config.MODEL_MOBILE, history, tpe
        )

    print("\n" + "═" * 60)
    print("  Training complete — summary")
    print("═" * 60)
    for name, s in results.items():
        print(f"  {name:20s}  val_acc={s['best_val_accuracy']:.4f}  "
              f"time/ep={s['time_per_epoch_s']:.1f}s")


if __name__ == "__main__":
    main()
