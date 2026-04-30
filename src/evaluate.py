"""
src/evaluate.py
---------------
Loads trained models and evaluates them on the held-out test set.
Produces:
  - Accuracy, Macro-F1, Precision, Recall per model
  - Confusion matrix (saved as PNG)
  - Per-class classification report (CSV)
  - Combined results JSON → used to fill paper tables

Usage:
    python src/evaluate.py
    python src/evaluate.py --model efficientnetb0
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_pipeline import load_data

# ─────────────────────────────────────────────────────────────────────────────
# Load a saved model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> tf.keras.Model:
    """Load the final saved model for the given name."""
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}_final.keras")
    if not os.path.exists(model_path):
        # Try phase2 best (for transfer models)
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}_phase2_best.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found for '{model_name}'.\n"
            f"Run training first:  python src/train.py --model {model_name}"
        )
    print(f"[Evaluate] Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Get predictions from generator
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model: tf.keras.Model, test_gen) -> tuple:
    """
    Run inference on the full test set.

    Returns
    -------
    y_true : np.ndarray (N,)   integer class indices
    y_pred : np.ndarray (N,)   integer class indices
    """
    test_gen.reset()
    steps = len(test_gen)

    y_prob = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Ground-truth labels from the generator
    y_true = test_gen.classes[:len(y_pred)]

    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names: list) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy"       : float(acc),
        "macro_f1"       : float(f1),
        "macro_precision": float(prec),
        "macro_recall"   : float(rec),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save per-class report
# ─────────────────────────────────────────────────────────────────────────────

def save_classification_report(
    y_true, y_pred, class_names: list, model_name: str
):
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).transpose()
    out_path = os.path.join(config.LOGS_DIR, f"{model_name}_classification_report.csv")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    df.to_csv(out_path, float_format="%.4f")
    print(f"[Evaluate] Classification report saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Count model parameters
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: tf.keras.Model) -> float:
    """Return total parameter count in millions. Keras 3 compatible."""
    import numpy as np
    total = sum(int(np.prod(w.shape)) for w in model.weights)
    return round(total / 1e6, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation for one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, test_gen, class_names: list) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'═'*60}")

    model  = load_model(model_name)
    params = count_params(model)

    y_true, y_pred = get_predictions(model, test_gen)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["params_M"] = params

    print(f"  Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"  Parameters     : {params}M")

    save_classification_report(y_true, y_pred, class_names, model_name)

    # Save confusion matrix data (not plot — that's in visualize.py)
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(config.LOGS_DIR, f"{model_name}_confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"[Evaluate] Confusion matrix saved → {cm_path}")

    # Merge timing info if available
    timing_path = os.path.join(config.LOGS_DIR, f"{model_name}_summary.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)
        metrics["time_per_epoch_s"] = timing.get("time_per_epoch_s", None)
        metrics["total_epochs"]     = timing.get("total_epochs", None)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["custom_cnn", "efficientnetb0", "mobilenetv2", "all"],
    )
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("\n[Evaluate] Loading dataset …")
    _, _, test_gen, class_names, _, _ = load_data()

    all_metrics = {}
    model_list  = (
        [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE]
        if args.model == "all"
        else [args.model]
    )

    for name in model_list:
        try:
            metrics = evaluate_model(name, test_gen, class_names)
            all_metrics[name] = metrics
        except FileNotFoundError as e:
            print(f"⚠️  {e}")

    # ── Save combined results ─────────────────────────────────────────────────
    out_path = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluate] Combined metrics saved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  FINAL TEST SET RESULTS")
    print("═"*70)
    print(f"  {'Model':<20} {'Acc(%)':>8} {'F1':>8} {'Params(M)':>10} {'s/ep':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for name, m in all_metrics.items():
        tpe = f"{m.get('time_per_epoch_s', 0):.1f}" if m.get("time_per_epoch_s") else "  --"
        print(f"  {name:<20} {m['accuracy']*100:>8.2f} {m['macro_f1']:>8.4f} "
              f"{m['params_M']:>10.1f} {tpe:>8}")


if __name__ == "__main__":
    main()
