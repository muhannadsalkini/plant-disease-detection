"""
main.py
-------
Master entry point for the BIL539 Plant Disease Detection project.
Runs the full pipeline in order, or individual stages via --stage flag.

Pipeline stages (run in order):
  1. download   — Download PlantVillage dataset from Kaggle
  2. train      — Train Custom CNN, EfficientNetB0, MobileNetV2
  3. sweep      — Hyperparameter sweep on EfficientNetB0
  4. evaluate   — Evaluate all models on test set, save metrics
  5. visualize  — Generate all paper figures

Usage:
  python main.py                          # run all stages
  python main.py --stage download
  python main.py --stage train --model efficientnetb0
  python main.py --stage sweep  --sweep lr
  python main.py --stage evaluate
  python main.py --stage visualize --figure comparison
"""

import os
import sys
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import config


def _setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU(s) available: {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU found — running on CPU.")


# ─────────────────────────────────────────────────────────────────────────────

def stage_download():
    print("\n" + "═"*60)
    print("  STAGE 1 — Download Dataset")
    print("═"*60)
    import download_dataset
    download_dataset.main()


def stage_train(model: str = "all", batch_size: int = config.BATCH_SIZE):
    print("\n" + "═"*60)
    print(f"  STAGE 2 — Train Model(s): {model}")
    print("═"*60)
    from src.train import (
        train_custom_cnn, _train_transfer, save_training_summary,
        build_efficientnetb0, build_mobilenetv2
    )
    from src.data_pipeline import load_data

    train_gen, val_gen, _, _, class_weights, _ = load_data(batch_size=batch_size)

    if model in ("custom_cnn", "all"):
        _, history, tpe = train_custom_cnn(train_gen, val_gen, class_weights)
        save_training_summary(config.MODEL_CUSTOM, history.history, tpe)

    if model in ("efficientnetb0", "all"):
        _, history, tpe = _train_transfer(
            config.MODEL_EFFICIENT, build_efficientnetb0,
            train_gen, val_gen, class_weights
        )
        save_training_summary(config.MODEL_EFFICIENT, history, tpe)

    if model in ("mobilenetv2", "all"):
        _, history, tpe = _train_transfer(
            config.MODEL_MOBILE, build_mobilenetv2,
            train_gen, val_gen, class_weights
        )
        save_training_summary(config.MODEL_MOBILE, history, tpe)


def stage_sweep(sweep: str = "all"):
    print("\n" + "═"*60)
    print(f"  STAGE 3 — Hyperparameter Sweep: {sweep}")
    print("═"*60)
    import subprocess, sys
    subprocess.run(
        [sys.executable, "src/hyperparameter_sweep.py", "--sweep", sweep],
        check=True,
    )


def stage_evaluate(model: str = "all"):
    print("\n" + "═"*60)
    print(f"  STAGE 4 — Evaluate: {model}")
    print("═"*60)
    from src.evaluate import evaluate_model, save_classification_report
    from src.data_pipeline import load_data
    import json

    _, _, test_gen, class_names, _, _ = load_data()

    model_list = (
        [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE]
        if model == "all" else [model]
    )

    all_metrics = {}
    for name in model_list:
        try:
            m = evaluate_model(name, test_gen, class_names)
            all_metrics[name] = m
        except FileNotFoundError as e:
            print(f"⚠️  {e}")

    out = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Main] Metrics saved → {out}")


def stage_visualize(figure: str = "all"):
    print("\n" + "═"*60)
    print(f"  STAGE 5 — Visualize: {figure}")
    print("═"*60)
    import subprocess, sys
    subprocess.run(
        [sys.executable, "src/visualize.py", "--figure", figure],
        check=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BIL539 Plant Disease Detection — Full Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["download", "train", "sweep", "evaluate", "visualize", "all"],
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["custom_cnn", "efficientnetb0", "mobilenetv2", "all"],
        help="Model to train/evaluate (default: all)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["lr", "dropout", "optimizer", "batch", "all"],
        help="Which hyperparameter to sweep (default: all)",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["curves", "confusion", "sweep", "comparison", "all"],
        help="Which figure type to generate (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    args = parser.parse_args()

    _setup_gpu()

    if args.stage in ("download", "all"):
        stage_download()

    if args.stage in ("train", "all"):
        stage_train(model=args.model, batch_size=args.batch_size)

    if args.stage in ("sweep", "all"):
        stage_sweep(sweep=args.sweep)

    if args.stage in ("evaluate", "all"):
        stage_evaluate(model=args.model)

    if args.stage in ("visualize", "all"):
        stage_visualize(figure=args.figure)

    print("\n" + "═"*60)
    print("  ✅  Pipeline complete!")
    print(f"  Models   → {config.MODELS_DIR}")
    print(f"  Logs     → {config.LOGS_DIR}")
    print(f"  Figures  → {config.FIGURES_DIR}")
    print("═"*60)


if __name__ == "__main__":
    main()
