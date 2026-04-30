"""
src/hyperparameter_sweep.py
---------------------------
Systematic hyperparameter sweep on EfficientNetB0 as described in §IV.F.

Sweeps:
  1. Learning rate   : [0.001, 0.0005, 0.0001]
  2. Dropout rate    : [0.2, 0.4, 0.5]
  3. Optimizer       : [adam, sgd, rmsprop]
  4. Batch size      : [16, 32, 64]

Each run trains for SWEEP_EPOCHS epochs (feature-extraction only, no fine-tuning)
to keep total compute manageable. Results are saved to results/logs/sweep_results.json.

Usage:
    python src/hyperparameter_sweep.py
    python src/hyperparameter_sweep.py --sweep lr          # only LR sweep
    python src/hyperparameter_sweep.py --sweep dropout
    python src/hyperparameter_sweep.py --sweep optimizer
    python src/hyperparameter_sweep.py --sweep batch
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_pipeline import load_data
from src.models import build_efficientnetb0

SWEEP_RESULTS_PATH = os.path.join(config.LOGS_DIR, "sweep_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# Run a single sweep experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    run_id: str,
    train_gen,
    val_gen,
    class_weights,
    lr: float       = config.LEARNING_RATE,
    dropout: float  = config.DROPOUT_RATE,
    optimizer: str  = config.OPTIMIZER,
    batch_size: int = config.BATCH_SIZE,
    epochs: int     = config.SWEEP_EPOCHS,
) -> dict:
    """
    Build and train EfficientNetB0 (feature-extraction only) with given hyperparams.
    Returns a result dict.
    """
    print(f"\n{'─'*60}")
    print(f"  Run: {run_id}")
    print(f"  lr={lr}  dropout={dropout}  optimizer={optimizer}  batch={batch_size}")
    print(f"{'─'*60}")

    # Rebuild generators if batch size differs from default
    # (generators are passed externally; batch_size is noted but not rebuilt here
    #  for performance — caller manages per-batch-size generators)

    model, _ = build_efficientnetb0(
        dropout_rate=dropout,
        lr=lr,
        optimizer=optimizer,
    )

    log_path = os.path.join(config.LOGS_DIR, f"sweep_{run_id}.csv")
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(log_path),
    ]

    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    n_epochs = len(history.history["loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    final_val_acc = history.history["val_accuracy"][-1]
    convergence   = int(history.history["val_accuracy"].index(best_val_acc)) + 1

    result = {
        "run_id"          : run_id,
        "lr"              : lr,
        "dropout"         : dropout,
        "optimizer"       : optimizer,
        "batch_size"      : batch_size,
        "epochs_run"      : n_epochs,
        "best_val_acc"    : float(best_val_acc),
        "final_val_acc"   : float(final_val_acc),
        "convergence_ep"  : convergence,
        "time_per_ep_s"   : float(elapsed / n_epochs),
    }

    print(f"  → best_val_acc={best_val_acc:.4f}  "
          f"convergence_ep={convergence}  "
          f"time/ep={elapsed/n_epochs:.1f}s")

    # Free GPU memory
    del model
    tf.keras.backend.clear_session()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sweep functions
# ─────────────────────────────────────────────────────────────────────────────

def sweep_learning_rate(train_gen, val_gen, class_weights) -> list:
    print("\n" + "═"*60)
    print("  SWEEP: Learning Rate")
    print("═"*60)
    results = []
    for lr in config.SWEEP_LR:
        rid = f"lr_{lr}"
        r = run_experiment(rid, train_gen, val_gen, class_weights, lr=lr)
        results.append(r)
    return results


def sweep_dropout(train_gen, val_gen, class_weights) -> list:
    print("\n" + "═"*60)
    print("  SWEEP: Dropout Rate")
    print("═"*60)
    results = []
    for d in config.SWEEP_DROPOUT:
        rid = f"dropout_{d}"
        r = run_experiment(rid, train_gen, val_gen, class_weights, dropout=d)
        results.append(r)
    return results


def sweep_optimizer(train_gen, val_gen, class_weights) -> list:
    print("\n" + "═"*60)
    print("  SWEEP: Optimizer")
    print("═"*60)
    results = []
    for opt in config.SWEEP_OPTIMIZERS:
        rid = f"optimizer_{opt}"
        r = run_experiment(rid, train_gen, val_gen, class_weights, optimizer=opt)
        results.append(r)
    return results


def sweep_batch_size(train_gen_16, train_gen_32, train_gen_64,
                     val_gen, class_weights) -> list:
    """
    Batch-size sweep requires three separate generators (each with different batch_size).
    The caller must supply them.
    """
    print("\n" + "═"*60)
    print("  SWEEP: Batch Size")
    print("═"*60)
    results = []
    for bs, tg in zip([16, 32, 64], [train_gen_16, train_gen_32, train_gen_64]):
        rid = f"batch_{bs}"
        r = run_experiment(rid, tg, val_gen, class_weights, batch_size=bs)
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save / load results
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_results() -> dict:
    if os.path.exists(SWEEP_RESULTS_PATH):
        with open(SWEEP_RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(all_results: dict):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(SWEEP_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Sweep] Results saved → {SWEEP_RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["lr", "dropout", "optimizer", "batch", "all"],
    )
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    all_results = load_existing_results()

    # Load default generators (batch_size=32)
    print("\n[Sweep] Loading dataset with batch_size=32 …")
    train_gen, val_gen, _, _, class_weights, _ = load_data(batch_size=32)

    if args.sweep in ("lr", "all") and "lr_sweep" not in all_results:
        all_results["lr_sweep"] = sweep_learning_rate(train_gen, val_gen, class_weights)
        save_results(all_results)

    if args.sweep in ("dropout", "all") and "dropout_sweep" not in all_results:
        all_results["dropout_sweep"] = sweep_dropout(train_gen, val_gen, class_weights)
        save_results(all_results)

    if args.sweep in ("optimizer", "all") and "optimizer_sweep" not in all_results:
        all_results["optimizer_sweep"] = sweep_optimizer(train_gen, val_gen, class_weights)
        save_results(all_results)

    if args.sweep in ("batch", "all") and "batch_sweep" not in all_results:
        print("\n[Sweep] Loading extra generators for batch-size sweep …")
        train_gen_16, val_gen_16, _, _, _, _ = load_data(batch_size=16)
        train_gen_64, val_gen_64, _, _, _, _ = load_data(batch_size=64)
        all_results["batch_sweep"] = sweep_batch_size(
            train_gen_16, train_gen, train_gen_64, val_gen, class_weights
        )
        save_results(all_results)

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SWEEP RESULTS SUMMARY")
    print("═"*60)
    for sweep_name, runs in all_results.items():
        print(f"\n  {sweep_name.upper()}")
        print(f"  {'Run ID':<30} {'Best ValAcc':>12} {'Conv.Ep':>8} {'s/ep':>8}")
        print(f"  {'─'*30} {'─'*12} {'─'*8} {'─'*8}")
        for r in runs:
            print(f"  {r['run_id']:<30} {r['best_val_acc']:>12.4f} "
                  f"{r['convergence_ep']:>8} {r['time_per_ep_s']:>8.1f}")


if __name__ == "__main__":
    main()
