"""
src/visualize.py
----------------
Generates all figures needed for the paper:

  1. Training & validation accuracy/loss curves per model
  2. Confusion matrix heatmaps
  3. Hyperparameter sweep bar charts (LR, Dropout, Optimizer)
  4. Model comparison bar chart (final accuracy)

All figures are saved as high-DPI PNG files in results/figures/.

Usage:
    python src/visualize.py
    python src/visualize.py --figure curves        # only training curves
    python src/visualize.py --figure confusion
    python src/visualize.py --figure sweep
    python src/visualize.py --figure comparison
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DPI = 200


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load CSV training history
# ─────────────────────────────────────────────────────────────────────────────

def _load_history(model_name: str, phase: str = "") -> pd.DataFrame | None:
    """Load the CSVLogger output for a given model/phase."""
    tag      = f"{model_name}_{phase}" if phase else model_name
    csv_path = os.path.join(config.LOGS_DIR, f"{tag}_history.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _merge_phases(model_name: str) -> pd.DataFrame | None:
    """Merge phase1 + phase2 history for transfer models (if separate CSVs)."""
    p1 = _load_history(model_name, "phase1")
    p2 = _load_history(model_name, "phase2")

    if p1 is not None and p2 is not None:
        # Offset phase2 epoch numbers
        p2 = p2.copy()
        p2["epoch"] = p2["epoch"] + len(p1)
        merged = pd.concat([p1, p2], ignore_index=True)
        return merged
    if p1 is not None:
        return p1
    # Fall back to single-phase log
    return _load_history(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(model_name: str):
    df = _merge_phases(model_name)
    if df is None:
        print(f"[Visualize] No history CSV found for {model_name} — skipping curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {model_name.replace('_', ' ').title()}", fontsize=13)

    # Accuracy
    axes[0].plot(df["epoch"] + 1, df["accuracy"],     label="Train Acc",  linewidth=1.8)
    axes[0].plot(df["epoch"] + 1, df["val_accuracy"], label="Val Acc",    linewidth=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].set_ylim([0, 1.05])

    # Loss
    axes[1].plot(df["epoch"] + 1, df["loss"],     label="Train Loss", linewidth=1.8)
    axes[1].plot(df["epoch"] + 1, df["val_loss"], label="Val Loss",   linewidth=1.8, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_training_curves.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")


def plot_all_training_curves():
    for name in [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE]:
        plot_training_curves(name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model_name: str, class_names: list | None = None):
    cm_path = os.path.join(config.LOGS_DIR, f"{model_name}_confusion_matrix.npy")
    if not os.path.exists(cm_path):
        print(f"[Visualize] No confusion matrix found for {model_name} — skipping.")
        return

    cm = np.load(cm_path)
    n  = cm.shape[0]

    # Normalise to percentage per row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig_size = max(14, n // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 1))

    tick_labels = class_names if class_names else list(range(n))
    # Shorten labels for readability
    short_labels = [
        l.replace("___", "\n").replace("_", " ")
        for l in tick_labels
    ] if class_names else tick_labels

    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.3,
        linecolor="lightgrey",
        annot=(n <= 15),     # only annotate if ≤15 classes (readability)
        fmt=".1f",
        cbar_kws={"label": "Row %"},
    )
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()

    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter sweep charts
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(labels, values, xlabel, ylabel, title, out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=color, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([max(0, min(values) - 0.05), 1.05])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")


def plot_sweep_charts():
    sweep_path = os.path.join(config.LOGS_DIR, "sweep_results.json")
    if not os.path.exists(sweep_path):
        print("[Visualize] No sweep_results.json found — run hyperparameter_sweep.py first.")
        return

    with open(sweep_path) as f:
        sweeps = json.load(f)

    colour_map = {
        "lr_sweep"       : "steelblue",
        "dropout_sweep"  : "darkorange",
        "optimizer_sweep": "seagreen",
        "batch_sweep"    : "mediumpurple",
    }
    label_map = {
        "lr_sweep"       : ("Learning Rate", "lr"),
        "dropout_sweep"  : ("Dropout Rate",  "dropout"),
        "optimizer_sweep": ("Optimizer",      "optimizer"),
        "batch_sweep"    : ("Batch Size",     "batch_size"),
    }

    for sweep_name, runs in sweeps.items():
        if sweep_name not in label_map:
            continue
        xlabel_title, key = label_map[sweep_name]
        labels = [str(r[key]) for r in runs]
        values = [r["best_val_acc"] for r in runs]
        _bar_chart(
            labels, values,
            xlabel=xlabel_title,
            ylabel="Best Val Accuracy",
            title=f"EfficientNetB0 — {xlabel_title} Sweep",
            out_path=os.path.join(config.FIGURES_DIR, f"sweep_{sweep_name}.png"),
            color=colour_map.get(sweep_name, "steelblue"),
        )

    # Also plot convergence epochs for LR sweep
    if "lr_sweep" in sweeps:
        runs   = sweeps["lr_sweep"]
        labels = [str(r["lr"]) for r in runs]
        values = [r["convergence_ep"] for r in runs]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color="cornflowerblue", edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Convergence Epoch")
        ax.set_title("EfficientNetB0 — Convergence vs. Learning Rate")
        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, "sweep_lr_convergence.png")
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model comparison chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison():
    metrics_path = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
    if not os.path.exists(metrics_path):
        print("[Visualize] No all_model_metrics.json found — run evaluate.py first.")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    names  = list(all_metrics.keys())
    accs   = [all_metrics[n]["accuracy"] * 100 for n in names]
    f1s    = [all_metrics[n]["macro_f1"] for n in names]
    labels = [n.replace("_", "\n") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison on PlantVillage Test Set", fontsize=13)

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Accuracy
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="white")
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim([max(0, min(accs) - 10), 105])

    # Macro F1
    bars2 = axes[1].bar(labels, f1s, color=colors, edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.4f", padding=3)
    axes[1].set_ylabel("Macro F1 Score")
    axes[1].set_title("Macro F1 Score")
    axes[1].set_ylim([max(0, min(f1s) - 0.1), 1.05])

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "model_comparison.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")

    # ── Parameter efficiency chart ─────────────────────────────────────────
    params = [all_metrics[n].get("params_M", 0) for n in names]
    if any(p > 0 for p in params):
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, params, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f M", padding=3)
        ax.set_ylabel("Total Parameters (M)")
        ax.set_title("Model Size Comparison")
        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, "model_params.png")
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["curves", "confusion", "sweep", "comparison", "all"],
    )
    args = parser.parse_args()

    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Load class names if available (needed for confusion matrix labels)
    class_names = None
    try:
        from src.data_pipeline import build_dataframe
        _, class_names = build_dataframe()
    except Exception:
        pass

    if args.figure in ("curves", "all"):
        plot_all_training_curves()

    if args.figure in ("confusion", "all"):
        for name in [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE]:
            plot_confusion_matrix(name, class_names)

    if args.figure in ("sweep", "all"):
        plot_sweep_charts()

    if args.figure in ("comparison", "all"):
        plot_model_comparison()

    print("\n✅ All figures saved to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
