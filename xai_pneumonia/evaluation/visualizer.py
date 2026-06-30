import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.utils import ensure_dir, get_project_root

sns.set_style("whitegrid")


def _figures_dir(output_dir: str = None) -> str:
    base_dir = output_dir or os.path.join(get_project_root(), "outputs", "figures")
    return ensure_dir(base_dir)


def plot_training_curves(history: Dict[str, list], output_dir: str = None) -> str:
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(history.get("loss", []), label="Train")
    axes[0].plot(history.get("val_loss", []), label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="Train")
    axes[1].plot(history.get("val_accuracy", []), label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(history.get("f1_pneumonia", []), label="Train")
    axes[2].plot(history.get("val_f1_pneumonia", []), label="Val")
    axes[2].set_title("F1 Pneumonia")
    axes[2].legend()

    axes[3].plot(history.get("lr", history.get("learning_rate", [])), label="Learning Rate")
    axes[3].set_title("Learning Rate Schedule")
    axes[3].legend()

    fig.tight_layout()
    path = os.path.join(figures_dir, "training_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, normalize: bool = True, output_dir: str = None) -> str:
    figures_dir = _figures_dir(output_dir)
    display = cm.astype(np.float32)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, where=row_sums != 0)

    annotations = np.empty_like(display, dtype=object)
    raw_cm = cm.astype(int)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                annotations[i, j] = f"{raw_cm[i, j]}\n{display[i, j]:.1%}"
            else:
                annotations[i, j] = str(raw_cm[i, j])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        display,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    path = os.path.join(figures_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_pr_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    output_dir: str = None,
) -> str:
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"AP = {ap:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(figures_dir, "roc_pr_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_xai_comparison(
    image: np.ndarray,
    heatmaps_dict: Dict[str, np.ndarray],
    label: str,
    prediction: str,
    confidence: float,
    case_id: str,
    output_dir: str = None,
) -> str:
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for axis, (method_name, overlay) in zip(axes[1:], heatmaps_dict.items()):
        axis.imshow(overlay)
        axis.set_title(method_name)
        axis.axis("off")

    fig.suptitle(f"True: {label} | Predicted: {prediction} ({confidence:.1%})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(figures_dir, f"xai_comparison_{case_id}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aggregate_xai_table(results_dict, output_dir: str = None) -> str:
    figures_dir = _figures_dir(output_dir)
    if isinstance(results_dict, pd.DataFrame):
        df = results_dict.copy()
    else:
        df = pd.DataFrame(results_dict)
    if "Metric" in df.columns:
        df = df.set_index("Metric")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    path = os.path.join(figures_dir, "aggregate_xai_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ig_baseline_comparison(
    image: np.ndarray,
    ig_black: np.ndarray,
    ig_noise: np.ndarray,
    ig_mean: np.ndarray,
    case_id: str,
    output_dir: str = None,
) -> str:
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["Original", "IG Black", "IG Noise", "IG Mean"]
    panels = [image, ig_black, ig_noise, ig_mean]
    for axis, title, panel in zip(axes, titles, panels):
        axis.imshow(panel, cmap="gray" if title == "Original" else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    path = os.path.join(figures_dir, f"ig_baseline_comparison_{case_id}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gradcam_layer_comparison(
    image: np.ndarray,
    heatmaps_by_layer: Dict[str, np.ndarray],
    case_id: str,
    output_dir: str = None,
) -> str:
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    for axis, (layer_name, overlay) in zip(axes[1:], heatmaps_by_layer.items()):
        axis.imshow(overlay)
        axis.set_title(layer_name)
        axis.axis("off")
    fig.tight_layout()
    path = os.path.join(figures_dir, f"gradcam_layer_comparison_{case_id}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
