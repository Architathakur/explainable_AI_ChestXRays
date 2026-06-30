import os
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from skimage.metrics import structural_similarity

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.utils import bbox_union_mask


def predict_split(
    model: tf.keras.Model,
    data_module: RSNAPneumoniaDataModule,
    split_name: str,
    batch_size: int = 32,
    sample_limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    dataset = data_module.make_tf_dataset(
        split_name,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        sample_limit=sample_limit,
    )
    probabilities = model.predict(dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    split_df = data_module.get_split_dataframe(split_name)
    if sample_limit is not None and sample_limit > 0:
        split_df = split_df.iloc[:sample_limit].copy()
    true_labels = split_df["label"].to_numpy(dtype=np.int32)
    return {
        "patient_ids": split_df["patient_id"].to_numpy(dtype=str),
        "y_true": true_labels,
        "y_pred": predicted_labels.astype(np.int32),
        "y_prob": probabilities.astype(np.float32),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_per_class": {0: float(precision[0]), 1: float(precision[1])},
        "recall_per_class": {0: float(recall[0]), 1: float(recall[1])},
        "f1_per_class": {0: float(f1[0]), 1: float(f1[1])},
        "support_per_class": {0: int(support[0]), 1: int(support[1])},
        "macro_f1": float(np.mean(f1)),
        "specificity": float(specificity),
        "auc_roc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "auc_pr": float(average_precision_score(y_true, y_prob[:, 1])),
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "classification_report_text": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "classification_report_dict": classification_report(
            y_true,
            y_pred,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"precision": precision_curve, "recall": recall_curve},
    }


def compute_pointing_game(heatmap: np.ndarray, bbox_mask: np.ndarray) -> float:
    max_index = int(np.argmax(heatmap))
    row, col = np.unravel_index(max_index, heatmap.shape)
    return float(bbox_mask[row, col] > 0)


def compute_iou(heatmap: np.ndarray, bbox_mask: np.ndarray, percentile: float = 50.0) -> float:
    threshold = np.percentile(heatmap, percentile)
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    intersection = np.logical_and(binary_mask > 0, bbox_mask > 0).sum()
    union = np.logical_or(binary_mask > 0, bbox_mask > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_energy_in_box(heatmap: np.ndarray, bbox_mask: np.ndarray) -> float:
    total_energy = float(np.sum(heatmap))
    if total_energy <= 0:
        return 0.0
    return float(np.sum(heatmap * bbox_mask) / total_energy)


def evaluate_localization_metrics(
    data_module: RSNAPneumoniaDataModule,
    heatmaps_by_method: Dict[str, Dict[str, np.ndarray]],
    split_name: str = "test",
) -> Dict[str, Dict[str, object]]:
    split_df = data_module.get_split_dataframe(split_name)
    positive_df = split_df[(split_df["label"] == 1) & (split_df["bboxes"].map(len) > 0)]
    results: Dict[str, Dict[str, object]] = {}

    for method_name, method_heatmaps in heatmaps_by_method.items():
        pointing_scores: List[float] = []
        ious: List[float] = []
        energy_ratios: List[float] = []

        for _, row in positive_df.iterrows():
            patient_id = row["patient_id"]
            if patient_id not in method_heatmaps:
                continue
            heatmap = method_heatmaps[patient_id]
            bbox_mask = bbox_union_mask(row["bboxes"], data_module.image_size)
            pointing_scores.append(compute_pointing_game(heatmap, bbox_mask))
            ious.append(compute_iou(heatmap, bbox_mask))
            energy_ratios.append(compute_energy_in_box(heatmap, bbox_mask))

        results[method_name] = {
            "pointing_game_scores": pointing_scores,
            "ious": ious,
            "energy_in_box_scores": energy_ratios,
            "pointing_game_accuracy": float(np.mean(pointing_scores) * 100.0) if pointing_scores else 0.0,
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_std": float(np.std(ious)) if ious else 0.0,
            "energy_in_box_mean": float(np.mean(energy_ratios)) if energy_ratios else 0.0,
            "energy_in_box_std": float(np.std(energy_ratios)) if energy_ratios else 0.0,
        }
    return results


def _generate_heatmap(
    explainer: object,
    image: np.ndarray,
    image_rgb: np.ndarray,
) -> np.ndarray:
    output = explainer.generate(image, image_rgb)
    if isinstance(output, tuple):
        heatmap = output[0]
    else:
        heatmap = output
    return heatmap.astype(np.float32)


def select_stability_subset(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[str]:
    quadrants = {
        "correct_pneumonia": [],
        "wrong_pneumonia": [],
        "correct_normal": [],
        "wrong_normal": [],
    }
    for patient_id, true_label, pred_label in zip(patient_ids, y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            quadrants["correct_pneumonia"].append(patient_id)
        elif true_label == 1 and pred_label == 0:
            quadrants["wrong_normal"].append(patient_id)
        elif true_label == 0 and pred_label == 0:
            quadrants["correct_normal"].append(patient_id)
        else:
            quadrants["wrong_pneumonia"].append(patient_id)

    rng = np.random.default_rng(42)
    selected: List[str] = []
    for ids in quadrants.values():
        if not ids:
            continue
        take = min(50, len(ids))
        chosen = rng.choice(np.array(ids, dtype=object), size=take, replace=False).tolist()
        selected.extend(chosen)
    return selected


def compute_stability_scores(
    data_module: RSNAPneumoniaDataModule,
    explainer_factories: Dict[str, Callable[[], object]],
    patient_ids: List[str],
    noise_sigma: float = 0.01,
    variants_per_image: int = 5,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for method_name, factory in explainer_factories.items():
        explainer = factory()
        ssim_scores: List[float] = []
        for patient_id in patient_ids:
            sample = data_module.load_sample(patient_id, augment=False, normalize=True)
            base_heatmap = _generate_heatmap(explainer, sample["image"], sample["image_rgb"])
            for _ in range(variants_per_image):
                noisy_rgb = np.clip(
                    sample["image_rgb"] + np.random.normal(0.0, noise_sigma, sample["image_rgb"].shape).astype(np.float32),
                    0.0,
                    1.0,
                )
                noisy_standardized = ((noisy_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) /
                                      np.array([0.229, 0.224, 0.225], dtype=np.float32))
                noisy_heatmap = _generate_heatmap(explainer, noisy_standardized, noisy_rgb)
                ssim_scores.append(
                    float(
                        structural_similarity(
                            base_heatmap,
                            noisy_heatmap,
                            data_range=1.0,
                        )
                    )
                )
        results[method_name] = {
            "ssim_scores": ssim_scores,
            "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            "ssim_std": float(np.std(ssim_scores)) if ssim_scores else 0.0,
        }
    return results


def measure_efficiency(
    data_module: RSNAPneumoniaDataModule,
    explainer_factories: Dict[str, Callable[[], object]],
    patient_ids: List[str],
    device_name: str,
    warmup: int,
    runs: int,
) -> Dict[str, Dict[str, float]]:
    timings: Dict[str, Dict[str, float]] = {}
    device_available = True
    if device_name.upper().startswith("/GPU"):
        device_available = bool(tf.config.list_physical_devices("GPU"))

    if not device_available:
        for method_name in explainer_factories:
            timings[method_name] = {"mean_ms": float("nan"), "std_ms": float("nan")}
        return timings

    effective_ids = patient_ids[:runs]
    with tf.device(device_name):
        for method_name, factory in explainer_factories.items():
            explainer = factory()
            samples = [data_module.load_sample(pid, augment=False, normalize=True) for pid in effective_ids]
            for sample in samples[: min(warmup, len(samples))]:
                _generate_heatmap(explainer, sample["image"], sample["image_rgb"])

            per_image_times: List[float] = []
            for sample in samples:
                start = time.perf_counter()
                _generate_heatmap(explainer, sample["image"], sample["image_rgb"])
                per_image_times.append((time.perf_counter() - start) * 1000.0)
            timings[method_name] = {
                "mean_ms": float(np.mean(per_image_times)) if per_image_times else float("nan"),
                "std_ms": float(np.std(per_image_times)) if per_image_times else float("nan"),
            }
    return timings


def build_xai_results_table(
    localization_results: Dict[str, Dict[str, object]],
    stability_results: Dict[str, Dict[str, object]],
    gpu_timings: Dict[str, Dict[str, float]],
    cpu_timings: Dict[str, Dict[str, float]],
    wilcoxon_iou: Dict[str, Dict[str, float]],
    wilcoxon_stability: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    method_order = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients"]
    rows = [
        {
            "Metric": "Pointing Game (%)",
            **{method: localization_results[method]["pointing_game_accuracy"] for method in method_order},
        },
        {
            "Metric": "Mean IoU",
            **{method: localization_results[method]["iou_mean"] for method in method_order},
        },
        {
            "Metric": "Energy-in-Box",
            **{method: localization_results[method]["energy_in_box_mean"] for method in method_order},
        },
        {
            "Metric": "SSIM Stability",
            **{method: stability_results[method]["ssim_mean"] for method in method_order},
        },
        {
            "Metric": "GPU Time (ms)",
            **{method: gpu_timings[method]["mean_ms"] for method in method_order},
        },
        {
            "Metric": "CPU Time (ms)",
            **{method: cpu_timings[method]["mean_ms"] for method in method_order},
        },
        {
            "Metric": "Wilcoxon p vs GC",
            "Grad-CAM": np.nan,
            "Grad-CAM++": wilcoxon_iou.get("Grad-CAM_vs_Grad-CAM++", {}).get("corrected_p_value", np.nan),
            "Integrated Gradients": wilcoxon_iou.get("Grad-CAM_vs_Integrated Gradients", {}).get("corrected_p_value", np.nan),
        },
        {
            "Metric": "Wilcoxon p vs GC++",
            "Grad-CAM": wilcoxon_iou.get("Grad-CAM_vs_Grad-CAM++", {}).get("corrected_p_value", np.nan),
            "Grad-CAM++": np.nan,
            "Integrated Gradients": wilcoxon_iou.get("Grad-CAM++_vs_Integrated Gradients", {}).get("corrected_p_value", np.nan),
        },
        {
            "Metric": "Stability p vs GC",
            "Grad-CAM": np.nan,
            "Grad-CAM++": wilcoxon_stability.get("Grad-CAM_vs_Grad-CAM++", {}).get("p_value", np.nan),
            "Integrated Gradients": wilcoxon_stability.get("Grad-CAM_vs_Integrated Gradients", {}).get("p_value", np.nan),
        },
        {
            "Metric": "Stability p vs GC++",
            "Grad-CAM": wilcoxon_stability.get("Grad-CAM_vs_Grad-CAM++", {}).get("p_value", np.nan),
            "Grad-CAM++": np.nan,
            "Integrated Gradients": wilcoxon_stability.get("Grad-CAM++_vs_Integrated Gradients", {}).get("p_value", np.nan),
        },
    ]
    return pd.DataFrame(rows)
