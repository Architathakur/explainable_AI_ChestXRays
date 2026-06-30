import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.evaluation.metrics import (
    build_xai_results_table,
    compute_classification_metrics,
    compute_stability_scores,
    evaluate_localization_metrics,
    measure_efficiency,
    predict_split,
    select_stability_subset,
)
from xai_pneumonia.evaluation.statistical_tests import (
    run_iou_wilcoxon_tests,
    run_mcnemar_test,
    run_stability_wilcoxon_tests,
)
from xai_pneumonia.evaluation.visualizer import (
    plot_aggregate_xai_table,
    plot_confusion_matrix,
    plot_roc_pr_curves,
)
from xai_pneumonia.model.resnet50_model import load_trained_model
from xai_pneumonia.utils import ensure_dir, get_project_root, save_json, set_global_determinism
from xai_pneumonia.xai.gradcam import GradCAM
from xai_pneumonia.xai.gradcam_plus_plus import GradCAMPlusPlus
from xai_pneumonia.xai.integrated_gradients import IntegratedGradients


def _load_heatmaps(heatmaps_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    suffix_map = {
        "Grad-CAM": "gradcam",
        "Grad-CAM++": "gradcam_plus_plus",
        "Integrated Gradients": "integrated_gradients",
    }
    heatmaps_by_method: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in suffix_map}
    for method_name, suffix in suffix_map.items():
        for file_name in os.listdir(heatmaps_dir):
            if file_name.endswith(f"_{suffix}.npy"):
                patient_id = file_name[: -(len(suffix) + 5)]
                heatmaps_by_method[method_name][patient_id] = np.load(os.path.join(heatmaps_dir, file_name))
    return heatmaps_by_method


def _build_explainer_factories(model: tf.keras.Model) -> Dict[str, object]:
    return {
        "Grad-CAM": lambda: GradCAM(model),
        "Grad-CAM++": lambda: GradCAMPlusPlus(model),
        "Integrated Gradients": lambda: IntegratedGradients(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the explainable pneumonia detection pipeline on the locked test set.")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/rsna-pneumonia-detection-challenge/")
    parser.add_argument("--model_path", type=str, default=os.path.join(get_project_root(), "model", "best_model.h5"))
    parser.add_argument("--heatmaps_dir", type=str, default=os.path.join(get_project_root(), "outputs", "heatmaps"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(get_project_root(), "outputs"))
    args = parser.parse_args()

    set_global_determinism(42)
    output_dir = ensure_dir(args.output_dir)
    report_path = os.path.join(output_dir, "evaluation_report.txt")

    data_module = RSNAPneumoniaDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        allow_test_access=True,
    )
    final_model = load_trained_model(args.model_path)
    phase1_model_path = os.path.join(get_project_root(), "model", "phase1_model.h5")
    if not os.path.exists(phase1_model_path):
        raise FileNotFoundError(f"Phase 1 model required for McNemar test is missing: {phase1_model_path}")
    phase1_model = load_trained_model(phase1_model_path)

    final_predictions = predict_split(final_model, data_module, "test", batch_size=32)
    phase1_predictions = predict_split(phase1_model, data_module, "test", batch_size=32)
    classification_metrics = compute_classification_metrics(
        final_predictions["y_true"],
        final_predictions["y_pred"],
        final_predictions["y_prob"],
    )

    heatmaps_by_method = _load_heatmaps(args.heatmaps_dir)
    localization_results = evaluate_localization_metrics(data_module, heatmaps_by_method, split_name="test")

    explainer_factories = _build_explainer_factories(final_model)
    stability_ids = select_stability_subset(
        final_predictions["patient_ids"],
        final_predictions["y_true"],
        final_predictions["y_pred"],
    )
    stability_results = compute_stability_scores(
        data_module,
        explainer_factories,
        stability_ids,
        noise_sigma=0.01,
        variants_per_image=5,
    )

    gpu_ids = final_predictions["patient_ids"][:100].tolist()
    cpu_ids = final_predictions["patient_ids"][:20].tolist()
    gpu_timings = measure_efficiency(data_module, explainer_factories, gpu_ids, "/GPU:0", warmup=10, runs=100)
    cpu_timings = measure_efficiency(data_module, explainer_factories, cpu_ids, "/CPU:0", warmup=5, runs=20)

    mcnemar_results = run_mcnemar_test(
        final_predictions["y_true"],
        phase1_predictions["y_pred"],
        final_predictions["y_pred"],
    )
    wilcoxon_iou = run_iou_wilcoxon_tests(localization_results)
    wilcoxon_stability = run_stability_wilcoxon_tests(stability_results)
    xai_table = build_xai_results_table(
        localization_results,
        stability_results,
        gpu_timings,
        cpu_timings,
        wilcoxon_iou,
        wilcoxon_stability,
    )

    figures_dir = os.path.join(output_dir, "figures")
    plot_confusion_matrix(classification_metrics["confusion_matrix"], normalize=True, output_dir=figures_dir)
    plot_roc_pr_curves(
        classification_metrics["roc_curve"]["fpr"],
        classification_metrics["roc_curve"]["tpr"],
        classification_metrics["auc_roc"],
        classification_metrics["pr_curve"]["precision"],
        classification_metrics["pr_curve"]["recall"],
        classification_metrics["auc_pr"],
        output_dir=figures_dir,
    )
    plot_aggregate_xai_table(
        xai_table,
        output_dir=figures_dir,
    )

    report_lines = [
        "Explainable AI for Pneumonia Detection in Chest X-rays",
        "=" * 60,
        "",
        "Classification Metrics",
        f"Accuracy: {classification_metrics['accuracy']:.4f}",
        f"Specificity: {classification_metrics['specificity']:.4f}",
        f"F1 Normal: {classification_metrics['f1_per_class'][0]:.4f}",
        f"F1 Pneumonia: {classification_metrics['f1_per_class'][1]:.4f}",
        f"Macro F1: {classification_metrics['macro_f1']:.4f}",
        f"AUC-ROC: {classification_metrics['auc_roc']:.4f}",
        f"AUC-PR: {classification_metrics['auc_pr']:.4f}",
        "",
        classification_metrics["classification_report_text"],
        "",
        "McNemar Test",
        json.dumps(mcnemar_results, indent=2),
        "",
        "XAI Comparison Table",
        xai_table.to_string(index=False),
        "",
        "Wilcoxon IoU Tests",
        json.dumps(wilcoxon_iou, indent=2),
        "",
        "Wilcoxon Stability Tests",
        json.dumps(wilcoxon_stability, indent=2),
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    save_json(
        {
            "classification_metrics": {
                key: value
                for key, value in classification_metrics.items()
                if key not in ["confusion_matrix", "confusion_matrix_normalized", "roc_curve", "pr_curve"]
            },
            "mcnemar": mcnemar_results,
            "wilcoxon_iou": wilcoxon_iou,
            "wilcoxon_stability": wilcoxon_stability,
            "xai_table": xai_table.to_dict(orient="records"),
        },
        os.path.join(output_dir, "evaluation_summary.json"),
    )

    print("\nClassification metrics")
    print(f"Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"Specificity: {classification_metrics['specificity']:.4f}")
    print(f"F1 Pneumonia: {classification_metrics['f1_per_class'][1]:.4f}")
    print(f"AUC-ROC: {classification_metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {classification_metrics['auc_pr']:.4f}")
    print("\nXAI comparison")
    print(xai_table.to_string(index=False))
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
