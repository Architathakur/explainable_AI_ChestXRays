import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai_pneumonia.model.cxr_torch_model import CXRMLPHead
from xai_pneumonia.utils import ensure_dir, get_project_root


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def best_with_constraint(
    rows: list[Dict[str, object]],
    constraint_key: str,
    minimum: float,
    sort_key: str = "f1",
) -> Optional[Dict[str, object]]:
    valid = [row for row in rows if float(row[constraint_key]) >= minimum]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row[sort_key]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the pneumonia decision threshold for a cached CXR foundation head.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=os.path.join(get_project_root(), "outputs", "threshold_tuning"))
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    features_npz = np.load(args.features)
    features = features_npz["features"].astype(np.float32)
    y_true = features_npz["labels"].astype(np.int64)

    head = CXRMLPHead(
        input_dim=int(checkpoint.get("feature_dim", features.shape[1])),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    with torch.no_grad():
        logits = head(torch.from_numpy(features))
        probabilities = torch.softmax(logits, dim=1).numpy()[:, 1]

    thresholds = np.arange(args.step, 1.0, args.step)
    rows = [metrics_at_threshold(y_true, probabilities, float(threshold)) for threshold in thresholds]
    best_f1 = max(rows, key=lambda row: float(row["f1"]))
    default_050 = metrics_at_threshold(y_true, probabilities, 0.5)
    best_recall_080 = best_with_constraint(rows, "recall", 0.80)
    best_precision_065 = best_with_constraint(rows, "precision", 0.65)

    summary = {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "auc_pr": float(average_precision_score(y_true, probabilities)),
        "default_0.50": default_050,
        "best_f1": best_f1,
        "best_f1_with_recall_at_least_0.80": best_recall_080,
        "best_f1_with_precision_at_least_0.65": best_precision_065,
    }

    output_path = os.path.join(output_dir, "threshold_tuning.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"AUC-ROC: {summary['auc_roc']:.4f}")
    print(f"AUC-PR: {summary['auc_pr']:.4f}")
    for label, row in [
        ("default_0.50", default_050),
        ("best_f1", best_f1),
        ("recall>=0.80", best_recall_080),
        ("precision>=0.65", best_precision_065),
    ]:
        if row is None:
            print(f"{label}: no threshold found")
            continue
        print(
            f"{label}: threshold={row['threshold']:.3f} "
            f"acc={row['accuracy']:.4f} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} f1={row['f1']:.4f} "
            f"tn={row['tn']} fp={row['fp']} fn={row['fn']} tp={row['tp']}"
        )
    print("\nFinal metrics table (best F1 threshold)")
    print("Metric | Value")
    print(f"Accuracy | {best_f1['accuracy']:.4f}")
    print(f"Precision | {best_f1['precision']:.4f}")
    print(f"Recall | {best_f1['recall']:.4f}")
    print(f"F1 | {best_f1['f1']:.4f}")
    print(f"AUC-ROC | {summary['auc_roc']:.4f}")
    print(f"AUC-PR | {summary['auc_pr']:.4f}")
    print(f"Confusion Matrix | [[{best_f1['tn']}, {best_f1['fp']}], [{best_f1['fn']}, {best_f1['tp']}]]")
    print(f"Saved threshold tuning summary to: {output_path}")


if __name__ == "__main__":
    main()
