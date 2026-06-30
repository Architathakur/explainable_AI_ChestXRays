import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.utils import bbox_union_mask, ensure_dir, get_project_root
from xai_pneumonia.xai.gradcam import GradCAM


def _compute_iou(binary_mask: np.ndarray, bbox_mask: np.ndarray) -> float:
    intersection = np.logical_and(binary_mask > 0, bbox_mask > 0).sum()
    union = np.logical_or(binary_mask > 0, bbox_mask > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def select_best_gradcam_layer(
    model: tf.keras.Model,
    data_module: RSNAPneumoniaDataModule,
    candidate_layers: Optional[List[str]] = None,
    sample_size: int = 200,
) -> Dict[str, float]:
    candidate_layers = candidate_layers or [
        "conv3_block4_out",
        "conv4_block6_out",
        "conv5_block3_out",
    ]
    val_df = data_module.get_split_dataframe("val")
    positive_annotated = val_df[(val_df["label"] == 1) & (val_df["bboxes"].map(len) > 0)]
    rng = np.random.default_rng(42)
    if len(positive_annotated) > sample_size:
        positive_annotated = positive_annotated.iloc[rng.choice(len(positive_annotated), size=sample_size, replace=False)]

    scores: Dict[str, float] = {}
    for layer_name in candidate_layers:
        explainer = GradCAM(model, target_layer_name=layer_name)
        ious: List[float] = []
        for _, row in positive_annotated.iterrows():
            sample = data_module.load_sample(row["patient_id"], augment=False, normalize=True)
            heatmap, _ = explainer.generate(sample["image"], sample["image_rgb"])
            binary_mask = (heatmap >= np.percentile(heatmap, 50)).astype(np.uint8)
            bbox_mask = bbox_union_mask(sample["bboxes"], data_module.image_size)
            ious.append(_compute_iou(binary_mask, bbox_mask))
        scores[layer_name] = float(np.mean(ious)) if ious else 0.0

    table = pd.DataFrame(
        [{"layer": layer_name, "mean_iou": score} for layer_name, score in scores.items()]
    ).sort_values("mean_iou", ascending=False)
    print("\nGrad-CAM Layer Selection Results")
    print(table.to_string(index=False))

    best_layer = table.iloc[0]["layer"]
    project_root = get_project_root()
    model_dir = ensure_dir(os.path.join(project_root, "model"))
    output_path = os.path.join(model_dir, "best_gradcam_layer.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(str(best_layer))
    print(f"\nSelected target layer: {best_layer}")
    return scores
