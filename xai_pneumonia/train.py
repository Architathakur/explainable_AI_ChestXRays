import argparse
import os
import random
from typing import Dict, List

import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.evaluation.metrics import compute_classification_metrics, predict_split
from xai_pneumonia.evaluation.visualizer import plot_training_curves
from xai_pneumonia.model.resnet50_model import (
    build_resnet50_model,
    compile_model,
    get_training_callbacks,
    save_model_history,
    unfreeze_last_layers,
)
from xai_pneumonia.utils import ensure_dir, get_project_root, set_global_determinism
from xai_pneumonia.xai.layer_selector import select_best_gradcam_layer


def _merge_histories(phase1_history: Dict[str, List[float]], phase2_history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    all_keys = set(phase1_history.keys()) | set(phase2_history.keys())
    phase1_epochs = len(next(iter(phase1_history.values()))) if phase1_history else 0
    phase2_epochs = len(next(iter(phase2_history.values()))) if phase2_history else 0
    for key in all_keys:
        part1 = phase1_history.get(key, [None] * phase1_epochs)
        part2 = phase2_history.get(key, [None] * phase2_epochs)
        merged[key] = part1 + part2

    if "lr" not in merged and "learning_rate" not in merged:
        merged["lr"] = [1e-3] * phase1_epochs + phase2_history.get("lr", phase2_history.get("learning_rate", [1e-4] * phase2_epochs))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an explainable pneumonia detection model on RSNA Chest X-rays.")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/rsna-pneumonia-detection-challenge/")
    parser.add_argument("--output_dir", type=str, default=os.path.join(get_project_root(), "outputs"))
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of phase 2 fine-tuning epochs.")
    parser.add_argument("--phase1_epochs", type=int, default=5, help="Frozen-backbone warmup epochs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune_layers", type=int, default=30, help="Number of ResNet layers to unfreeze for phase 2. Use 0 to skip fine-tuning.")
    parser.add_argument("--fit_verbose", type=int, default=1, choices=[0, 1, 2], help="Keras fit verbosity. Use 2 for one line per epoch.")
    parser.add_argument("--train_limit", type=int, default=None, help="Optional train sample limit for smoke tests.")
    parser.add_argument("--val_limit", type=int, default=None, help="Optional validation sample limit for smoke tests.")
    parser.add_argument("--skip_mean_image", action="store_true", help="Skip mean-image creation needed only for Integrated Gradients.")
    parser.add_argument("--skip_layer_selection", action="store_true", help="Skip Grad-CAM layer selection during training.")
    args = parser.parse_args()

    set_global_determinism(42)
    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(os.path.join(get_project_root(), "model"))

    data_module = RSNAPneumoniaDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        allow_test_access=False,
    )
    print("Split summary:")
    print(data_module.get_split_summary().to_string(index=False))

    class_weights = data_module.compute_class_weights()
    print(f"Class weights: {class_weights}")
    if args.skip_mean_image:
        print("Skipping mean training image creation.")
    else:
        mean_image = data_module.compute_mean_training_image(sample_size=1000, force=False)
        print(f"Mean training image saved with shape: {mean_image.shape}")

    train_dataset = data_module.make_tf_dataset(
        "train",
        training=True,
        shuffle=True,
        batch_size=args.batch_size,
        sample_limit=args.train_limit,
    )
    val_dataset = data_module.make_tf_dataset(
        "val",
        training=False,
        shuffle=False,
        batch_size=args.batch_size,
        sample_limit=args.val_limit,
    )

    model = build_resnet50_model(freeze_backbone=True, learning_rate=1e-3)

    print("\nPhase 1: Training classification head")
    phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.phase1_epochs,
        class_weight=class_weights,
        verbose=args.fit_verbose,
    )
    phase1_path = os.path.join(model_dir, "phase1_model.h5")
    model.save(phase1_path)

    phase2_history: Dict[str, List[float]] = {}
    if args.epochs > 0 and args.fine_tune_layers > 0:
        print(f"\nPhase 2: Fine-tuning last {args.fine_tune_layers} ResNet layers")
        unfreeze_last_layers(model, n_layers=args.fine_tune_layers)
        compile_model(model, learning_rate=1e-4)
        phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=get_training_callbacks(output_dir=output_dir, patience=10),
            verbose=args.fit_verbose,
        )
        phase2_history = phase2.history
    else:
        print("\nSkipping phase 2 fine-tuning.")

    history = _merge_histories(phase1.history, phase2_history)
    history_path = save_model_history(history, os.path.join(model_dir, "history.json"))
    plot_training_curves(history, os.path.join(output_dir, "figures"))

    best_model_path = os.path.join(model_dir, "best_model.h5")
    model.save(best_model_path)

    layer_scores = {}
    if args.skip_layer_selection:
        print("\nSkipping Grad-CAM target-layer selection.")
    else:
        print("\nSelecting the best Grad-CAM target layer on the validation set")
        layer_scores = select_best_gradcam_layer(model, data_module)

    val_predictions = predict_split(
        model,
        data_module,
        "val",
        batch_size=args.batch_size,
        sample_limit=args.val_limit,
    )
    val_metrics = compute_classification_metrics(
        val_predictions["y_true"],
        val_predictions["y_pred"],
        val_predictions["y_prob"],
    )

    print("\nValidation metrics summary")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1 Pneumonia: {val_metrics['f1_per_class'][1]:.4f}")
    print(f"AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {val_metrics['auc_pr']:.4f}")
    print("\nClassification report:")
    print(val_metrics["classification_report_text"])
    if layer_scores:
        print("Layer selection IoU means:")
        for layer_name, score in layer_scores.items():
            print(f"  {layer_name}: {score:.4f}")
    print(f"\nSaved phase 1 model to: {phase1_path}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
