import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.model.cxr_torch_model import CXRClassifier, CXRFeatureExtractor, CXRMLPHead
from xai_pneumonia.train_cxr_foundation import RSNATorchDataset, get_device, make_feature_cache_path
from xai_pneumonia.utils import bbox_union_mask, ensure_dir, get_project_root, normalize_map, overlay_heatmap_on_grayscale


METHODS = ("Grad-CAM", "Grad-CAM++", "Integrated Gradients")


@dataclass
class ExplanationResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    probability: float
    predicted_label: int


def read_xray_image(path: str, image_size: int = 224) -> Tuple[np.ndarray, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        ds = pydicom.dcmread(path)
        image = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = np.max(image) - image
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        image = image.astype(np.float32)

    image -= np.min(image)
    max_value = float(np.max(image))
    if max_value > 0:
        image /= max_value
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    display = image.astype(np.float32)
    model_input = (2.0 * display - 1.0) * 1024.0
    tensor = torch.from_numpy(model_input[None, None, :, :].astype(np.float32))
    return display, tensor


def load_model(checkpoint_path: str, weights: str, hidden_dim: int, dropout: float, device: torch.device, xrv_cache_dir: Optional[str] = None) -> CXRClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extractor = CXRFeatureExtractor(weights=checkpoint.get("weights", weights), cache_dir=xrv_cache_dir).to(device)
    extractor.freeze()
    head = CXRMLPHead(
        input_dim=int(checkpoint.get("feature_dim", extractor.feature_dim)),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    if "encoder_state_dict" in checkpoint:
        extractor.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model = CXRClassifier(extractor, head).to(device)
    model.eval()
    return model


def predict_probability(model: CXRClassifier, image_tensor: torch.Tensor, device: torch.device) -> Tuple[int, float]:
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).detach().cpu())
    return pred, float(probs[1].detach().cpu())


class DenseNetXAI:
    def __init__(self, model: CXRClassifier, device: torch.device, ig_steps: int = 50) -> None:
        self.model = model
        self.device = device
        self.ig_steps = ig_steps
        self.target_layer = self.model.feature_extractor.encoder.features.denseblock4

    def _score_class(self, image_tensor: torch.Tensor, class_index: Optional[int]) -> Tuple[torch.Tensor, int, float]:
        logits = self.model(image_tensor.to(self.device))
        probs = torch.softmax(logits, dim=1)
        predicted = int(torch.argmax(probs, dim=1).detach().cpu()[0])
        target = predicted if class_index is None else int(class_index)
        return logits[:, target], predicted, float(probs[0, 1].detach().cpu())

    def grad_cam(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> ExplanationResult:
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_, __, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = self.target_layer.register_forward_hook(forward_hook)
        self.model.zero_grad(set_to_none=True)
        try:
            score, predicted, probability = self._score_class(image_tensor, class_index)
            score.sum().backward()
            feature_map = activations[-1].detach()
            gradient = gradients[-1].detach()
        finally:
            handle.remove()

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        heatmap = normalize_map(cam[0, 0].detach().cpu().numpy())
        return ExplanationResult(heatmap=heatmap, overlay=np.empty((0, 0, 3), dtype=np.uint8), probability=probability, predicted_label=predicted)

    def grad_cam_plus_plus(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> ExplanationResult:
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        try:
            from pytorch_grad_cam import GradCAMPlusPlus as TorchGradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            _, predicted, probability = self._score_class(image_tensor, class_index)
            target = predicted if class_index is None else int(class_index)
            with TorchGradCAMPlusPlus(model=self.model, target_layers=[self.target_layer]) as cam:
                grayscale_cam = cam(input_tensor=image_tensor.to(self.device), targets=[ClassifierOutputTarget(target)])
            heatmap = normalize_map(grayscale_cam[0])
            return ExplanationResult(heatmap=heatmap, overlay=np.empty((0, 0, 3), dtype=np.uint8), probability=probability, predicted_label=predicted)
        except Exception:
            return self._manual_grad_cam_plus_plus(image_tensor, class_index)

    def _manual_grad_cam_plus_plus(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> ExplanationResult:
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_, __, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = self.target_layer.register_forward_hook(forward_hook)
        self.model.zero_grad(set_to_none=True)
        try:
            score, predicted, probability = self._score_class(image_tensor, class_index)
            score.sum().backward()
            feature_map = activations[-1].detach()
            gradient = gradients[-1].detach()
        finally:
            handle.remove()

        grad2 = gradient.pow(2)
        grad3 = gradient.pow(3)
        denom = 2.0 * grad2 + (feature_map * grad3).sum(dim=(2, 3), keepdim=True)
        alphas = grad2 / torch.where(denom != 0.0, denom, torch.ones_like(denom) * 1e-8)
        weights = (alphas * torch.relu(gradient)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        heatmap = normalize_map(cam[0, 0].detach().cpu().numpy())
        return ExplanationResult(heatmap=heatmap, overlay=np.empty((0, 0, 3), dtype=np.uint8), probability=probability, predicted_label=predicted)

    def integrated_gradients(self, image_tensor: torch.Tensor, class_index: Optional[int] = None) -> ExplanationResult:
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ImportError("captum is required for Integrated Gradients. Install it with `pip install captum`.") from exc

        image_tensor = image_tensor.to(self.device)
        _, predicted, probability = self._score_class(image_tensor, class_index)
        target = predicted if class_index is None else int(class_index)
        baseline = torch.zeros_like(image_tensor)
        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(
            image_tensor,
            baselines=baseline,
            target=target,
            n_steps=self.ig_steps,
            method="gausslegendre",
        )
        heatmap = attributions.detach().abs().sum(dim=1)[0].cpu().numpy()
        heatmap = normalize_map(heatmap)
        return ExplanationResult(heatmap=heatmap, overlay=np.empty((0, 0, 3), dtype=np.uint8), probability=probability, predicted_label=predicted)

    def explain_all(self, image_tensor: torch.Tensor, display_image: np.ndarray, class_index: Optional[int] = None) -> Dict[str, ExplanationResult]:
        results = {
            "Grad-CAM": self.grad_cam(image_tensor, class_index),
            "Grad-CAM++": self.grad_cam_plus_plus(image_tensor, class_index),
            "Integrated Gradients": self.integrated_gradients(image_tensor, class_index),
        }
        for result in results.values():
            result.overlay = overlay_heatmap_on_grayscale(display_image, result.heatmap, alpha=0.4)
        return results


def save_side_by_side(
    original: np.ndarray,
    results: Dict[str, ExplanationResult],
    output_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(output_path))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    for axis, method in zip(axes[1:], METHODS):
        result = results[method]
        axis.imshow(result.overlay)
        axis.set_title(f"{method}\np={result.probability:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_grid(
    rows: Sequence[Tuple[str, np.ndarray, Dict[str, ExplanationResult]]],
    output_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(output_path))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, max(3, len(rows) * 2.1)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, (case_title, original, results) in enumerate(rows):
        axes[row_idx, 0].imshow(original, cmap="gray")
        axes[row_idx, 0].set_ylabel(case_title, fontsize=8)
        axes[row_idx, 0].set_title("Original" if row_idx == 0 else "")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for col_idx, method in enumerate(METHODS, start=1):
            axes[row_idx, col_idx].imshow(results[method].overlay)
            axes[row_idx, col_idx].set_title(method if row_idx == 0 else "")
            axes[row_idx, col_idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_cached_predictions(features_path: str, checkpoint_path: str, hidden_dim: int, dropout: float) -> Dict[str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    npz = np.load(features_path)
    features = npz["features"].astype(np.float32)
    labels = npz["labels"].astype(np.int64)
    patient_ids = npz["patient_ids"].astype(str)
    head = CXRMLPHead(int(checkpoint.get("feature_dim", features.shape[1])), hidden_dim=hidden_dim, dropout=dropout)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(features))
        probs = torch.softmax(logits, dim=1).numpy()
    return {
        "patient_ids": patient_ids,
        "y_true": labels,
        "y_prob": probs[:, 1],
        "y_pred": np.argmax(probs, axis=1).astype(np.int64),
    }


def select_case_ids(patient_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, List[str]]:
    buckets = {"pneumonia": [], "normal": [], "failure": []}
    for patient_id, true_label, pred_label in zip(patient_ids, y_true, y_pred):
        if true_label == 1 and pred_label == 1 and len(buckets["pneumonia"]) < 20:
            buckets["pneumonia"].append(str(patient_id))
        elif true_label == 0 and pred_label == 0 and len(buckets["normal"]) < 20:
            buckets["normal"].append(str(patient_id))
        elif true_label != pred_label and len(buckets["failure"]) < 10:
            buckets["failure"].append(str(patient_id))
    return buckets


def top_activation_iou(heatmap: np.ndarray, bboxes: Iterable[Iterable[float]]) -> float:
    bbox_mask = bbox_union_mask(bboxes, (224, 224))
    if bbox_mask.sum() == 0:
        return float("nan")
    threshold = np.percentile(heatmap, 80.0)
    activation_mask = heatmap >= threshold
    intersection = np.logical_and(activation_mask, bbox_mask > 0).sum()
    union = np.logical_or(activation_mask, bbox_mask > 0).sum()
    return float(intersection / union) if union else 0.0


def pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def evaluate_xai(
    explainer: DenseNetXAI,
    data_module: RSNAPneumoniaDataModule,
    patient_ids: Sequence[str],
    timing_ids: Sequence[str],
    consistency_id: Optional[str],
) -> Dict[str, Dict[str, float]]:
    localization: Dict[str, List[float]] = {method: [] for method in METHODS}
    timings: Dict[str, List[float]] = {method: [] for method in METHODS}
    consistency: Dict[str, List[float]] = {method: [] for method in METHODS}

    for patient_id in patient_ids:
        sample = data_module.load_sample(patient_id, normalize=False)
        if not sample["bboxes"]:
            continue
        display, tensor = read_xray_image(sample["image_path"])
        results = explainer.explain_all(tensor, display, class_index=1)
        for method, result in results.items():
            localization[method].append(top_activation_iou(result.heatmap, sample["bboxes"]))

    for patient_id in timing_ids[:50]:
        sample = data_module.load_sample(patient_id, normalize=False)
        display, tensor = read_xray_image(sample["image_path"])
        for method in METHODS:
            start = time.perf_counter()
            if method == "Grad-CAM":
                result = explainer.grad_cam(tensor)
            elif method == "Grad-CAM++":
                result = explainer.grad_cam_plus_plus(tensor)
            else:
                result = explainer.integrated_gradients(tensor)
            result.overlay = overlay_heatmap_on_grayscale(display, result.heatmap, alpha=0.4)
            timings[method].append((time.perf_counter() - start) * 1000.0)

    if consistency_id:
        sample = data_module.load_sample(consistency_id, normalize=False)
        display, tensor = read_xray_image(sample["image_path"])
        for method in METHODS:
            heatmaps = []
            for _ in range(3):
                if method == "Grad-CAM":
                    heatmaps.append(explainer.grad_cam(tensor).heatmap)
                elif method == "Grad-CAM++":
                    heatmaps.append(explainer.grad_cam_plus_plus(tensor).heatmap)
                else:
                    heatmaps.append(explainer.integrated_gradients(tensor).heatmap)
            consistency[method].extend([pixel_correlation(heatmaps[0], heatmaps[1]), pixel_correlation(heatmaps[0], heatmaps[2])])

    return {
        method: {
            "mean_iou": float(np.nanmean(localization[method])) if localization[method] else 0.0,
            "time_ms_mean": float(np.mean(timings[method])) if timings[method] else 0.0,
            "time_ms_std": float(np.std(timings[method])) if timings[method] else 0.0,
            "consistency": float(np.mean(consistency[method])) if consistency[method] else 0.0,
        }
        for method in METHODS
    }


def save_metric_figures(
    predictions: Dict[str, np.ndarray],
    threshold: float,
    xai_results: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    figures_dir = ensure_dir(os.path.join(output_dir, "figures"))
    y_true = predictions["y_true"]
    y_prob = predictions["y_prob"]
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision_score(y_true, y_prob)).plot()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curve.png"), dpi=180)
    plt.close()

    metrics_rows = [
        ("Accuracy", float((y_true == y_pred).mean())),
        ("Precision", float(cm[1, 1] / max(cm[:, 1].sum(), 1))),
        ("Recall", float(cm[1, 1] / max(cm[1, :].sum(), 1))),
        ("F1", float((2 * cm[1, 1]) / max((2 * cm[1, 1]) + cm[0, 1] + cm[1, 0], 1))),
        ("AUC-ROC", float(roc_auc_score(y_true, y_prob))),
        ("AUC-PR", float(average_precision_score(y_true, y_prob))),
    ]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    ax.table(cellText=[[name, f"{value:.4f}"] for name, value in metrics_rows], colLabels=["Metric", "Value"], loc="center")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "metrics_table.png"), dpi=180)
    plt.close(fig)

    methods = list(METHODS)
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(x - 0.18, [xai_results[m]["mean_iou"] for m in methods], width=0.36, label="Mean IoU")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, [xai_results[m]["time_ms_mean"] for m in methods], width=0.36, color="#d95f02", label="Time (ms)")
    ax1.set_xticks(x, methods)
    ax1.set_ylabel("Mean IoU")
    ax2.set_ylabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "xai_quantitative.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DenseNet/PyTorch XAI overlays and quantitative reports.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(get_project_root(), "rsna-pneumonia-detection-challenge"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(get_project_root(), "outputs", "cxr_foundation_densenet121-res224-all", "best_cxr_foundation.pt"))
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--weights", type=str, default="densenet121-res224-all")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(get_project_root(), "outputs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--xrv_cache_dir", type=str, default=os.path.join(get_project_root(), "outputs", "torchxrayvision"))
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(os.path.join(output_dir, "figures"))
    device = get_device(args.device)
    model = load_model(args.checkpoint, args.weights, args.hidden_dim, args.dropout, device, args.xrv_cache_dir)
    explainer = DenseNetXAI(model, device, ig_steps=args.ig_steps)

    if args.mode == "single":
        if not args.image_path:
            raise ValueError("--image_path is required for single mode.")
        display, tensor = read_xray_image(args.image_path)
        results = explainer.explain_all(tensor, display)
        pred = next(iter(results.values())).predicted_label
        prob = next(iter(results.values())).probability
        name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_side_by_side(display, results, os.path.join(figures_dir, f"{name}_xai.png"), f"{name} pred={pred} p={prob:.3f}")
        print(f"Saved XAI overlay to {os.path.join(figures_dir, f'{name}_xai.png')}")
        return

    data_module = RSNAPneumoniaDataModule(args.data_dir, allow_test_access=True)
    features_path = args.features or make_feature_cache_path(
        os.path.join(get_project_root(), "outputs", "feature_cache"),
        args.weights,
        "val",
        None,
    )
    predictions = load_cached_predictions(features_path, args.checkpoint, args.hidden_dim, args.dropout)
    threshold_pred = (predictions["y_prob"] >= args.threshold).astype(np.int64)
    selected = select_case_ids(predictions["patient_ids"], predictions["y_true"], threshold_pred)

    grids: Dict[str, List[Tuple[str, np.ndarray, Dict[str, ExplanationResult]]]] = {"pneumonia": [], "normal": [], "failure": []}
    for bucket, ids in selected.items():
        for patient_id in ids:
            sample = data_module.load_sample(patient_id, normalize=False)
            display, tensor = read_xray_image(sample["image_path"])
            results = explainer.explain_all(tensor, display)
            case_title = f"{patient_id[:8]} y={sample['label']} p={results['Grad-CAM'].probability:.2f}"
            grids[bucket].append((case_title, display, results))
            save_side_by_side(display, results, os.path.join(figures_dir, "cases", f"{patient_id}_xai.png"), case_title)

    save_grid(grids["pneumonia"], os.path.join(figures_dir, "xai_comparison_pneumonia.png"), "Correct Pneumonia Cases")
    save_grid(grids["normal"], os.path.join(figures_dir, "xai_comparison_normal.png"), "Correct Normal Cases")
    save_grid(grids["failure"], os.path.join(figures_dir, "xai_failure_cases.png"), "Misclassified Cases")

    annotated_ids = data_module.get_random_patient_ids("val", n=50, label=1, annotated_only=True)
    timing_ids = predictions["patient_ids"][:50].astype(str).tolist()
    consistency_id = timing_ids[0] if timing_ids else None
    xai_results = evaluate_xai(explainer, data_module, annotated_ids, timing_ids, consistency_id)
    with open(os.path.join(output_dir, "xai_quantitative.json"), "w", encoding="utf-8") as handle:
        json.dump(xai_results, handle, indent=2)
    save_metric_figures(predictions, args.threshold, xai_results, output_dir)

    print("Method | Mean IoU | Time (ms) | Consistency")
    for method in METHODS:
        row = xai_results[method]
        print(f"{method} | {row['mean_iou']:.4f} | {row['time_ms_mean']:.1f} +/- {row['time_ms_std']:.1f} | {row['consistency']:.4f}")


if __name__ == "__main__":
    main()
