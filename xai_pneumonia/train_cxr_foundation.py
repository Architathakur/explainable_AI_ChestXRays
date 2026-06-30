import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from xai_pneumonia.data.data_loader import RSNAPneumoniaDataModule
from xai_pneumonia.model.cxr_torch_model import CXRClassifier, CXRFeatureExtractor, CXRMLPHead
from xai_pneumonia.utils import ensure_dir, get_project_root


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    f1_pneumonia: float
    auc_roc: float
    auc_pr: float


class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RSNATorchDataset(Dataset):
    def __init__(
        self,
        data_module: RSNAPneumoniaDataModule,
        split_name: str,
        sample_limit: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        split_df = data_module.get_split_dataframe(split_name)
        if sample_limit is not None and sample_limit > 0:
            split_df = split_df.iloc[:sample_limit].copy()
        self.rows = split_df[["patient_id", "image_path", "label"]].to_dict("records")
        self.augment = augment

    @staticmethod
    def _read_dicom(image_path: str, augment: bool = False) -> np.ndarray:
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = np.max(image) - image
        image -= np.min(image)
        max_value = float(np.max(image))
        if max_value > 0:
            image /= max_value
        image *= 255.0
        image = cv2.resize(image.astype(np.float32), (224, 224), interpolation=cv2.INTER_AREA)
        if augment:
            center = (112.0, 112.0)
            angle = float(np.random.uniform(-7.0, 7.0))
            scale = float(np.random.uniform(0.94, 1.06))
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            translate = np.random.uniform(-8.0, 8.0, size=2)
            matrix[0, 2] += float(translate[0])
            matrix[1, 2] += float(translate[1])
            image = cv2.warpAffine(
                image,
                matrix,
                (224, 224),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            image = np.clip(image * float(np.random.uniform(0.9, 1.1)), 0.0, 255.0)
        return image

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image = self._read_dicom(row["image_path"], augment=self.augment)
        image = (2.0 * (image.astype(np.float32) / 255.0) - 1.0) * 1024.0
        image = torch.from_numpy(image[None, :, :].astype(np.float32))
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label, str(row["patient_id"])


def class_weights_from_data_module(data_module: RSNAPneumoniaDataModule, device: torch.device) -> torch.Tensor:
    class_weights = data_module.compute_class_weights()
    return torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32, device=device)


def make_balanced_sampler(labels: np.ndarray, epoch_size: Optional[int] = None) -> WeightedRandomSampler:
    labels = labels.astype(np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    sample_weights = 1.0 / np.maximum(class_counts[labels], 1.0)
    num_samples = int(epoch_size) if epoch_size and epoch_size > 0 else int(len(labels))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def make_balanced_sampler_from_dataset(dataset: RSNATorchDataset, epoch_size: Optional[int] = None) -> WeightedRandomSampler:
    labels = np.array([int(row["label"]) for row in dataset.rows], dtype=np.int64)
    return make_balanced_sampler(labels, epoch_size)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, loss: float) -> EpochMetrics:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    _, _, f1, _ = precision_recall_fscore_support(y_true, predictions, labels=[0, 1], zero_division=0)
    return EpochMetrics(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, predictions)),
        f1_pneumonia=float(f1[1]),
        auc_roc=float(roc_auc_score(y_true, probabilities[:, 1])) if len(np.unique(y_true)) == 2 else 0.0,
        auc_pr=float(average_precision_score(y_true, probabilities[:, 1])) if len(np.unique(y_true)) == 2 else 0.0,
    )


def run_head_epoch(
    head: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    is_train = optimizer is not None
    head.train(is_train)
    losses: List[float] = []
    labels_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels_all.append(labels.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())

    return metrics_from_logits(
        np.concatenate(labels_all),
        np.concatenate(logits_all),
        float(np.mean(losses)),
    )


def run_image_epoch(
    model: CXRClassifier,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []
    labels_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels_all.append(labels.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())

    return metrics_from_logits(
        np.concatenate(labels_all),
        np.concatenate(logits_all),
        float(np.mean(losses)),
    )


def make_feature_cache_path(cache_dir: str, weights: str, split: str, sample_limit: Optional[int]) -> str:
    suffix = f"n{sample_limit}" if sample_limit else "full"
    safe_weights = weights.replace("/", "_")
    return os.path.join(cache_dir, f"{safe_weights}_{split}_{suffix}.npz")


@torch.no_grad()
def extract_features(
    extractor: CXRFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    extractor.eval()
    features_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    ids_all: List[str] = []
    for images, labels, patient_ids in tqdm(loader, desc=f"extract {split}", total=len(loader)):
        images = images.to(device)
        features = extractor(images)
        features_all.append(features.detach().cpu().numpy())
        labels_all.append(labels.numpy())
        ids_all.extend(patient_ids)
    return np.concatenate(features_all), np.concatenate(labels_all), np.array(ids_all, dtype=str)


def load_or_extract_features(
    extractor: CXRFeatureExtractor,
    dataset: RSNATorchDataset,
    split: str,
    weights: str,
    cache_dir: Optional[str],
    batch_size: int,
    device: torch.device,
    sample_limit: Optional[int],
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_path = make_feature_cache_path(cache_dir, weights, split, sample_limit) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["features"].astype(np.float32), cached["labels"].astype(np.int64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    features, labels, patient_ids = extract_features(extractor, loader, device, split)
    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        np.savez_compressed(
            cache_path,
            features=features.astype(np.float16),
            labels=labels.astype(np.int64),
            patient_ids=patient_ids,
        )
        print(f"Saved feature cache: {cache_path} ({features.shape[0]} x {features.shape[1]})")
    return features.astype(np.float32), labels.astype(np.int64)


def metric_dict(metrics: EpochMetrics) -> Dict[str, float]:
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "f1_pneumonia": metrics.f1_pneumonia,
        "auc_roc": metrics.auc_roc,
        "auc_pr": metrics.auc_pr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CXR-pretrained frozen encoder + MLP head on RSNA.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(get_project_root(), "rsna-pneumonia-detection-challenge"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default="densenet121-res224-all")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(get_project_root(), "outputs", "feature_cache"))
    parser.add_argument("--xrv_cache_dir", type=str, default=os.path.join(get_project_root(), "outputs", "torchxrayvision"))
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--fine_tune_train_limit", type=int, default=None)
    parser.add_argument("--fine_tune_val_limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--balanced_epoch_size", type=int, default=None)
    parser.add_argument("--initial_checkpoint", type=str, default=None)
    parser.add_argument("--fine_tune_augment", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.output_dir is None:
        safe_weights = args.weights.replace("/", "_")
        args.output_dir = os.path.join(get_project_root(), "outputs", f"cxr_foundation_{safe_weights}")
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    print(f"Using device: {device}")

    data_module = RSNAPneumoniaDataModule(data_dir=args.data_dir, batch_size=args.batch_size, allow_test_access=False)
    print("Split summary:")
    print(data_module.get_split_summary().to_string(index=False))

    train_dataset = RSNATorchDataset(data_module, "train", sample_limit=args.train_limit)
    val_dataset = RSNATorchDataset(data_module, "val", sample_limit=args.val_limit)

    weights_tensor = class_weights_from_data_module(data_module, device)
    if args.loss == "focal":
        criterion = FocalLoss(weight=weights_tensor, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    history: Dict[str, List[float]] = {}

    cache_dir = None if args.no_cache else args.cache_dir
    train_cache_path = make_feature_cache_path(cache_dir, args.weights, "train", args.train_limit) if cache_dir else None
    val_cache_path = make_feature_cache_path(cache_dir, args.weights, "val", args.val_limit) if cache_dir else None
    can_train_from_cache_only = (
        args.fine_tune_epochs == 0
        and train_cache_path
        and val_cache_path
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
    )
    skip_feature_phase = args.epochs == 0 and args.fine_tune_epochs > 0

    extractor: Optional[CXRFeatureExtractor] = None
    classifier: Optional[CXRClassifier] = None
    if can_train_from_cache_only:
        print("Using cached features only; skipping CXR encoder initialization.")
        cached_train = np.load(train_cache_path)
        cached_val = np.load(val_cache_path)
        train_features = cached_train["features"].astype(np.float32)
        train_labels = cached_train["labels"].astype(np.int64)
        val_features = cached_val["features"].astype(np.float32)
        val_labels = cached_val["labels"].astype(np.int64)
        feature_dim = int(train_features.shape[1])
        head = CXRMLPHead(feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    elif skip_feature_phase:
        print("Skipping cached-feature training phase; initializing encoder for image fine-tuning only.")
        extractor = CXRFeatureExtractor(weights=args.weights, cache_dir=args.xrv_cache_dir).to(device)
        extractor.freeze()
        feature_dim = extractor.feature_dim
        head = CXRMLPHead(feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        classifier = CXRClassifier(extractor, head).to(device)
    else:
        extractor = CXRFeatureExtractor(weights=args.weights, cache_dir=args.xrv_cache_dir).to(device)
        extractor.freeze()
        feature_dim = extractor.feature_dim
        head = CXRMLPHead(feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        classifier = CXRClassifier(extractor, head).to(device)
        train_features, train_labels = load_or_extract_features(
            extractor,
            train_dataset,
            "train",
            args.weights,
            cache_dir,
            args.feature_batch_size,
            device,
            args.train_limit,
            args.num_workers,
        )
        val_features, val_labels = load_or_extract_features(
            extractor,
            val_dataset,
            "val",
            args.weights,
            cache_dir,
            args.feature_batch_size,
            device,
            args.val_limit,
            args.num_workers,
        )

    if args.initial_checkpoint:
        initial_checkpoint = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        head.load_state_dict(initial_checkpoint["head_state_dict"])
        print(f"Loaded initial head checkpoint: {args.initial_checkpoint}")

    if args.epochs > 0:
        train_feature_dataset = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
        train_sampler = make_balanced_sampler(train_labels, args.balanced_epoch_size) if args.balanced_sampler else None
        train_feature_loader = DataLoader(
            train_feature_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        val_feature_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels)),
            batch_size=args.batch_size,
            shuffle=False,
        )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = -1.0
    best_path = os.path.join(output_dir, "best_cxr_foundation.pt")
    if args.initial_checkpoint:
        best_score = float(initial_checkpoint.get("best_val_f1", -1.0))
        torch.save(
            {
                "weights": args.weights,
                "feature_dim": feature_dim,
                "head_state_dict": head.state_dict(),
                "best_val_f1": best_score,
                "initial_checkpoint": args.initial_checkpoint,
            },
            best_path,
        )
        if args.fine_tune_epochs > 0 and args.epochs == 0:
            best_score = -1.0

    print("\nPhase 1: frozen CXR encoder, train MLP head")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_head_epoch(head, train_feature_loader, optimizer, criterion, device)
        val_metrics = run_head_epoch(head, val_feature_loader, None, criterion, device)
        for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for key, value in metric_dict(metrics).items():
                history.setdefault(f"{prefix}_{key}", []).append(value)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_f1={val_metrics.f1_pneumonia:.4f} val_auc_roc={val_metrics.auc_roc:.4f} "
            f"val_auc_pr={val_metrics.auc_pr:.4f}"
        )
        if val_metrics.f1_pneumonia > best_score:
            best_score = val_metrics.f1_pneumonia
            epochs_without_improvement = 0
            torch.save(
                {
                    "weights": args.weights,
                    "feature_dim": feature_dim,
                    "head_state_dict": head.state_dict(),
                    "best_val_f1": best_score,
                    "history": history,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs without val F1 improvement.")
                break

    if args.fine_tune_epochs > 0:
        print("\nPhase 2: short fine-tune of DenseNet denseblock4 + head")
        if classifier is None or extractor is None:
            extractor = CXRFeatureExtractor(weights=args.weights, cache_dir=args.xrv_cache_dir).to(device)
            extractor.freeze()
            classifier = CXRClassifier(extractor, head).to(device)
        classifier.unfreeze_last_block()
        fine_tune_train_dataset = (
            RSNATorchDataset(
                data_module,
                "train",
                sample_limit=args.fine_tune_train_limit,
                augment=args.fine_tune_augment,
            )
            if args.fine_tune_train_limit
            else RSNATorchDataset(data_module, "train", augment=args.fine_tune_augment)
        )
        fine_tune_val_dataset = (
            RSNATorchDataset(data_module, "val", sample_limit=args.fine_tune_val_limit)
            if args.fine_tune_val_limit
            else val_dataset
        )
        fine_tune_sampler = (
            make_balanced_sampler_from_dataset(fine_tune_train_dataset, args.balanced_epoch_size)
            if args.balanced_sampler
            else None
        )
        fine_tune_loader = DataLoader(
            fine_tune_train_dataset,
            batch_size=args.batch_size,
            shuffle=fine_tune_sampler is None,
            sampler=fine_tune_sampler,
            num_workers=0,
        )
        val_image_loader = DataLoader(fine_tune_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(classifier.trainable_parameters(), lr=args.fine_tune_lr, weight_decay=1e-5)
        for epoch in range(1, args.fine_tune_epochs + 1):
            train_metrics = run_image_epoch(classifier, fine_tune_loader, optimizer, criterion, device)
            val_metrics = run_image_epoch(classifier, val_image_loader, None, criterion, device)
            print(
                f"fine_tune {epoch:02d}/{args.fine_tune_epochs} "
                f"loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
                f"val_f1={val_metrics.f1_pneumonia:.4f} val_auc_roc={val_metrics.auc_roc:.4f} "
                f"val_auc_pr={val_metrics.auc_pr:.4f}"
            )
            if val_metrics.f1_pneumonia > best_score:
                best_score = val_metrics.f1_pneumonia
                torch.save(
                    {
                        "weights": args.weights,
                        "feature_dim": feature_dim,
                        "head_state_dict": head.state_dict(),
                        "encoder_state_dict": extractor.encoder.state_dict(),
                        "best_val_f1": best_score,
                        "fine_tuned": True,
                    },
                    best_path,
                )

    best_checkpoint = torch.load(best_path, map_location=device)
    head.load_state_dict(best_checkpoint["head_state_dict"])
    if "encoder_state_dict" in best_checkpoint:
        if classifier is None or extractor is None:
            extractor = CXRFeatureExtractor(weights=args.weights, cache_dir=args.xrv_cache_dir).to(device)
            classifier = CXRClassifier(extractor, head).to(device)
        extractor.encoder.load_state_dict(best_checkpoint["encoder_state_dict"])

    if "encoder_state_dict" in best_checkpoint:
        val_image_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        classifier.eval()
        logits_all: List[np.ndarray] = []
        labels_all: List[np.ndarray] = []
        with torch.no_grad():
            for images, labels, _ in val_image_loader:
                logits_all.append(classifier(images.to(device)).detach().cpu().numpy())
                labels_all.append(labels.numpy())
        logits = np.concatenate(logits_all)
        val_labels_for_report = np.concatenate(labels_all)
    else:
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(val_features).to(device)).detach().cpu().numpy()
        val_labels_for_report = val_labels

    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    _, _, f1, _ = precision_recall_fscore_support(val_labels_for_report, predictions, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(val_labels_for_report, predictions, labels=[0, 1])
    final_metrics = {
        "accuracy": float(accuracy_score(val_labels_for_report, predictions)),
        "precision": float(precision_score(val_labels_for_report, predictions, zero_division=0)),
        "recall": float(recall_score(val_labels_for_report, predictions, zero_division=0)),
        "f1_pneumonia": float(f1[1]),
        "auc_roc": float(roc_auc_score(val_labels_for_report, probabilities[:, 1])),
        "auc_pr": float(average_precision_score(val_labels_for_report, probabilities[:, 1])),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": classification_report(val_labels_for_report, predictions, digits=4, zero_division=0),
    }

    history_path = os.path.join(output_dir, "history.json")
    metrics_path = os.path.join(output_dir, "val_metrics.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print("\nValidation metrics summary")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Pneumonia: {final_metrics['f1_pneumonia']:.4f}")
    print(f"AUC-ROC: {final_metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {final_metrics['auc_pr']:.4f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: {final_metrics['confusion_matrix']}")
    print("\nClassification report:")
    print(final_metrics["classification_report"])
    print(f"Saved model checkpoint to: {best_path}")
    print(f"Saved history to: {history_path}")
    print(f"Saved validation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
