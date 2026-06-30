import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

random.seed(42)
np.random.seed(42)

from xai_pneumonia.utils import (
    ensure_dir,
    get_project_root,
    imagenet_normalize,
    load_json,
    one_hot,
    resize_bboxes,
    save_json,
    set_global_determinism,
)


@dataclass
class SampleRecord:
    patient_id: str
    image_path: str
    label: int
    bboxes: List[List[float]]


class RSNAPneumoniaDataModule:
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        seed: int = 42,
        allow_test_access: bool = False,
    ) -> None:
        set_global_determinism(seed)
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.allow_test_access = allow_test_access
        self.project_root = get_project_root()
        self.package_data_dir = ensure_dir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
        self.splits_path = os.path.join(self.package_data_dir, "splits.json")
        self.mean_training_image_path = os.path.join(self.package_data_dir, "mean_training_image.npy")
        self.labels_csv = os.path.join(self.data_dir, "stage_2_train_labels.csv")
        self.images_dir = os.path.join(self.data_dir, "stage_2_train_images")
        self.metadata = self._build_metadata()
        self.splits = self._load_or_create_splits()

    def _build_metadata(self) -> pd.DataFrame:
        if not os.path.exists(self.labels_csv):
            raise FileNotFoundError(f"Could not find RSNA labels CSV at {self.labels_csv}")

        labels_df = pd.read_csv(self.labels_csv)
        labels_df["Target"] = labels_df["Target"].fillna(0).astype(int)
        grouped_rows: List[Dict[str, object]] = []

        for patient_id, group in labels_df.groupby("patientId"):
            positive_rows = group[group["Target"] == 1]
            bboxes: List[List[float]] = []
            for _, row in positive_rows.iterrows():
                if pd.notna(row["x"]) and pd.notna(row["y"]) and pd.notna(row["width"]) and pd.notna(row["height"]):
                    bboxes.append(
                        [
                            float(row["x"]),
                            float(row["y"]),
                            float(row["width"]),
                            float(row["height"]),
                        ]
                    )
            grouped_rows.append(
                {
                    "patient_id": str(patient_id),
                    "image_path": os.path.join(self.images_dir, f"{patient_id}.dcm"),
                    "label": int(group["Target"].max()),
                    "bboxes": bboxes,
                }
            )

        metadata = pd.DataFrame(grouped_rows).sort_values("patient_id").reset_index(drop=True)
        return metadata

    def _split_ids_for_label(self, patient_ids: np.ndarray) -> Tuple[List[str], List[str], List[str]]:
        label_df = pd.DataFrame({"patient_id": patient_ids})
        splitter_primary = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=self.seed)
        train_idx, temp_idx = next(
            splitter_primary.split(label_df, groups=label_df["patient_id"])
        )
        train_ids = label_df.iloc[train_idx]["patient_id"].tolist()
        temp_df = label_df.iloc[temp_idx].reset_index(drop=True)

        splitter_secondary = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=self.seed)
        val_idx, test_idx = next(
            splitter_secondary.split(temp_df, groups=temp_df["patient_id"])
        )
        val_ids = temp_df.iloc[val_idx]["patient_id"].tolist()
        test_ids = temp_df.iloc[test_idx]["patient_id"].tolist()
        return train_ids, val_ids, test_ids

    def _load_or_create_splits(self) -> Dict[str, List[str]]:
        if os.path.exists(self.splits_path):
            return load_json(self.splits_path)

        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        rng = np.random.default_rng(self.seed)
        for label in [0, 1]:
            patient_ids = self.metadata.loc[self.metadata["label"] == label, "patient_id"].to_numpy()
            train_ids, val_ids, test_ids = self._split_ids_for_label(patient_ids)
            splits["train"].extend(train_ids)
            splits["val"].extend(val_ids)
            splits["test"].extend(test_ids)

        for split_name in splits:
            shuffled = np.array(splits[split_name], dtype=object)
            rng.shuffle(shuffled)
            splits[split_name] = shuffled.tolist()

        save_json(splits, self.splits_path)
        return splits

    def get_split_dataframe(self, split_name: str) -> pd.DataFrame:
        if split_name == "test" and not self.allow_test_access:
            raise PermissionError("Test split is locked. Enable allow_test_access=True only in explain/evaluate.")
        patient_ids = set(self.splits[split_name])
        split_df = self.metadata[self.metadata["patient_id"].isin(patient_ids)].copy()
        split_df["split"] = split_name
        split_df = split_df.sort_values("patient_id").reset_index(drop=True)
        return split_df

    def get_split_summary(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for split_name in ["train", "val", "test"]:
            if split_name == "test" and not self.allow_test_access:
                patient_ids = self.splits["test"]
                split_df = self.metadata[self.metadata["patient_id"].isin(patient_ids)].copy()
            else:
                split_df = self.get_split_dataframe(split_name)
            total = len(split_df)
            positive = int(split_df["label"].sum())
            negative = total - positive
            rows.append(
                {
                    "split": split_name,
                    "count": total,
                    "positive": positive,
                    "negative": negative,
                    "positive_ratio": positive / total if total else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def compute_class_weights(self) -> Dict[int, float]:
        train_df = self.get_split_dataframe("train")
        classes = np.array([0, 1], dtype=np.int32)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"].to_numpy())
        return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}

    @staticmethod
    def _safe_read_dicom(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        try:
            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array
            original_shape = (int(pixel_array.shape[0]), int(pixel_array.shape[1]))
            pixel_array -= np.min(pixel_array)
            max_val = float(np.max(pixel_array))
            if max_val > 0:
                pixel_array /= max_val
            pixel_array *= 255.0
            pixel_array = np.clip(pixel_array, 0.0, 255.0).astype(np.uint8)
        except Exception:
            original_shape = (1024, 1024)
            pixel_array = np.zeros(original_shape, dtype=np.uint8)
        return pixel_array, original_shape

    def _augment_image(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation = np.random.uniform(-10.0, 10.0)
        zoom = np.random.uniform(0.9, 1.1)
        shear_deg = np.random.uniform(-5.0, 5.0)
        shear = np.tan(np.deg2rad(shear_deg))
        matrix = cv2.getRotationMatrix2D(center, rotation, zoom)
        matrix[0, 1] += shear
        augmented = cv2.warpAffine(
            image_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        brightness = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.3:
            gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            augmented = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return augmented

    def load_sample(
        self,
        patient_id: str,
        augment: bool = False,
        normalize: bool = True,
    ) -> Dict[str, object]:
        row = self.metadata[self.metadata["patient_id"] == patient_id]
        if row.empty:
            raise KeyError(f"Unknown patient_id: {patient_id}")
        record = row.iloc[0]
        dicom_gray, original_shape = self._safe_read_dicom(record["image_path"])
        resized_gray = cv2.resize(dicom_gray, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
        if augment:
            image_rgb = self._augment_image(image_rgb)
        image_float = image_rgb.astype(np.float32) / 255.0
        standardized = imagenet_normalize(image_float) if normalize else image_float.astype(np.float32)
        resized_boxes = resize_bboxes(record["bboxes"], original_shape, self.image_size)
        return {
            "patient_id": record["patient_id"],
            "image": standardized.astype(np.float32),
            "image_rgb": image_float.astype(np.float32),
            "label": int(record["label"]),
            "label_one_hot": one_hot(int(record["label"])),
            "bboxes": resized_boxes,
            "original_shape": original_shape,
            "image_path": record["image_path"],
        }

    def load_image_from_path(
        self,
        image_path: str,
        normalize: bool = True,
    ) -> Dict[str, object]:
        dicom_gray, original_shape = self._safe_read_dicom(image_path)
        resized_gray = cv2.resize(dicom_gray, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        standardized = imagenet_normalize(image_float) if normalize else image_float.astype(np.float32)
        patient_id = os.path.splitext(os.path.basename(image_path))[0]
        row = self.metadata[self.metadata["patient_id"] == patient_id]
        label = int(row.iloc[0]["label"]) if not row.empty else -1
        bboxes = resize_bboxes(row.iloc[0]["bboxes"], original_shape, self.image_size) if not row.empty else []
        return {
            "patient_id": patient_id,
            "image": standardized.astype(np.float32),
            "image_rgb": image_float.astype(np.float32),
            "label": label,
            "label_one_hot": one_hot(label) if label in [0, 1] else None,
            "bboxes": bboxes,
            "original_shape": original_shape,
            "image_path": image_path,
        }

    def compute_mean_training_image(self, sample_size: int = 1000, force: bool = False) -> np.ndarray:
        if os.path.exists(self.mean_training_image_path) and not force:
            return np.load(self.mean_training_image_path)

        train_df = self.get_split_dataframe("train")
        sampled_ids = train_df["patient_id"].tolist()
        rng = np.random.default_rng(self.seed)
        if len(sampled_ids) > sample_size:
            sampled_ids = rng.choice(sampled_ids, size=sample_size, replace=False).tolist()

        images: List[np.ndarray] = []
        for patient_id in sampled_ids:
            sample = self.load_sample(patient_id, augment=False, normalize=True)
            images.append(sample["image"])
        mean_image = np.mean(np.stack(images, axis=0), axis=0).astype(np.float32)
        np.save(self.mean_training_image_path, mean_image)
        return mean_image

    def get_mean_training_image(self) -> np.ndarray:
        return self.compute_mean_training_image(force=False)

    def _numpy_loader(self, patient_id_bytes: bytes, label: np.ndarray, training: bool) -> Tuple[np.ndarray, np.ndarray]:
        patient_id = patient_id_bytes.decode("utf-8")
        sample = self.load_sample(patient_id, augment=training, normalize=True)
        return sample["image"].astype(np.float32), sample["label_one_hot"].astype(np.float32)

    def make_tf_dataset(
        self,
        split_name: str,
        training: bool = False,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        sample_limit: Optional[int] = None,
    ):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required only for the legacy ResNet/TensorFlow data path. "
                "Use train_cxr_foundation.py for the DenseNet/PyTorch pipeline."
            ) from exc

        split_df = self.get_split_dataframe(split_name)
        if sample_limit is not None and sample_limit > 0:
            split_df = split_df.iloc[:sample_limit].copy()
        patient_ids = split_df["patient_id"].to_numpy(dtype=str)
        labels = split_df["label"].to_numpy(dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((patient_ids, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(split_df), seed=self.seed, reshuffle_each_iteration=True)

        def _map_fn(patient_id: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            image, target = tf.numpy_function(
                lambda pid, y: self._numpy_loader(pid, y, training),
                [patient_id, label],
                [tf.float32, tf.float32],
            )
            image.set_shape((self.image_size[0], self.image_size[1], 3))
            target.set_shape((2,))
            return image, target

        dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size or self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_records(self, split_name: str) -> List[SampleRecord]:
        split_df = self.get_split_dataframe(split_name)
        records: List[SampleRecord] = []
        for _, row in split_df.iterrows():
            records.append(
                SampleRecord(
                    patient_id=row["patient_id"],
                    image_path=row["image_path"],
                    label=int(row["label"]),
                    bboxes=row["bboxes"],
                )
            )
        return records

    def get_random_patient_ids(
        self,
        split_name: str,
        n: int,
        label: Optional[int] = None,
        annotated_only: bool = False,
    ) -> List[str]:
        split_df = self.get_split_dataframe(split_name)
        if label is not None:
            split_df = split_df[split_df["label"] == label]
        if annotated_only:
            split_df = split_df[split_df["bboxes"].map(len) > 0]
        ids = split_df["patient_id"].tolist()
        rng = np.random.default_rng(self.seed)
        if len(ids) <= n:
            return ids
        return rng.choice(ids, size=n, replace=False).tolist()
