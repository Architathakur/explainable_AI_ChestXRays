import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_global_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def imagenet_denormalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return np.clip((image * IMAGENET_STD) + IMAGENET_MEAN, 0.0, 1.0)


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min)


def to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        grayscale = np.mean(image, axis=-1)
    else:
        grayscale = image
    grayscale = np.clip(grayscale, 0.0, 1.0)
    return (grayscale * 255.0).astype(np.uint8)


def overlay_heatmap_on_grayscale(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    image_rgb = image
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    grayscale = to_uint8_grayscale(image_rgb)
    grayscale_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    heatmap_uint8 = (normalize_map(heatmap) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(grayscale_rgb, 1.0 - alpha, colored, alpha, 0)
    return overlay


def bbox_union_mask(bboxes: Iterable[Iterable[float]], image_size: Tuple[int, int]) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in bboxes:
        if bbox is None:
            continue
        x, y, w, h = [int(round(float(value))) for value in bbox]
        x1 = np.clip(x, 0, width - 1)
        y1 = np.clip(y, 0, height - 1)
        x2 = np.clip(x + w, 0, width)
        y2 = np.clip(y + h, 0, height)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def resize_bboxes(
    bboxes: Iterable[Iterable[float]],
    original_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> List[List[float]]:
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)
    resized: List[List[float]] = []
    for bbox in bboxes:
        if bbox is None:
            continue
        x, y, w, h = bbox
        resized.append(
            [
                float(x) * scale_x,
                float(y) * scale_y,
                float(w) * scale_x,
                float(h) * scale_y,
            ]
        )
    return resized


def one_hot(label: int, num_classes: int = 2) -> np.ndarray:
    encoded = np.zeros((num_classes,), dtype=np.float32)
    encoded[int(label)] = 1.0
    return encoded
