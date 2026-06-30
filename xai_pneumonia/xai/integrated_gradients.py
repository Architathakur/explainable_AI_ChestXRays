import os
import random
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_project_root,
    imagenet_denormalize,
    imagenet_normalize,
    normalize_map,
    overlay_heatmap_on_grayscale,
)


class IntegratedGradients:
    def __init__(
        self,
        model: tf.keras.Model,
        steps: int = 50,
        class_index: int = 1,
        mean_image_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.steps = steps
        self.class_index = class_index
        self.mean_image_path = mean_image_path or os.path.join(get_project_root(), "data", "mean_training_image.npy")

    def _get_baseline(self, image_shape: Tuple[int, int, int], baseline_type: str) -> Tuple[np.ndarray, str]:
        if baseline_type == "black":
            baseline = np.zeros(image_shape, dtype=np.float32)
            baseline = imagenet_normalize(baseline)
            return baseline, "black"
        if baseline_type == "noise":
            noise = np.random.normal(0.5, 0.1, image_shape).astype(np.float32)
            noise = np.clip(noise, 0.0, 1.0)
            baseline = imagenet_normalize(noise)
            return baseline, "gaussian_noise"
        if baseline_type == "mean":
            if not os.path.exists(self.mean_image_path):
                raise FileNotFoundError(f"Mean training image not found at {self.mean_image_path}")
            baseline = np.load(self.mean_image_path).astype(np.float32)
            return baseline, "mean_training_image"
        raise ValueError(f"Unsupported baseline_type: {baseline_type}")

    @tf.function
    def _interpolate_images(self, baseline: tf.Tensor, image: tf.Tensor, alphas: tf.Tensor) -> tf.Tensor:
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        delta = image - baseline
        return baseline + alphas_x * delta

    @tf.function
    def _compute_gradients(self, interpolated_images: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            predictions = self.model(interpolated_images, training=False)
            class_scores = predictions[:, self.class_index]
        gradients = tape.gradient(class_scores, interpolated_images)
        return gradients

    def generate(
        self,
        image: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        baseline_type: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        baseline_np, baseline_label = self._get_baseline(image.shape, baseline_type)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        baseline_tensor = tf.convert_to_tensor(baseline_np, dtype=tf.float32)
        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        interpolated = self._interpolate_images(baseline_tensor, image_tensor, alphas)
        gradients = self._compute_gradients(interpolated)
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)
        integrated = (image_tensor - baseline_tensor) * avg_grads
        attribution = tf.reduce_mean(tf.math.abs(integrated), axis=-1).numpy()
        attribution = normalize_map(attribution)

        display_image = original_image if original_image is not None else imagenet_denormalize(image)
        overlay = overlay_heatmap_on_grayscale(display_image, attribution, alpha=0.4)
        return attribution.astype(np.float32), overlay.astype(np.uint8), baseline_label
