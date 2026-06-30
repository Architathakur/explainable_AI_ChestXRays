import random
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.utils import imagenet_denormalize, normalize_map, overlay_heatmap_on_grayscale
from xai_pneumonia.xai.common import build_feature_extractor, get_best_gradcam_layer


class GradCAMPlusPlus:
    def __init__(self, model: tf.keras.Model, target_layer_name: Optional[str] = None, class_index: int = 1) -> None:
        self.model = model
        self.target_layer_name = target_layer_name or get_best_gradcam_layer()
        self.class_index = class_index
        self.extractor = build_feature_extractor(model, self.target_layer_name)

    def generate(
        self,
        image: np.ndarray,
        original_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = self.extractor(input_tensor, training=False)
            class_score = predictions[:, self.class_index]
        first_grads = tape.gradient(class_score, conv_outputs)
        del tape

        conv_outputs = conv_outputs[0]
        first_grads = first_grads[0]
        grad2 = tf.square(first_grads)
        grad3 = tf.pow(first_grads, 3)
        eps = tf.constant(1e-8, dtype=tf.float32)

        alpha_denom = 2.0 * grad2 + tf.reduce_sum(conv_outputs * grad3, axis=(0, 1), keepdims=True)
        alpha_denom = tf.where(tf.not_equal(alpha_denom, 0.0), alpha_denom, eps)
        alphas = grad2 / alpha_denom
        positive_gradients = tf.nn.relu(first_grads)
        weights = tf.reduce_sum(alphas * positive_gradients, axis=(0, 1))

        heatmap = tf.reduce_sum(conv_outputs * weights[tf.newaxis, tf.newaxis, :], axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        heatmap = normalize_map(heatmap)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_map(heatmap)

        display_image = original_image if original_image is not None else imagenet_denormalize(image)
        overlay = overlay_heatmap_on_grayscale(display_image, heatmap, alpha=0.4)
        return heatmap.astype(np.float32), overlay.astype(np.uint8)
