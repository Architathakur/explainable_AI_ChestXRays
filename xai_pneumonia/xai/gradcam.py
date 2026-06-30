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


class GradCAM:
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

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.extractor(input_tensor, training=False)
            class_score = predictions[:, self.class_index]

        gradients = tape.gradient(class_score, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
        conv_outputs = conv_outputs[0]
        weights = pooled_gradients[0]
        heatmap = tf.reduce_sum(conv_outputs * weights[tf.newaxis, tf.newaxis, :], axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        heatmap = normalize_map(heatmap)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_map(heatmap)

        display_image = original_image if original_image is not None else imagenet_denormalize(image)
        overlay = overlay_heatmap_on_grayscale(display_image, heatmap, alpha=0.4)
        return heatmap.astype(np.float32), overlay.astype(np.uint8)
