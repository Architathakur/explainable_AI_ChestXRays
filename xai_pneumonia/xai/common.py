import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.utils import get_project_root


def get_best_gradcam_layer(default_layer: str = "conv5_block3_out") -> str:
    project_root = get_project_root()
    layer_path = os.path.join(project_root, "model", "best_gradcam_layer.txt")
    if os.path.exists(layer_path):
        with open(layer_path, "r", encoding="utf-8") as handle:
            layer_name = handle.read().strip()
        if layer_name:
            return layer_name
    return default_layer


def find_layer_recursive(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer:
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return find_layer_recursive(layer, layer_name)
            except ValueError:
                continue
    raise ValueError(f"Layer {layer_name} could not be found in model {model.name}.")


def build_feature_extractor(model: tf.keras.Model, target_layer_name: Optional[str] = None) -> tf.keras.Model:
    target_layer_name = target_layer_name or get_best_gradcam_layer()
    target_layer = find_layer_recursive(model, target_layer_name)
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
        name=f"{model.name}_{target_layer_name}_extractor",
    )
