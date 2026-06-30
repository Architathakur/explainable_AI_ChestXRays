import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from xai_pneumonia.utils import ensure_dir, get_project_root, set_global_determinism


class F1ScorePneumonia(tf.keras.metrics.Metric):
    def __init__(self, name: str = "f1_pneumonia", class_index: int = 1, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.class_index = class_index
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_positive = tf.cast(tf.equal(y_true_labels, self.class_index), tf.float32)
        y_pred_positive = tf.cast(tf.equal(y_pred_labels, self.class_index), tf.float32)

        tp = tf.reduce_sum(y_true_positive * y_pred_positive)
        fp = tf.reduce_sum((1.0 - y_true_positive) * y_pred_positive)
        fn = tf.reduce_sum(y_true_positive * (1.0 - y_pred_positive))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self) -> tf.Tensor:
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self) -> None:
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def reset_states(self) -> None:
        self.reset_state()

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update({"class_index": self.class_index})
        return config


def build_resnet50_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weights: str = "imagenet",
) -> tf.keras.Model:
    set_global_determinism(42)
    inputs = layers.Input(shape=input_shape, name="input_image")
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
    )
    backbone.trainable = not freeze_backbone

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_pneumonia_classifier")
    compile_model(model, learning_rate=learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1ScorePneumonia(name="f1_pneumonia"),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )


def unfreeze_last_layers(model: tf.keras.Model, n_layers: int = 30) -> None:
    for layer in model.layers:
        layer.trainable = False

    head_layers = {"global_average_pooling", "dropout_1", "dense_512", "batch_norm", "dropout_2", "predictions"}
    for layer in model.layers:
        if layer.name in head_layers:
            layer.trainable = True

    start_unfreeze = False
    for layer in model.layers:
        if layer.name == "conv5_block1_1_conv":
            start_unfreeze = True
        if start_unfreeze:
            layer.trainable = True

    trainable_layers = [layer for layer in model.layers if layer.trainable and layer.name not in head_layers]
    if len(trainable_layers) < n_layers:
        for layer in model.layers[-n_layers:]:
            layer.trainable = True


def get_training_callbacks(
    output_dir: Optional[str] = None,
    patience: int = 10,
) -> List[callbacks.Callback]:
    project_root = get_project_root()
    model_dir = ensure_dir(os.path.join(project_root, "model"))
    output_dir = output_dir or model_dir
    ensure_dir(output_dir)
    checkpoint_path = os.path.join(model_dir, "best_model.h5")

    return [
        callbacks.EarlyStopping(
            monitor="val_f1_pneumonia",
            patience=patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_f1_pneumonia",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_model_history(history: Dict[str, List[float]], output_path: Optional[str] = None) -> str:
    project_root = get_project_root()
    model_dir = ensure_dir(os.path.join(project_root, "model"))
    output_path = output_path or os.path.join(model_dir, "history.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return output_path


def load_trained_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"F1ScorePneumonia": F1ScorePneumonia},
    )
