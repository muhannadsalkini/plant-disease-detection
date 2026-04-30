"""
src/models.py
-------------
Defines all three model architectures described in the paper:

  Model 1 — Custom CNN (trained from scratch, §IV.C)
  Model 2 — EfficientNetB0 with transfer learning (§IV.D)
  Model 3 — MobileNetV2 with transfer learning (§IV.E)

Each builder function returns a compiled tf.keras.Model ready for training.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build optimizer from string name
# ─────────────────────────────────────────────────────────────────────────────

def _get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from adam/sgd/rmsprop.")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Custom CNN (Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int   = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float          = config.LEARNING_RATE,
    optimizer: str     = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    Three convolutional blocks (32 → 64 → 128 filters), each with:
        Conv2D (3×3)  →  BatchNorm  →  ReLU  →  MaxPool (2×2)
    Followed by:
        GlobalAveragePooling  →  Dense(256, ReLU)  →  Dropout  →  Softmax(38)
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — EfficientNetB0 (Transfer Learning)
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnetb0(
    input_shape: tuple  = (*config.IMAGE_SIZE, 3),
    num_classes: int    = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float           = config.LEARNING_RATE,
    optimizer: str      = config.OPTIMIZER,
    trainable_base: bool = False,         # False  = feature-extraction phase
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
) -> tf.keras.Model:
    """
    EfficientNetB0 with ImageNet weights.

    Phase 1 (feature extraction): base_model frozen, only top layers trained.
    Phase 2 (fine-tuning):        top `fine_tune_layers` of base unfrozen.

    Call build_efficientnetb0(trainable_base=True) for the fine-tuning phase,
    or use the provided `unfreeze_top_layers()` helper.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    if not trainable_base:
        base_model.trainable = False
    else:
        # Unfreeze only the top N layers
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Classification head
    # EfficientNetB0 was pretrained on [0,255] pixels; our pipeline sends [0,1].
    # Rescale back to [0,255] before the base model.
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=255.0)(inputs)   # [0,1] → [0,255]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="efficientnetb0")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — MobileNetV2 (Transfer Learning)
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2(
    input_shape: tuple  = (*config.IMAGE_SIZE, 3),
    num_classes: int    = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float           = config.LEARNING_RATE,
    optimizer: str      = config.OPTIMIZER,
    trainable_base: bool = False,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
) -> tf.keras.Model:
    """
    MobileNetV2 with ImageNet weights.
    Same two-phase strategy as EfficientNetB0.
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    if not trainable_base:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Classification head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: unfreeze top N layers and recompile for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def prepare_fine_tuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
    fine_tune_lr: float   = config.FINE_TUNE_LR,
    optimizer: str        = config.OPTIMIZER,
):
    """
    Unfreeze the top `fine_tune_layers` layers of `base_model` and recompile
    `model` with a lower learning rate for the fine-tuning phase.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=_get_optimizer(optimizer, fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[Models] Fine-tuning: {fine_tune_layers} top layers unfrozen, lr={fine_tune_lr}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model: tf.keras.Model):
    # Keras 3 compatible: use np.prod(w.shape) instead of backend.count_params
    trainable     = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    total         = trainable + non_trainable
    print(f"\n{'─'*40}")
    print(f"  Model       : {model.name}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable   : {trainable:,}")
    print(f"  Frozen      : {non_trainable:,}")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cnn = build_custom_cnn()
    print_model_summary(cnn)

    eff, eff_base = build_efficientnetb0()
    print_model_summary(eff)

    mob, mob_base = build_mobilenetv2()
    print_model_summary(mob)
