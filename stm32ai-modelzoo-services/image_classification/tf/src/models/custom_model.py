from typing import Tuple, Optional
import keras
from keras import layers


def get_custom_model(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    dropout: Optional[float] = None,
    pretrained: bool = False,
    **kwargs
) -> keras.Model:
    """
    Lightweight custom image classification model for embedded deployment.

    Intended for grayscale or sensor-like inputs such as (96, 64, 1).
    """

    if pretrained:
        print("WARNING: No pretrained weights are available for 'custom_model'. Random weights will be used.")

    if input_shape is None:
        raise ValueError("`input_shape` must be provided.")
    if num_classes is None:
        raise ValueError("`num_classes` must be provided.")

    inputs = keras.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(16, kernel_size=3, padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    # Block 2
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool3")(x)

    # Optional extra capacity, still lightweight
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    if dropout is not None and dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_model")
    return model
