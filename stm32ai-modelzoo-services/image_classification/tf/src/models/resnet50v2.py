# /*---------------------------------------------------------------------------------------------
#  * Copyright 2018 The TensorFlow Authors.
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import keras
from keras import layers
from keras.src.applications import imagenet_utils


def get_resnet50v2(input_shape: tuple, num_classes: int = None, dropout: float = None, 
                   pretrained: bool = True, **kwargs) -> keras.Model:
    """
    Returns a ResNet50v2 model with a custom classifier.

    Args:
        input_shape (tuple): The shape of the input tensor.
        dropout (float, optional): The dropout rate for the custom classifier. Defaults to 1e-6.
        num_classes (int, optional): The number of output classes. Defaults to None.
        pretrained (tool, optional): The pre-trained weights to use. Either "imagenet"
        or None. Defaults to "imagenet".

    Returns:
        keras.Model: The ResNet50V2 model with a custom classifier.
    """


    if dropout:
        # Model loaded for training
        base_model = keras.applications.resnet_v2.ResNet50V2(input_shape=input_shape, 
                                            weights="imagenet" if pretrained else None, 
                                            pooling="avg",
                                            classes=num_classes,
                                            classifier_activation="softmax",
                                            include_top=False)
        x = layers.Dropout(rate=dropout, name="dropout")(base_model.output)
        if num_classes > 2:
            outputs = layers.Dense(num_classes, activation="softmax")(x)
        else:
            outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        # Instantiate a base model
        base_model = keras.applications.resnet_v2.ResNet50V2(input_shape=input_shape, 
                                            weights="imagenet" if pretrained else None, 
                                            pooling="avg",
                                            classes=num_classes,
                                            classifier_activation="softmax",
                                            include_top=True)
        outputs = base_model.output
    
    # Create the Keras model
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="resnet50v2")

    return model

