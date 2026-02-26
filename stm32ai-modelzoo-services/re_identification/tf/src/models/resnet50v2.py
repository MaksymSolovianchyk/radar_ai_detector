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


def get_resnet50v2(input_shape: tuple, num_classes: int = None, 
                    dropout: float = None, pretrained: bool = False, **kwargs) -> keras.Model:
    """
    Returns a ResNet50v2 model with a custom classifier.

    Args:
        input_shape (tuple): The shape of the input tensor.
        dropout (float, optional): The dropout rate for the custom classifier. Defaults to 1e-6.
        num_classes (int, optional): The number of output classes. Defaults to None.
        pretrained_weights (str, optional): The pre-trained weights to use. Either "imagenet"
        or None. Defaults to "imagenet".

    Returns:
        keras.Model: The ResNet50V2 model with a custom classifier.
    """
    
    # commented because no longer used later-on
#    if pretrained_weights:
#        training = False
#        # model is set in inference mode so that moving avg and var of any BN are kept untouched
#        # should help the convergence according to Keras tutorial
#    else:
#        training = True

    pretrained_weights = "imagenet" if pretrained else None
    # Instantiate a base model
    base_model = keras.applications.resnet_v2.ResNet50V2(input_shape=input_shape, 
                                        weights=pretrained_weights, 
                                        pooling="avg",
                                        classes=num_classes,
                                        classifier_activation="softmax",
                                        include_top=False)

    if dropout:
        x = layers.Dropout(rate=dropout, name="dropout")(base_model.output)
        if num_classes > 2:
            outputs = layers.Dense(num_classes, activation="softmax")(x)
        else:
            outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        if num_classes > 2:
            outputs = layers.Dense(num_classes, activation="softmax")(base_model.output)
        else:
            outputs = layers.Dense(1, activation="sigmoid")(base_model.output)
    
    # Create the Keras model
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="resnet50_v2")

    return model

