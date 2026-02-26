#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import Optional, List, Union
import tensorflow as tf
import numpy as np


def ai_runner_invoke(image_processed,ai_runner_interpreter):
    preds, _ = ai_runner_interpreter.invoke(image_processed)
    preds = np.squeeze(preds, axis=0)   # Remove 5th outputted dimension
    preds = np.array(preds, dtype=np.float32)
    return preds

def change_model_number_of_classes(model,num_classes):

    output_shape = num_classes

    # If the model already has the correct number of classes -> dont do anything
    for outp in model.outputs:
        if outp.shape[-1] == output_shape:
            return model

    l = -1
    l_list = []

    while True:

        layer_type = type(model.layers[l])
        layer_config = model.layers[l].get_config()

        if layer_type in [tf.keras.layers.Conv2D,
                          tf.keras.layers.Conv2DTranspose,
                          tf.keras.layers.Conv1D,
                          tf.keras.layers.Conv1DTranspose,
                          tf.keras.layers.Dense]:
            if layer_type in [tf.keras.layers.Conv2D,tf.keras.layers.Conv2DTranspose,tf.keras.layers.Conv1D,tf.keras.layers.Conv1DTranspose]:
                layer_config['filters'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)
            else:
                layer_config['units'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)

            for i,new_l in enumerate(l_list[::-1]):
                outputs = new_l(outputs)

            return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)

        else:
            l_list.append(layer_type(**layer_config))
            l-=1

    return None

def change_model_input_shape(model,new_inp_shape):
    
    conf = model.get_config()
    conf['layers'][0]['config']['batch_shape'] = new_inp_shape
    new_model = model.__class__.from_config(conf, custom_objects={})

    # iterate over all the layers that we want to get weights from
    weights = [layer.get_weights() for layer in model.layers[1:]]
    for layer, weight in zip(new_model.layers[1:], weights):
        layer.set_weights(weight)

    old_inp_shape = model.get_config()['layers'][0]['config']['batch_shape']

    return new_model, old_inp_shape

def get_loss(num_classes: int) -> tf.keras.losses:
    """
    Returns the appropriate loss function based on the number of classes in the dataset.

    Args:
        num_classes (int): The number of classes in the dataset.

    Returns:
        tf.keras.losses: The appropriate loss function based on the number of classes in the dataset.
    """
    # We use the sparse version of the categorical crossentropy because
    # this is what we use to load the dataset.
    if num_classes > 2:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return loss

def segmentation_loss(logits: tf.Tensor, labels: tf.Tensor, num_classes: int = 21, ignore_label: 
                      int = 255, loss_weights: Optional[Union[List[float], tf.Tensor]] = None) -> tf.Tensor:
    """
    Calculate the weighted softmax cross-entropy loss for segmentation tasks.

    Args:
        logits (tf.Tensor): The raw output of the network, which represents the
                            prediction for each pixel.
        labels (tf.Tensor): The ground truth labels for each pixel.
        num_classes (int, optional): The number of classes in the segmentation task.
                                     Defaults to 21.
        ignore_label (int, optional): The label that should be ignored in the loss
                                      computation. Defaults to 255.
        loss_weights (list or tf.Tensor, optional): Weights for each class that are
                                                    applied to the loss. The default
                                                    is None, which creates a list of
                                                    weights with 0.5 for the background
                                                    class and 1.0 for all other classes.

    Returns:
        tf.Tensor: The computed loss as a scalar tensor.
    """
    # If no specific loss weights are provided, initialize them with default values.
    if loss_weights is None:
        if num_classes == 21:
            loss_weights = [0.5] + [1.0] * (num_classes - 1)
        else:
            loss_weights = [1.0] * (num_classes - 1)

    with tf.name_scope('seg_loss'):
        # Flatten logits and labels tensors for processing.
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.reshape(labels, [-1])

        # Create a mask to exclude the ignored label from the loss computation.
        not_ignored_mask = tf.not_equal(labels, ignore_label)
        labels = tf.boolean_mask(labels, not_ignored_mask)
        logits = tf.boolean_mask(logits, not_ignored_mask)

        # Cast labels to an integer type for further processing.
        labels = tf.cast(labels, tf.int32)

        # Apply class weights if provided.
        class_weights = tf.constant(loss_weights, dtype=tf.float32)
        weights = tf.gather(class_weights, labels)
        # Convert labels to one-hot encoding for compatibility with softmax cross entropy.
        labels_one_hot = tf.one_hot(labels, depth=num_classes)

        # Compute the softmax cross entropy loss for each pixel.
        pixel_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        weighted_pixel_losses = pixel_losses * weights
        total_loss = tf.reduce_sum(weighted_pixel_losses)

        # Calculate the number of pixels that contribute to the loss (excluding ignored ones).
        num_positive = tf.reduce_sum(tf.cast(not_ignored_mask, tf.float32))

        # Normalize the loss by the number of contributing pixels to get the final loss value.
        loss = total_loss / (num_positive + 1e-5)

        return loss


def get_custom_loss(num_classes: int = 21) -> tf.keras.losses.Loss:
    """
    Creates a custom loss function for a segmentation model with predefined parameters.

    Args:
        num_classes (int, optional): The number of classes in the segmentation task.
                                     Defaults to 21.
        ignore_label (int, optional): The label that should be ignored in the loss
                                      computation. Defaults to 255.
        loss_weights (Optional[Union[List[float], tf.Tensor]], optional): Weights for each class that are
                                                    applied to the loss. The default
                                                    is None, which creates a list of
                                                    weights with 0.5 for the background
                                                    class and 1.0 for all other classes.

    Returns:
        tf.keras.losses.Loss: A custom loss function that can be used in training a model.
    """
    def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        The custom loss function that will be used during model training.

        Args:
            y_true (tf.Tensor): The ground truth labels for each pixel.
            y_pred (tf.Tensor): The predicted labels for each pixel.

        Returns:
            tf.Tensor: The computed loss as a scalar tensor.
        """
        # Call the segmentation_loss function with the predefined parameters.
        return segmentation_loss(logits=y_pred, labels=y_true, num_classes=num_classes)

    return custom_loss