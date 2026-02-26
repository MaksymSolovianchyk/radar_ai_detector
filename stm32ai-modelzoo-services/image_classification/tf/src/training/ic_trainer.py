
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple, List, Dict, Optional

import mlflow
from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import numpy as np 
import tensorflow as tf

# Suppress TensorFlow warnings to reduce log clutter
import logging
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import utility functions and modules
from common.utils import (
    log_to_file, log_last_epoch_history, LRTensorBoard, check_training_determinism,
    model_summary, collect_callback_args, vis_training_curves
)
from common.training import (
    set_frozen_layers, set_dropout_rate, get_optimizer, lr_schedulers,
    set_all_layers_trainable_parameter
)
from image_classification.tf.src.utils import get_loss, change_model_number_of_classes, change_model_input_shape
from image_classification.tf.src.data_augmentation import DataAugmentationLayer


# Define a custom callback for multi-resolution training
class MultiResCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to dynamically change the input resolution
    of the model during training.

    Args:
        image_sizes (List[int]): List of resolutions to cycle through.
        period (int): Number of batches before changing resolution.
        name (str, optional): Name of the callback.
    """
    def __init__(self, image_sizes, period, name=None):
        super().__init__()
        self.resolutions = image_sizes
        self.period = period

    def on_train_batch_begin(self, batch, logs=None):
        # Change the resolution of the input layer based on the batch number
        res = self.resolutions[((batch - 1) // self.period) % len(self.resolutions)]
        self.model.layers[0].change_res(res)


# Function to add preprocessing layers to the model
def _add_preprocessing_layers(
        model: tf.keras.Model,
        input_shape: Tuple = None,
        scale: float = None,
        offset: float = None,
        mean: float = None,
        std: float = None,
        data_augmentation: Dict = None,
        batches_per_epoch: float = None):
    """
    Adds preprocessing layers (rescaling and data augmentation) to the model.

    Args:
        model (tf.keras.Model): The base model.
        input_shape (Tuple): Input shape of the model.
        scale (float): Scaling factor for rescaling.
        offset (float): Offset for rescaling.
        mean (float): Mean for normalization.
        std (float): Standard deviation for normalization.
        data_augmentation (Dict): Data augmentation configuration.
        batches_per_epoch (float): Number of training batches per epoch.

    Returns:
        tf.keras.Model: The augmented model with preprocessing layers.
    """
    data_aug_args = DefaultMunch.fromDict(data_augmentation.config)
    if data_aug_args.random_periodic_resizing is not None:
        model, _ = change_model_input_shape(model, (None, None, None, 3))

    model_layers = []
    model_layers.append(tf.keras.Input(shape=input_shape))

    # Add data augmentation layer if specified
    if data_augmentation:
        # defining rescaling and normalization in case the three values are provided for std and mean
        if isinstance(std, float) and isinstance(mean, float):
            pixels_range = ((offset - mean) / std, (scale * 255 + offset - mean) / std)
        elif isinstance(std, list) and isinstance(mean, list):
            if len(std) != 3 or len(mean) != 3:
                raise ValueError("If std and mean are lists, they must have three elements each.")
            pixel_range_min = [(offset - m) / s for m, s in zip(mean, std)]
            pixel_range_max = [(scale * 255 + offset - m) / s for m, s in zip(mean, std)]
            pixels_range = (min(pixel_range_min), max(pixel_range_max))
        else:
            raise TypeError("std and mean must be either floats or lists of length 3.")

        model_layers.append(
            DataAugmentationLayer(
                data_augmentation_fn=data_augmentation.function_name,
                config=data_augmentation.config,
                pixels_range=pixels_range,
                batches_per_epoch=batches_per_epoch
            )
        )
    model_layers.append(model)
    augmented_model = tf.keras.Sequential(model_layers, name="augmented_model")

    return augmented_model


# Function to create Keras callbacks
def _get_callbacks(callbacks_dict: DictConfig, output_dir: str = None, logs_dir: str = None,
                   saved_models_dir: str = None) -> List[tf.keras.callbacks.Callback]:
    """
    Creates a list of Keras callbacks for training.

    Args:
        callbacks_dict (DictConfig): Configuration for callbacks.
        output_dir (str): Directory for saving outputs.
        logs_dir (str): Directory for saving logs.
        saved_models_dir (str): Directory for saving models.

    For each callback, the attributes and their values used in the config
    file are used to create a string that is the callback instantiation as
    it would be written in a Python script. Then, the string is evaluated.
    If the evaluation succeeds, the callback object is returned. If it fails,
    an error is thrown with a message saying that the name and/or arguments
    of the callback are incorrect.

    Returns:
        List[tf.keras.callbacks.Callback]: List of callbacks.
    """
    message = "\nPlease check the 'training.callbacks' section of your configuration file."
    lr_scheduler_names = lr_schedulers.get_scheduler_names()
    num_lr_schedulers = 0

    # Generate the callbacks used in the config file (there may be none)
    callback_list = []
    if callbacks_dict is not None:
        if type(callbacks_dict) != DefaultMunch:
            raise ValueError(f"\nInvalid callbacks syntax{message}")
        for name in callbacks_dict.keys():
            if name in ("ModelCheckpoint", "TensorBoard", "CSVLogger"):
                raise ValueError(f"\nThe `{name}` callback is built-in and can't be redefined.{message}")
            elif name in lr_scheduler_names:
                text = f"lr_schedulers.{name}"
            elif name == 'MultiResCallback':
                text = f"{name}"
            else:
                text = f"tf.keras.callbacks.{name}"

            # Add the arguments to the callback string
            # and evaluate it to get the callback object
            text += collect_callback_args(name, args=callbacks_dict[name], message=message)
            try:
                callback = eval(text)
            except ValueError as error:
                raise ValueError(f"\nThe callback name `{name}` is unknown, or its arguments are incomplete "
                                 f"or invalid\nReceived: {text}{message}") from error
            callback_list.append(callback)

            if name in lr_scheduler_names + ["ReduceLROnPlateau", "LearningRateScheduler"]:
                num_lr_schedulers += 1
            
    # Check that there is only one scheduler
    if num_lr_schedulers > 1:
        raise ValueError(f"\nFound more than one learning rate scheduler{message}")

    # Add built-in callbacks that saves the best model obtained so far
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, saved_models_dir, "best_augmented_model.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ))
    # Add the Keras callback that saves the model at the end of the epoch
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, saved_models_dir, "last_augmented_model.keras"),
        save_best_only=False,
        monitor="val_accuracy",
        mode="max"
    ))
    # Add the TensorBoard callback
    callback_list.append(LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir)))
    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback_list.append(tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv")))

    return callback_list


# Main class for training image classification models
class ICTrainer:
    def __init__(self, cfg, model=None, dataloaders=None):
        """
        Initializes the trainer with configuration, model, and datasets.

        Args:
            cfg: Configuration object.
            model: TensorFlow model.
            dataloaders: Dictionary containing training, validation, and test datasets.
        """
        self.cfg = cfg
        self.model = model
        self.train_ds = dataloaders['train']
        self.valid_ds = dataloaders['valid']
        self.test_ds = dataloaders['test']

        self.output_dir = HydraConfig.get().runtime.output_dir
        self.saved_models_dir = cfg.general.saved_models_dir
        self.class_names = cfg.dataset.class_names
        self.num_classes = len(self.class_names)
        self.augmented_model = None
        self.callbacks = None
        self.history = None

    def prepare(self):
        """
        Prepares the model, datasets, and callbacks for training.
        """
        # Print dataset statistics
        print("Dataset stats:")
        train_size = sum([x.shape[0] for x, _ in self.train_ds])
        valid_size = sum([x.shape[0] for x, _ in self.valid_ds])
        if self.test_ds:
            test_size = sum([x.shape[0] for x, _ in self.test_ds])

        print("  classes:", self.num_classes)
        print("  training set size:", train_size)
        print("  validation set size:", valid_size)
        if self.test_ds:
            print("  test set size:", test_size)
        else:
            print("  no test set")

        # Log dataset information
        if self.cfg.dataset.dataset_name:
            log_to_file(self.output_dir, f"Dataset : {self.cfg.dataset.dataset_name}")

        # Prepare the model
        if self.cfg.model:
            cfm = self.cfg.model
            print(f"[INFO] : Using `{cfm.model_name}` model")
            log_to_file(self.cfg.output_dir, (f"Model name : {cfm.model_name}"))
        elif self.cfg.model.model_path:
            self.model = change_model_number_of_classes(self.model, self.num_classes)
            print(f"[INFO] : Initialized model with weights from model file {self.cfg.model.model_path}")
            log_to_file(self.cfg.output_dir, (f"Weights from model file : {self.cfg.model.model_path}"))

        # Add preprocessing layers if not resuming training
        if self.cfg.training.resume_training_from:
            model_summary(self.model)
            self.augmented_model = self.model
        else:
            model_summary(self.model)
            input_shape = tuple(self.model.inputs[0].shape[1:])
            self.augmented_model = _add_preprocessing_layers(
                self.model,
                input_shape=input_shape,
                scale=self.cfg.preprocessing.rescaling.scale,
                offset=self.cfg.preprocessing.rescaling.offset,
                mean=getattr(self.cfg.preprocessing.normalization, 'mean', 0.0),
                std=getattr(self.cfg.preprocessing.normalization, 'std', 1.0),
                data_augmentation=self.cfg.data_augmentation,
                batches_per_epoch=len(self.train_ds)
            )
            self.augmented_model.compile(
                loss=get_loss(num_classes=self.num_classes),
                metrics=['accuracy'],
                optimizer=get_optimizer(cfg=self.cfg.training.optimizer)
            )

        # Configure MultiResCallback if applicable
        data_aug_args = DefaultMunch.fromDict(self.cfg.data_augmentation.config)
        if data_aug_args.random_periodic_resizing is not None:
            rpr = DefaultMunch.fromDict(data_aug_args.random_periodic_resizing)
            if rpr.image_sizes is not None:
                self.cfg.training.callbacks['MultiResCallback'] = DefaultMunch.fromDict({
                    'image_sizes': rpr.image_sizes,
                    'period': rpr.period if rpr.period is not None else 10
                })
            else:
                print("[WARNING]: 'random_periodic_resizing' can't be used because [image_sizes] argument is missing.")

        # Generate callbacks
        self.callbacks = _get_callbacks(
            callbacks_dict=self.cfg.training.callbacks,
            output_dir=self.output_dir,
            saved_models_dir=self.saved_models_dir,
            logs_dir=self.cfg.general.logs_dir
        )

    def enable_determinism(self):
        """
        Enables deterministic operations for reproducibility.
        """
        if self.cfg.general.deterministic_ops:
            sample_ds = self.train_ds.take(1)
            tf.config.experimental.enable_op_determinism()
            if not check_training_determinism(self.augmented_model, sample_ds):
                print("[WARNING]: Some operations cannot be run deterministically. Setting deterministic_ops to False.")
                tf.config.experimental.enable_op_determinism.__globals__["_pywrap_determinism"].enable(False)

    def fit(self):
        """
        Trains the model using the training dataset.
        """
        print("Starting training...")
        start_time = timer()
        steps_per_epoch = self.cfg.training.dryrun if self.cfg.training.dryrun else None
        self.history = self.augmented_model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks
        )
        last_epoch = log_last_epoch_history(self.cfg, self.output_dir)
        end_time = timer()
        fit_run_time = int(end_time - start_time)
        average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1), 2)
        print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
        log_to_file(self.cfg.output_dir, (
            f"Training runtime : {fit_run_time} s\n" +
            f"Average time per epoch : {average_time_per_epoch} s"
        ))
        vis_training_curves(history=self.history, output_dir=self.output_dir)

    def save_and_evaluate(self):
        """
        Saves the best model and evaluates it on validation and test datasets.
        """
        # Load the best model checkpoint
        models_dir = os.path.join(self.output_dir, self.saved_models_dir)
        checkpoint_filepath = os.path.join(models_dir, "best_augmented_model.keras")
        checkpoint_model = tf.keras.models.load_model(
            checkpoint_filepath,
            custom_objects={'DataAugmentationLayer': DataAugmentationLayer}
        )
        output_model_input_shape = tuple(self.model.inputs[0].shape)
        best_model = checkpoint_model.layers[-1]
        best_model, _ = change_model_input_shape(best_model, output_model_input_shape)
        best_model.compile(loss=get_loss(self.num_classes), metrics=['accuracy'])
        best_model_path = os.path.join(self.output_dir, f"{self.saved_models_dir}/best_model.keras")
        best_model.save(best_model_path)
        setattr(best_model, 'model_path', best_model_path)
        print('[INFO] : Training complete.')
        return best_model

    def train(self):
        """
        Executes the full training pipeline: prepare, train, save, and evaluate.
        """
        self.prepare()
        self.enable_determinism()
        self.fit()
        return self.save_and_evaluate()
