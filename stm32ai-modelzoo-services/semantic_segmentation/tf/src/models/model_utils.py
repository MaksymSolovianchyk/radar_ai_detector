# /*---------------------------------------------------------------------------------------------
#  * Copyright 2018 The TensorFlow Authors.
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig


def prepare_kwargs_for_model(cfg: DictConfig):
    """
    Prepares a dictionary of keyword arguments for model instantiation based on the configuration.

    Args:
        cfg (DictConfig): The configuration object containing model, training, and dataset parameters.

    Returns:
        dict: A dictionary of keyword arguments to be used for model creation.
    """
    model_kwargs = {
        'alpha': getattr(cfg.model, 'alpha', None),
        'model_type': getattr(cfg.model, 'model_type', None),
        'input_shape': getattr(cfg.model, 'input_shape', None),
        'pretrained': getattr(cfg.model, 'pretrained', None),
        'dropout': getattr(cfg.training, 'dropout', None),
        'backbone': getattr(cfg.model, 'backbone', None),
        'output_stride': getattr(cfg.model, 'output_stride', None),
        'num_classes': getattr(cfg.dataset, 'num_classes', None),
    }
    return model_kwargs