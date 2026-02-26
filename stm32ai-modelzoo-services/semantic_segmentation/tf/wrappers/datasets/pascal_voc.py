# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024-2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from semantic_segmentation.tf.src.datasets import load_pascal_voc, prepare_kwargs_for_dataloader


__all__ = ["get_pascal_voc"]

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='pascal_voc', use_case='semantic_segmentation')
def get_pascal_voc(cfg):
    """
    Thin wrapper: prepare args then call load_pascal_voc (single build location).

    Args:
        cfg: configuration object.

    Returns:
        Dict with keys: train, val, test, predict, quantization.
    """
    args = prepare_kwargs_for_dataloader(cfg)
    dataloaders = load_pascal_voc(
        training_path=args['training_path'],
        training_masks_path=args['training_masks_path'],
        training_files_path=args['training_files_path'],
        validation_path=args['validation_path'],
        validation_masks_path=args['validation_masks_path'],
        validation_files_path=args['validation_files_path'],
        validation_split=args["validation_split"],
        test_path=args['test_path'],
        test_masks_path=args['test_masks_path'],
        test_files_path=args['test_files_path'],
        prediction_path=args['prediction_path'],
        quantization_path=args['quantization_path'],
        quantization_files_path=args['quantization_files_path'],
        quantization_split=args['quantization_split'],
        image_size=args['image_size'],
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        aspect_ratio=args['aspect_ratio'],
        interpolation=args['interpolation'],
        scale=args['scale'],
        offset=args['offset']
    )
    return dataloaders

# Additional aliases
for _alias in ("pascal_voc_person", "person_coco_2017_pascal_voc_2012", "coco_2017_pascal_voc_2012"):
    DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name=_alias, use_case='semantic_segmentation')(get_pascal_voc)
