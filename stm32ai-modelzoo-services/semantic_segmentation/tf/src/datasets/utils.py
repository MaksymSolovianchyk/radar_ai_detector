# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024-2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import tensorflow as tf
from typing import Dict, Optional, List


def preprocess_data(dataloaders: Dict[str, tf.data.Dataset],
                    scale: float = None,
                    offset: float = None,
                    mean: Optional[List[float]] = None,
                    std: Optional[List[float]] = None) -> Dict[str, tf.data.Dataset]:
    """
    Apply scale/offset and optional mean/std normalization to image tensors only.
    Supports datasets yielding either dicts with key 'image' or tuples (image, mask).
    Masks are left unchanged.
    """
    def _prep(sample):
        # Handle (image, mask) tuple
        if isinstance(sample, tuple):
            # This branch is unused now; tuple handling done by specialized mapper below.
            raise RuntimeError("Tuple samples should be processed by tuple-aware mapper, not _prep.")
        # Handle dict format
        if 'image' not in sample:
            raise KeyError("Sample missing 'image' key.")
        img = tf.cast(sample['image'], tf.float32)
        if scale is not None:
            img *= scale
        if offset is not None:
            img += offset
        if mean is not None and std is not None:
            img = (img - tf.constant(mean, dtype=tf.float32)) / tf.constant(std, dtype=tf.float32)
        sample['image'] = img
        return sample
    for k, ds in dataloaders.items():
        if ds is not None:
            # Peek first element structure to choose mapping strategy
            try:
                it = iter(ds)
                first = next(it)
            except StopIteration:
                continue
            except Exception:
                # Fallback to generic _prep
                dataloaders[k] = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
                continue
            if isinstance(first, tuple):
                # Expect (image, mask) -> map with function receiving two positional args
                def _tuple_map(img, mask):
                    img = tf.cast(img, tf.float32)
                    if scale is not None:
                        img *= scale
                    if offset is not None:
                        img += offset
                    if mean is not None and std is not None:
                        img = (img - tf.constant(mean, dtype=tf.float32)) / tf.constant(std, dtype=tf.float32)
                    return img, mask
                dataloaders[k] = ds.map(_tuple_map, num_parallel_calls=tf.data.AUTOTUNE)
            elif isinstance(first, dict):
                dataloaders[k] = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                # Unknown structure; leave unchanged
                pass
    return dataloaders


def prepare_kwargs_for_dataloader(cfg):
    """
    Collect dataloader parameters.

    Args:
        cfg: configuration object.

    Returns:
        Dict of all required dataloader kwargs including image_size from model.input_shape.
    """
    input_shape = getattr(cfg.model, 'input_shape', None)
    model_path = cfg.model.model_path
    file_extension = str(model_path).split('.')[-1]
    input_shape = cfg.model.input_shape
    if file_extension in ['h5', 'keras']:
        image_size = tuple(input_shape)[:-1]
    elif file_extension == 'tflite':
        image_size = tuple(input_shape)[-3:-1]
    elif file_extension == 'onnx':
        image_size = tuple(input_shape)[-2:]
    
    dataloader_kwargs = {
        'training_path': cfg.dataset.training_path,
        'training_masks_path': cfg.dataset.training_masks_path,
        'training_files_path': cfg.dataset.training_files_path,
        'validation_path': cfg.dataset.validation_path,
        'validation_masks_path': cfg.dataset.validation_masks_path,
        'validation_files_path': cfg.dataset.validation_files_path,
        'validation_split': cfg.dataset.validation_split,
        'test_path': cfg.dataset.test_path,
        'test_masks_path': cfg.dataset.test_masks_path,
        'test_files_path': cfg.dataset.test_files_path,
        'prediction_path': cfg.dataset.prediction_path,
        'prediction_files_path': cfg.dataset.prediction_files_path,
        'quantization_path': cfg.dataset.quantization_path,
        'quantization_files_path': cfg.dataset.quantization_files_path,
        'quantization_split': cfg.dataset.quantization_split,
        'image_size': tuple(image_size),
        'color_mode': cfg.preprocessing.color_mode,
        'batch_size': getattr(cfg.training, 'batch_size', 32) if cfg.training else 32,
        'seed': cfg.general.global_seed,
        'aspect_ratio': cfg.preprocessing.resizing.aspect_ratio,
        'interpolation': cfg.preprocessing.resizing.interpolation,
        'scale': cfg.preprocessing.rescaling.scale,
        'offset': cfg.preprocessing.rescaling.offset,
    }
    return dataloader_kwargs