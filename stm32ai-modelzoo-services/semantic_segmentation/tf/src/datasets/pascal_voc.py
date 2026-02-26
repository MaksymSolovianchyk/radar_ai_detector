# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024-2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import os
import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict
from omegaconf import DictConfig
from typing import Tuple

COLOR_MAP = {
    (0, 0, 0): 0,          # background
    (128, 0, 0): 1,        # aeroplane
    (0, 128, 0): 2,        # bicycle
    (128, 128, 0): 3,      # bird
    (0, 0, 128): 4,        # boat
    (128, 0, 128): 5,      # bottle
    (0, 128, 128): 6,      # bus
    (128, 128, 128): 7,    # car
    (64, 0, 0): 8,         # cat
    (192, 0, 0): 9,        # chair
    (64, 128, 0): 10,      # cow
    (192, 128, 0): 11,     # dining table
    (64, 0, 128): 12,      # dog
    (192, 0, 128): 13,     # horse
    (64, 128, 128): 14,    # motorbike
    (192, 128, 128): 15,   # person
    (0, 64, 0): 16,        # potted plant
    (128, 64, 0): 17,      # sheep
    (0, 192, 0): 18,       # sofa
    (128, 192, 0): 19,     # train
    (0, 64, 128): 20       # tv/monitor
}


def _check_and_convert_mask(rgb_image: tf.Tensor) -> tf.Tensor:
    """
    Checks the mask values and converts an RGB image to a label map if necessary using the predefined COLOR_MAP.

    Args:
        rgb_image (tf.Tensor): A tensor of shape (height, width, 3) containing RGB values.

    Returns:
        tf.Tensor: A tensor of shape (height, width) containing the label map or the original image.
    """
    # Get unique values and cast to int32 for consistency
    unique_values = tf.cast(tf.unique(tf.reshape(rgb_image, [-1]))[0], tf.int32)

    # Check if any value is outside the range [0, 20]
    if tf.reduce_any(tf.logical_or(unique_values < 0, unique_values > 20)):
        # Create an empty label map
        label_map = tf.zeros((tf.shape(rgb_image)[0], tf.shape(rgb_image)[1]), dtype=tf.int32)
        
        # Iterate over the color map and assign labels
        for color, label in COLOR_MAP.items():
            color_tensor = tf.constant(color, dtype=tf.uint8)
            mask = tf.reduce_all(tf.equal(rgb_image, color_tensor), axis=-1)
            # tf.print("Comparing with color:", color_tensor, "Mask sum:", tf.reduce_sum(tf.cast(mask, tf.uint8)))
            label_map = tf.where(mask, tf.fill(tf.shape(label_map), label), label_map)
        
        return tf.cast(label_map, tf.uint8)  # Ensure label map is uint8 after conversion
    else:
        # tf.print('Mask values are within the acceptable range.')
        return tf.cast(rgb_image, tf.uint8)  # Return the original image if no conversion is needed
    

def _preprocess_mask(mask_path: str = None, input_size: list = None, aspect_ratio: str = None,
                    interpolation_method: str = 'nearest') -> tf.Tensor:
    """
    Loads the mask file and pre-process it according to configuration parameters

    Args:
        mask_path (str): path to the mask of the candidate image
        input_size (list): image resolution in pixels [height, width]
        aspect_ratio (str): "fit" or "crop"
        interpolation_method (str): resizing interpolation method, default is 'nearest'

    Returns:
        A tensor containing the pre-processed mask
    """
    # Read and decode the mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask)

    # Resize the mask based on the aspect ratio
    if aspect_ratio == "fit":
        mask = tf.image.resize(mask, input_size, method=interpolation_method, preserve_aspect_ratio=False)
    else:
        mask = tf.image.resize_with_crop_or_pad(mask, input_size[0], input_size[1])

    # Cast the mask to uint8 and normalize
    mask = tf.cast(mask, tf.uint8)
    mask = tf.where(mask == 255, tf.zeros_like(mask), mask)

    return mask


def _get_image(image_path: str = None, images_nb_channels: int = None, aspect_ratio: str = None,
              interpolation: str = None, scale: float = None, offset: int = None, input_size: list = None) -> tf.Tensor:
    """
    Loads an image from a file path. Resize it with appropriate interpolation method. Scale it according to
    config parameters.

    Args:
        image_path (str): path to candidate image
        images_nb_channels (int): 1 if greyscale, 3 if RGB
        aspect_ratio (str): "fit" or "crop"
        interpolation (str): resizing interpolation method
        scale (float): rescaling pixels value
        offset (int): offset value on pixels
        input_size (list): image resolution in pixels [height, width]

    Returns:
        A tensor containing the pixels, appropriately resized and scaled

    """
    # load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=images_nb_channels)

    height = input_size[0]
    width = input_size[1]

    if aspect_ratio == "fit":
        img = tf.image.resize(img, [height, width], method=interpolation, preserve_aspect_ratio=False)
    else:
        img = tf.image.resize_with_crop_or_pad(img, height, width)

    # Rescale the image
    img_processed = scale * tf.cast(img, tf.float32) + offset

    return img_processed


def _load_image_and_mask_pascal_voc(image_path: str = None, mask_path: str = None, images_nb_channels: int = None,
                                   aspect_ratio: str = None, interpolation: str = None, scale: float = None,
                                   offset: int = None, input_size: list = None) -> tuple:
    """
        Loads an image and a mask from a file path. Preprocess them according to config file parameters

        Args:
            image_path (str): path to candidate image
            mask_path (str): path to corresponding mask
            images_nb_channels (int): 1 if greyscale, 3 if RGB
            aspect_ratio (str): "fit' or "crop"
            interpolation (str): resizing interpolation method
            scale (float): rescaling pixels value
            offset (int): offset value on pixels
            input_size (list): image resolution in pixels [height, width]

        Returns:
            A tuple of tensor containing the image pixels and the mask, appropriately resized and scaled

        """
    input_image = _get_image(image_path, images_nb_channels, aspect_ratio, interpolation, scale, offset, input_size)
    if mask_path=='None':
        input_mask = tf.cast(tf.zeros((input_size[0],input_size[1],3)),tf.uint8)
    else:
        input_mask = _preprocess_mask(mask_path, input_size, aspect_ratio, interpolation_method='nearest')
    
    return input_image, input_mask


def _get_path_dataset(images_dir: str = None, masks_dir: str = None, ids_file_path: str = None, seed: int = None,
                     shuffle: bool = True) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a dataset root directory path.

    Args:
        images_dir (str): path to image directory for dataset construction
        masks_dir (str): path to mask directory for dataset construction
        ids_file_path: (str): file path. The file contains a list of image to be considered for dataset creation.
        seed (int): seed when performing shuffle.
        shuffle (bool): Initial shuffling (or not) of the input files paths.

    Returns:
        dataset(tf.data.Dataset) -> dataset with a tuple (path, label) of each sample. 
    """

    if masks_dir is not None:
        with open(ids_file_path, 'r') as file:
            ids = file.read().splitlines()

        image_paths = [os.path.join(images_dir, img_id + ".jpg") for img_id in ids]
        mask_paths = [os.path.join(masks_dir, img_id + ".png") for img_id in ids]  # Adjust the extension if necessary

        # Filter out non-existing files
        existing_image_paths = []
        existing_mask_paths = []
        for img_path, msk_path in zip(image_paths, mask_paths):
            if os.path.exists(img_path) and os.path.exists(msk_path):
                existing_image_paths.append(img_path)
                existing_mask_paths.append(msk_path)
            else:
                print(f"Warning: Skipping {img_path} because the image or mask does not exist.")
        data_list = [existing_image_paths, existing_mask_paths]
    else:
        image_paths = [os.path.join(images_dir,file) for file in os.listdir(images_dir) if file.endswith(".jpg")]
        mask_paths  = ['None']*len(image_paths)

        data_list = [image_paths, mask_paths]

    if shuffle:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data_list[0]))
        data_list = [np.take(data_list[0], perm, axis=0), np.take(data_list[1], perm, axis=0)]

    return data_list


def _get_train_eval_ds(images_path: str = None, images_masks: str = None, files_path: str = None,
                      image_size: tuple[int] = None, batch_size: int = None, seed: int = None,
                      shuffle: bool = True, to_cache: bool = False,
                      validation_split: float = 0.2, color_mode: str = "rgb",
                      aspect_ratio: str = "fit", interpolation: str = "bilinear",
                      scale: float = 1.0, offset: float = 0.0) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images under a given dataset root directory and returns training 
    and validation tf.Data.datasets.

    Args:
        images_path (str): path to image directory for dataset construction
        images_masks (str): path to mask directory for dataset construction
        files_path: (str): file path. The file contains a list of image to be considered for dataset creation.
        image_size (tuple[int]): Size of the input images to resize them to.
        batch_size (int): Batch size to use for training and validation.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to shuffle the dataset at each iteration.
        to_cache (bool): Whether or not to cache the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    channels = 1 if color_mode == "grayscale" else 3

    datalist = _get_path_dataset(images_dir=images_path, masks_dir=images_masks, ids_file_path=files_path, seed=seed)

    dataset = tf.data.Dataset.from_tensor_slices((datalist[0], datalist[1]))

    train_size = int(len(dataset)*(1-validation_split))
    train_ds = dataset.take(train_size)
    eval_ds = dataset.skip(train_size)

    if shuffle:
        train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True, seed=seed)
    
    # Map the paths to the actual loaded and preprocessed images and masks
    mapping_params = (channels, aspect_ratio, interpolation, scale, offset, image_size)
    train_ds = train_ds.map(lambda img, msk: _load_image_and_mask_pascal_voc(img, msk, *mapping_params),
                            num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.map(lambda img, msk: _load_image_and_mask_pascal_voc(img, msk, *mapping_params),
                        num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    eval_ds = eval_ds.batch(batch_size)

    if to_cache:
        train_ds = train_ds.cache()
        eval_ds = eval_ds.cache()
    
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    eval_ds = eval_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, eval_ds


def get_ds(images_path: str = None, images_masks: str = None, files_path: str = None,
           image_size: tuple[int] = None, batch_size: int = None, seed: int = None,
           shuffle: bool = True, to_cache: bool = False, color_mode: str = "rgb",
           aspect_ratio: str = "fit", interpolation: str = "bilinear",
           scale: float = 1.0, offset: float = 0.0) -> tf.data.Dataset:
    """
    Loads the images from the given dataset root directory and returns a tf.data.Dataset.

    Args:
        images_path (str): path to image directory for dataset construction
        images_masks (str): path to mask directory for dataset construction
        files_path: (str): file path. The file contains a list of image to be considered for dataset creation.
        image_size (tuple[int]): Size of the input images to resize them to.
        batch_size (int): Batch size to use for the dataset.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to shuffle the dataset at each iteration.
        to_cache (bool): Whether or not to cache the dataset.

    Returns:
        tf.data.Dataset: Dataset containing the images.
    """
    channels = 1 if color_mode == "grayscale" else 3
    datalist = _get_path_dataset(images_dir=images_path, masks_dir=images_masks, ids_file_path=files_path, seed=seed)

    dataset = tf.data.Dataset.from_tensor_slices((datalist[0], datalist[1]))
    
    if shuffle:
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True, seed=seed)

    mapping_params = (channels, aspect_ratio, interpolation, scale, offset, image_size)
    dataset = dataset.map(lambda img, msk: _load_image_and_mask_pascal_voc(img, msk, *mapping_params),
                          num_parallel_calls=tf.data.AUTOTUNE)
   
    dataset = dataset.batch(batch_size)

    if to_cache:
        dataset = dataset.cache()
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_pred_ds(prediction_path: str,
                image_size: Tuple[int, int],
                batch_size: int,
                color_mode: str,
                aspect_ratio: str,
                interpolation: str,
                scale: float,
                offset: float) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding (image_tensor, path_string).
    Keep batch_size=1 to simplify downstream usage.
    """
    if not prediction_path or not os.path.isdir(prediction_path):
        return None
    channels = 1 if color_mode == "grayscale" else 3
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    file_paths = [os.path.join(prediction_path, f)
                  for f in os.listdir(prediction_path)
                  if f.lower().endswith(exts) and os.path.isfile(os.path.join(prediction_path, f))]
    if not file_paths:
        return None

    def _load(path):
        data = tf.io.read_file(path)
        img = tf.image.decode_image(data, channels=channels)
        img.set_shape([None, None, channels])
        h, w = image_size
        if aspect_ratio == "fit":
            img = tf.image.resize(img, [h, w], method=interpolation, preserve_aspect_ratio=False)
        else:
            img = tf.image.resize_with_crop_or_pad(img, h, w)
        img = scale * tf.cast(img, tf.float32) + offset
        return img, path

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1)  # keep path easy to extract
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_pascal_voc(training_path: str, training_masks_path: str, training_files_path: str,
                    validation_path: str, validation_masks_path: str, validation_files_path: str,
                    test_path: str, test_masks_path: str, test_files_path: str,
                    prediction_path: str, quantization_path: str, quantization_files_path: str, quantization_split: Optional[float],
                    image_size: Tuple[int, int], color_mode: str, batch_size: int, seed: int,
                    aspect_ratio: str, interpolation: str, validation_split: float,
                    scale: float = 1.0, offset: float = 0.0
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images from the given dataset root directories and returns training,
    validation, and test tf.data.Datasets.

    Args:
        cfg: (dict): config dictionary
        image_size (tuple[int]): resizing (width, height) of input images
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, quantization, pred,
        and test datasets.
    """


    if training_path and not validation_path:
        # There is no validation. We split the training set in two to create one.
        train_ds, eval_ds = _get_train_eval_ds(images_path=training_path, images_masks=training_masks_path,
                                            files_path=training_files_path, image_size=image_size,
                                            batch_size=batch_size, seed=seed, shuffle=True, to_cache=False,
                                            validation_split=validation_split, color_mode=color_mode,
                                            aspect_ratio=aspect_ratio, interpolation=interpolation,
                                            scale=scale, offset=offset)
    elif training_path and validation_path:
        train_ds = get_ds(images_path=training_path, images_masks=training_masks_path,
                          files_path=training_files_path, image_size=image_size,
                          batch_size=batch_size, seed=seed, shuffle=True, to_cache=False,
                          color_mode=color_mode, aspect_ratio=aspect_ratio,
                          interpolation=interpolation, scale=scale, offset=offset)

        eval_ds = get_ds(images_path=validation_path, images_masks=validation_masks_path,
                        files_path=validation_files_path, image_size=image_size,
                        batch_size=batch_size, seed=seed, shuffle=False, to_cache=False,
                        color_mode=color_mode, aspect_ratio=aspect_ratio,
                        interpolation=interpolation, scale=scale, offset=offset)
    elif validation_path:
        eval_ds = get_ds(images_path=validation_path, images_masks=validation_masks_path,
                        files_path=validation_files_path, image_size=image_size,
                        batch_size=batch_size, seed=seed, shuffle=False, to_cache=False,
                        color_mode=color_mode, aspect_ratio=aspect_ratio,
                        interpolation=interpolation, scale=scale, offset=offset)
        train_ds = None
    else:
        train_ds = None
        eval_ds = None

    if quantization_path:
        quantization_ds = get_ds(images_path=quantization_path, images_masks=None,
                                 files_path=quantization_files_path, image_size=image_size,
                                 batch_size=1, seed=seed, shuffle=False, to_cache=False,
                                 color_mode=color_mode, aspect_ratio=aspect_ratio,
                                 interpolation=interpolation, scale=scale, offset=offset)
        if quantization_split:
            dataset_size = int(len(quantization_ds) * quantization_split)
            quantization_ds = quantization_ds.take(dataset_size)
    elif training_path is not None:
        quantization_ds = get_ds(images_path=training_path, images_masks=None,
                                 files_path=None, image_size=image_size,
                                 batch_size=1, seed=seed, shuffle=False, to_cache=False,
                                 color_mode=color_mode, aspect_ratio=aspect_ratio,
                                 interpolation=interpolation, scale=scale, offset=offset)
        if quantization_split:
            dataset_size = int(len(quantization_ds) * quantization_split)
            quantization_ds = quantization_ds.take(dataset_size)
    else:
        quantization_ds = None

    if test_path:
        test_ds = get_ds(images_path=test_path, images_masks=test_masks_path,
                         files_path=test_files_path, image_size=image_size,
                         batch_size=batch_size, seed=seed, shuffle=False, to_cache=False,
                         color_mode=color_mode, aspect_ratio=aspect_ratio,
                         interpolation=interpolation, scale=scale, offset=offset)
    else:
        test_ds = None

    if prediction_path:
        predict_ds = get_pred_ds(prediction_path=prediction_path,
                                 image_size=image_size,
                                 batch_size=1,
                                 color_mode=color_mode,
                                 aspect_ratio=aspect_ratio,
                                 interpolation=interpolation,
                                 scale=scale,
                                 offset=offset)
    else:
        predict_ds = None
    dataloaders={'train': train_ds, 'valid': eval_ds, 'test': test_ds, 'quantization': quantization_ds, 'predict': predict_ds}
    return dataloaders




