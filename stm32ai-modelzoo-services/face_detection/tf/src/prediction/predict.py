# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from onnx import ModelProto
import onnxruntime
from hydra.core.hydra_config import HydraConfig

from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.data_augmentation import remap_pixel_values_range
from face_detection.tf.src.utils import ai_runner_invoke, bbox_normalized_to_abs_coords, plot_bounding_boxes, keypoints_normalized_to_abs_coords
from face_detection.tf.src.models import model_family
from face_detection.tf.src.preprocessing  import get_prediction_data_loader
from face_detection.tf.src.postprocessing  import get_nmsed_detections

def crop_and_save(image_path, image_array, boxes, base_filename, output_dir, cfg, stretch_percents=None):
    """
    Crop and save images with independent stretching for each coordinate.

    Args:
        image_path: Path to the original image.
        image_array: The resized or processed image array (used for coordinate normalization).
        boxes: List of bounding boxes (xmin, ymin, xmax, ymax) relative to image_array.
        base_filename: Base filename for saving crops.
        output_dir: Directory to save cropped images.
        cfg: Configuration object.
        stretch_percents: List of 4 floats (stretch_xmin%, stretch_ymin%, stretch_xmax%, stretch_ymax%)
                         representing the stretch percentage for each coordinate.
                         If None, defaults to (0, 0, 0, 0) (no stretch).
    """
    if stretch_percents is None:
        stretch_percents = [0, 0, 0, 0]

    original_image = cv2.imread(image_path.numpy().decode('utf-8'))
    if original_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    h_array, w_array = image_array.shape[:2]
    h_orig, w_orig = original_image.shape[:2]

    # Create a subfolder for this image inside the output directory
    image_folder = os.path.join(output_dir, base_filename)
    os.makedirs(image_folder, exist_ok=True)

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box

        # Normalize coordinates based on image_array size
        xmin_norm = xmin / w_array
        ymin_norm = ymin / h_array
        xmax_norm = xmax / w_array
        ymax_norm = ymax / h_array

        # Scale normalized coordinates to original image size
        xmin_scaled = int(xmin_norm * w_orig)
        ymin_scaled = int(ymin_norm * h_orig)
        xmax_scaled = int(xmax_norm * w_orig)
        ymax_scaled = int(ymax_norm * h_orig)

        # Calculate width and height of the box
        box_width = xmax_scaled - xmin_scaled
        box_height = ymax_scaled - ymin_scaled

        # Unpack stretch percentages for each coordinate
        stretch_xmin_percent, stretch_ymin_percent, stretch_xmax_percent, stretch_ymax_percent = stretch_percents

        # Calculate stretch amounts for each coordinate
        stretch_xmin = int(box_width * (stretch_xmin_percent / 100))
        stretch_ymin = int(box_height * (stretch_ymin_percent / 100))
        stretch_xmax = int(box_width * (stretch_xmax_percent / 100))
        stretch_ymax = int(box_height * (stretch_ymax_percent / 100))

        # Apply stretching by adjusting each coordinate independently
        xmin_stretched = max(0, xmin_scaled - stretch_xmin)
        ymin_stretched = max(0, ymin_scaled - stretch_ymin)
        xmax_stretched = min(w_orig - 1, xmax_scaled + stretch_xmax)
        ymax_stretched = min(h_orig - 1, ymax_scaled + stretch_ymax)

        # Check if coordinates are valid after stretching
        if xmax_stretched <= xmin_stretched or ymax_stretched <= ymin_stretched:
            # Skipping invalid box
            continue

        cropped_bgr = original_image[ymin_stretched:ymax_stretched, xmin_stretched:xmax_stretched]

        if cropped_bgr.size == 0:
            # Skipping empty crop for box
            continue

        # Convert BGR to RGB for displaying with matplotlib
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

        # Show the cropped image using matplotlib
        if cfg.general.display_figures:
            plt.figure(figsize=(4, 4))
            plt.imshow(cropped_rgb)
            plt.title(f"Crop {i}")
            plt.axis('off')
            plt.show()

        # Save the cropped image inside the image-specific folder
        output_filename = os.path.join(image_folder, f"{base_filename}_crop_{i}.jpg")
        cv2.imwrite(output_filename, cropped_bgr)

def _view_image_and_boxes(cfg, image, img_path, boxes=None,key_points=None, classes=None, scores=None, class_names=None):
    """
    Display image and bounding boxes, optionally cropping and saving detected regions.

    Args:
        cfg: Configuration object.
        image: Image array.
        img_path: Path to the image file.
        boxes: Bounding boxes.
        classes: Class indices.
        scores: Detection scores.
        class_names: List of class names.
    """
    # Convert TF tensors to numpy
    image = np.array(image, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.int32)
    key_points = np.array(key_points, dtype=np.int32)
    classes = np.array(classes, dtype=np.int32)

    file_name_with_extension = os.path.basename(img_path.numpy().decode('utf-8'))
    file_name, _ = os.path.splitext(file_name_with_extension)
    output_dir = "{}/{}".format(HydraConfig.get().runtime.output_dir,"predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Calculate dimensions for the displayed image
    image_width, image_height = np.shape(image)[:2]
    display_size = 7
    if image_width >= image_height:
        x_size = display_size
        y_size = round((image_width / image_height) * display_size)
    else:
        x_size = round((image_height / image_width) * display_size)
        y_size = display_size

    # Display the image and the bounding boxes
    fig, ax = plt.subplots(figsize=(x_size, y_size))
    if cfg.preprocessing.color_mode.lower() == 'bgr':
        image = image[...,::-1]
    ax.imshow(image)
    plot_bounding_boxes(ax, boxes,key_points, classes, scores, class_names)
    # turning off the grid
    plt.grid(visible=False)
    plt.axis('off')
    plt.savefig('{}/{}_predict.jpg'.format(output_dir,file_name))
    if cfg.general.display_figures:
        plt.show()
    plt.close()
    # Crop and save predicted boxes
    if model_family(cfg.model.model_type) in ["facedetect_front",'yunet']:
        crop_and_save(img_path, image, boxes, file_name, output_dir, cfg, stretch_percents = cfg.postprocessing.crop_stretch_percents)
    

def _predict_float_model(cfg, model):
    """
    Run inference using a loaded Keras model object.
    Args:
        cfg: Configuration object.
        model: Loaded tf.keras.Model object.
    """
    image_size = model.input.shape[1:3]
    cpr = cfg.preprocessing.rescaling
    pixels_range = (cpr.offset, 255 * cpr.scale + cpr.offset)
    data_loader = get_prediction_data_loader(cfg, image_size=image_size)
    cpp = cfg.postprocessing
    for images, image_paths in data_loader:
        batch_size = tf.shape(images)[0]

        # Predict the images and get the NMS'ed detections
        predictions = model(images)

        # Decode and NMS the predictions
        boxes, scores, classes, key_points = get_nmsed_detections(cfg, predictions, image_size)
 
        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        key_points = keypoints_normalized_to_abs_coords(key_points, image_size=image_size)      
        for i in range(batch_size):
            _view_image_and_boxes(cfg, 
                                 images[i],
                                 image_paths[i],
                                 boxes[i],
                                 key_points[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)


def _predict_quantized_model(cfg, interpreter):
    """
    Run inference using a loaded TFLite Interpreter object.
    Args:
        cfg: Configuration object.
        interpreter: Loaded tf.lite.Interpreter object.
    """
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = "tflite_model"  # No path, so use generic name
    ai_runner_interpreter = ai_runner_interp(target, name_model)
    input_details = interpreter.get_input_details()[0]
    batch_size = 1
    input_shape = tuple(input_details['shape'][1:])
    image_size = input_shape[:2]
    #image_size = tuple(input_shape)[-3:-1]
    
    output_details = interpreter.get_output_details()

    data_loader = get_prediction_data_loader(cfg, image_size=image_size, batch_size=batch_size)
    cpr = cfg.preprocessing.rescaling
    pixels_range = (cpr.offset, 255 * cpr.scale + cpr.offset)
    cpp = cfg.postprocessing
    
    for images, image_paths in data_loader:
        batch_size = tf.shape(images)[0]

        # Allocate input tensor to predict the batch of images
        input_index = input_details['index']
        tensor_shape = (batch_size,) + input_shape
        interpreter.resize_tensor_input(input_index, tensor_shape)
        interpreter.allocate_tensors()

        input_dtype = input_details['dtype']
        is_float = np.issubdtype(input_dtype, np.floating)

        if is_float:
            predict_images = images
        else:
            # Rescale the image using the model's coefficients
            scale = input_details['quantization'][0]
            zero_points = input_details['quantization'][1]
            predict_images = images / scale + zero_points
    
        # Convert the image data type to the model input data type
        predict_images = tf.cast(predict_images, input_dtype)
        # and clip to the min/max values of this data type
        if is_float:
            min_val = np.finfo(input_dtype).min
            max_val = np.finfo(input_dtype).max
        else:
            min_val = np.iinfo(input_dtype).min
            max_val = np.iinfo(input_dtype).max

        predict_images = tf.clip_by_value(predict_images, min_val, max_val)

        if target == 'host':
            # Predict the images
            interpreter.set_tensor(input_index, predict_images)
            interpreter.invoke()
            if model_family(cfg.model.model_type) in ["facedetect_front"]:
                predictions = []
                # face_detect_model_front
                predictions_r = (interpreter.get_tensor(output_details[0]['index']),
                                interpreter.get_tensor(output_details[1]['index']),
                                interpreter.get_tensor(output_details[2]['index']),
                                interpreter.get_tensor(output_details[3]['index']))
                for i, pred in enumerate(predictions_r):
                    is_float = np.issubdtype(pred.dtype, np.floating)
                    if not is_float:
                        scale, zero_point = output_details[i]['quantization']
                        out_deq = (pred.astype(np.float32) - zero_point) * scale
                        predictions.append(out_deq)
                    else:
                        predictions.append(pred)
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            data        = ai_interp_input_quant(ai_runner_interpreter,images.numpy(),'.tflite')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)
        
        # Decode and NMS the predictions
        boxes, scores, classes, key_points = get_nmsed_detections(cfg, predictions, image_size)
 
        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        key_points = keypoints_normalized_to_abs_coords(key_points, image_size=image_size)      
        for i in range(batch_size):
            _view_image_and_boxes(cfg, 
                                 images[i],
                                 image_paths[i],
                                 boxes[i],
                                 key_points[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)


def _predict_onnx_model(cfg, model, num_classes=None):
    """
    Run inference using a loaded ONNX InferenceSession object.

    Args:
        cfg: Configuration object.
        model: Loaded onnxruntime.InferenceSession object.
        num_classes: Optional, not used in main logic.
    """
    print("ONNX model prediction: {}".format(cfg.model.model_path))
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = "onnx_model"  # No path, so use generic name
    ai_runner_interpreter = ai_runner_interp(target, name_model)

    # Get the model input shape
    input_shape = model.get_inputs()[0].shape
    batch_size = 1

    input_chpos = getattr(cfg.prediction, 'input_chpos', 'chlast') if hasattr(cfg, 'prediction') else 'chlast'
    if cfg.model.framework == "tf":
        # Dataloader is channel last with TF
        if input_chpos=="chfirst" or target == 'host':
            need_transpose = True
        else:
            need_transpose = False
    else:
        # Dataloader is already channel first with Torch
        need_transpose = False

    inputs  = model.get_inputs()
    outputs = model.get_outputs()
    input_shape = tuple(input_shape)[-3:]
    image_size = tuple(input_shape)[1:]

    # Create the data loader
    data_loader = get_prediction_data_loader(cfg, image_size=image_size, batch_size=batch_size)
    
    cpr = cfg.preprocessing.rescaling
    # if the scale and offsets are 3 number lists instead of scalars using averages
    offset = np.mean(cpr.offset) if isinstance(cpr.offset, (list, tuple)) else cpr.offset
    scale = np.mean(cpr.scale) if isinstance(cpr.scale, (list, tuple)) else cpr.scale
    
    # calculating pixels range
    pixels_range = (offset, 255 * scale + offset)

    inputs  = model.get_inputs()
    outputs = model.get_outputs()

    for images, image_paths in data_loader:
        batch_size = tf.shape(images)[0]
        images = images.numpy()

        if need_transpose == True:
            if images.ndim == 4:
                # For a 4D array with shape [n, h, w, c], the new order will be [n, c, h, w]
                axes_order = (0, 3, 1, 2)
            elif images.ndim == 3:
                # For a 3D array with shape [n, h, c], the new order will be [n, c, h]
                axes_order = (0, 2, 1)
            else:
                raise ValueError("The input array must have either 3 or 4 dimensions.")
            t_images = np.transpose(images, axes_order)
        else:
            t_images = images
        if target == 'host':
            predictions = model.run([o.name for o in outputs], {inputs[0].name: t_images})
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            data        = ai_interp_input_quant(ai_runner_interpreter,t_images,'.onnx')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if len(predictions) == 1:
            predictions = predictions[0]

        # Decode and NMS the predictions
        boxes, scores, classes, key_points = get_nmsed_detections(cfg, predictions, image_size)
 
        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        key_points = keypoints_normalized_to_abs_coords(key_points, image_size=image_size)      
        for i in range(batch_size):
            _view_image_and_boxes(cfg, 
                                 images[i],
                                 image_paths[i],
                                 boxes[i],
                                 key_points[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)
                                 

def predict(cfg, model):
    """
    Run inference on all the images within the test set.

    Args:
        cfg (config): The configuration file.
        model: The loaded model object (Keras model, TFLite Interpreter, or ONNX InferenceSession).
    Returns:
        None.
    """
    print("Use ctl+c to exit the script")

    # Keras model
    if isinstance(model, tf.keras.Model):
        _predict_float_model(cfg, model)
    # TFLite Interpreter
    elif isinstance(model, tf.lite.Interpreter):
        _predict_quantized_model(cfg, model)
    # ONNX InferenceSession
    elif isinstance(model, onnxruntime.InferenceSession):
        _predict_onnx_model(cfg, model)
    else:
        raise RuntimeError("Evaluation internal error: unsupported model type")