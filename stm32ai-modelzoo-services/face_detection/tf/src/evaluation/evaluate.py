# /*---------------------------------------------------------------------------------------------keras
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import shutil
from pathlib import Path
from string import ascii_letters, digits
import random
from timeit import default_timer as timer
from datetime import timedelta
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from tabulate import tabulate
import math
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from onnx import ModelProto
import onnxruntime

from face_detection.tf.src.postprocessing import get_nmsed_detections, get_detections
from face_detection.tf.src.preprocessing import get_evaluation_data_loader
from face_detection.tf.src.models import model_family
from face_detection.tf.src.utils import ai_runner_invoke, bbox_normalized_to_abs_coords, ObjectDetectionMetricsData, \
                  calculate_objdet_metrics, calculate_average_metrics
from common.utils import count_h5_parameters, log_to_file, \
                         ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import model_is_quantized



def evaluate_keras_model(cfg: DictConfig, model: tf.keras.Model, num_classes: int = None) -> dict:
    """
    Evaluate a Keras object detection model on the evaluation dataset.

    Args:
        cfg (DictConfig): Configuration object containing evaluation parameters.
        model (tf.keras.Model): The Keras model to evaluate.
        num_classes (int, optional): Number of classes in the dataset.

    Returns:
        dict: Dictionary of evaluation metrics for each class.
    """
    tf.print(f'[INFO] : Evaluating the Keras model {model.model_path}')
    # Create the data loader
    image_size = model.input.shape[1:3]
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])

    exmpl,_  = iter(data_loader).next()

    batch_size = exmpl.shape[0]

    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])
    
    cpp = cfg.postprocessing
    metrics_data = None
    num_detections = 0

    for i,data in enumerate(tqdm(data_loader)):
        images, gt_labels = data
        image_size = tf.shape(images)[1:3]

        # Predict the images, decode and NMS the detections
        predictions = model(images)

        # Decode the predictions
        boxes, scores, keypoints = get_detections(cfg, predictions, image_size)

        if i==0:
            num_detections = boxes.shape[1]
            metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, num_classes, num_detections, dataset_size, batch_size)

        metrics_data.add_data(gt_labels, boxes, scores, keypoints)
        metrics_data.update_batch_index(i, cfg.postprocessing.confidence_thresh, cfg.postprocessing.NMS_thresh, image_size)

    
    groundtruths, detections = metrics_data.get_data()
    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics


def _evaluate_tflite_quantized_model(cfg: DictConfig,  model: str, num_classes: int = None, output_dir: str = None) -> dict:
    """
    Evaluate a quantized TFLite object detection model on the evaluation dataset.

    Args:
        cfg (DictConfig): Configuration object containing evaluation parameters.
        model (str): TFLite Interpreter object.
        num_classes (int, optional): Number of classes in the dataset.
        output_dir (str, optional): Directory to save evaluation outputs.

    Returns:
        dict: Dictionary of evaluation metrics for each class.
    """
    tf.print(f'[INFO] : Evaluating the quantized model {model.model_path}')
    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(model.model_path)
    ai_runner_interpreter = ai_runner_interp(target,name_model)

    input_details = model.get_input_details()[0]

    model_batch_size = input_details['shape_signature'][0]
    if model_batch_size!=1 and target == 'host':
        batch_size = 64
    else:
        batch_size = 1

    input_shape = tuple(input_details['shape'][1:])
    image_size = input_shape[:2]
    #image_size = tuple(input_shape)[-3:-1]
    
    output_details = model.get_output_details()

    # Create the data loader
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False, batch_size=batch_size)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])

    exmpl,_  = iter(data_loader).next()

    batch_size = exmpl.shape[0]

    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])

    cpp = cfg.postprocessing
    metrics_data = None
    num_detections = 0
    predictions_all = []
    images_full = []

    for i,data in enumerate(tqdm(data_loader)):
        imag, gt_labels = data
        batch_size = int(tf.shape(imag)[0])

        # Allocate input tensor to predict the batch of images
        input_index = input_details['index']
        tensor_shape = (batch_size,) + input_shape
        
        model.resize_tensor_input(input_index, tensor_shape)
        model.allocate_tensors()
    
        input_dtype = input_details['dtype']
        is_float = np.issubdtype(input_dtype, np.floating)

        if is_float:
            predict_images = imag
        else:
            # Rescale the image using the model's coefficients
            scale = input_details['quantization'][0]
            zero_points = input_details['quantization'][1]
            predict_images = imag / scale + zero_points
    
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

        if "evaluation" in cfg and cfg.evaluation:
            if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input==True: 
                images_full.append(predict_images)

        if target == 'host':
            # Predict the images
            model.set_tensor(input_index, predict_images)
            model.invoke()
            if model_family(cfg.model.model_type) in ["facedetect_front"]:
                predictions = []
                # face_detect_model_front
                predictions_r = (model.get_tensor(output_details[0]['index']),
                                model.get_tensor(output_details[1]['index']),
                                model.get_tensor(output_details[2]['index']),
                                model.get_tensor(output_details[3]['index']))
                for j, pred in enumerate(predictions_r):
                    is_float = np.issubdtype(pred.dtype, np.floating)
                    if not is_float:
                        scale, zero_point = output_details[j]['quantization']
                        out_deq = (pred.astype(np.float32) - zero_point) * scale
                        predictions.append(out_deq)
                    else:
                        predictions.append(pred)
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            data        = ai_interp_input_quant(ai_runner_interpreter,imag.numpy(),'.tflite')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if "evaluation" in cfg and cfg.evaluation:
            if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output==True:
                predictions_all.append(predictions)

        # Decode the predictions
        boxes, scores, keypoints = get_detections(cfg, predictions, image_size)

        if i==0:
            num_detections = boxes.shape[1]
            metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, num_classes, num_detections, dataset_size, batch_size)

        metrics_data.add_data(gt_labels, boxes, scores, keypoints)
        metrics_data.update_batch_index(i, cfg.postprocessing.confidence_thresh, cfg.postprocessing.NMS_thresh, image_size)

    # Saves evaluation dataset in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input==True: 
            if "npy_in_name" in cfg.evaluation and cfg.evaluation.npy_in_name:
                npy_in_name = cfg.evaluation.npy_in_name
            else:
                npy_in_name = "unknown_npy_in_name"
            images_full = np.concatenate(images_full, axis=0)
            np.save(os.path.join(output_dir, f"{npy_in_name}.npy"), images_full)

    # Saves model output in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output==True: 
            if "npy_out_name" in cfg.evaluation and cfg.evaluation.npy_out_name:
                npy_out_name = cfg.evaluation.npy_out_name
            else:
                npy_out_name = "unknown_npy_out_name"
            predictions_all = np.concatenate(predictions_all, axis=0)
            np.save(os.path.join(output_dir, f"{npy_out_name}.npy"), predictions_all)

    groundtruths, detections = metrics_data.get_data()
    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics    
    

def _evaluate_onnx_model(cfg: DictConfig, model: str, num_classes: int = None) -> dict:
    """
    Evaluate an ONNX object detection model on the evaluation dataset.

    Args:
        cfg (DictConfig): Configuration object containing evaluation parameters.
        model (str): ONNX InferenceSession object.
        num_classes (int, optional): Number of classes in the dataset.

    Returns:
        dict: Dictionary of evaluation metrics for each class.
    """
    tf.print(f'[INFO] : Evaluating the ONNX model {model.model_path}')
    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(model.model_path)

    onx = ModelProto()
    with open(model.model_path, "rb") as f:
        content = f.read()
        onx.ParseFromString(content)
      
    # Get the model input shape
    input_shape = model.get_inputs()[0].shape
    batch_size = input_shape[0]
        
    ai_runner_interpreter = ai_runner_interp(target,name_model)
    
    model_batch_size = input_shape[0]
    if model_batch_size!=1 and target == 'host':
        batch_size = 64
    else:
        batch_size = 1

    input_chpos = getattr(cfg.evaluation, 'input_chpos', 'chlast') if hasattr(cfg, 'evaluation') else 'chlast'
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
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False, batch_size=batch_size)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])

    exmpl,_  = iter(data_loader).next()

    batch_size = exmpl.shape[0]

    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])

    cpp = cfg.postprocessing
    metrics_data = None
    num_detections = 0

    for i,data in enumerate(tqdm(data_loader)):
        images, gt_labels = data
        images = images.numpy()

        # Predict the images
        if need_transpose == True:
            if images.ndim == 4:
                # For a 4D array with shape [n, h, w, c], the new order will be [n, c, h, w]
                axes_order = (0, 3, 1, 2)
            elif images.ndim == 3:
                # For a 3D array with shape [n, h, c], the new order will be [n, c, h]
                axes_order = (0, 2, 1)
            else:
                raise ValueError("The input array must have either 3 or 4 dimensions.")
            images = np.transpose(images, axes_order)

        if target == 'host':
            predictions = model.run([o.name for o in outputs], {inputs[0].name: images})
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            data        = ai_interp_input_quant(ai_runner_interpreter,images,'.onnx')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if len(predictions) == 1:
            predictions = predictions[0]

        # Decode the predictions
        #predictions = [np.transpose(e, [0, 2, 3, 1]) for e in predictions] 
        boxes, scores, keypoints = get_detections(cfg, predictions, image_size)

        if i==0:
            num_detections = boxes.shape[1]
            metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, num_classes, num_detections, dataset_size, batch_size)

        metrics_data.add_data(gt_labels, boxes, scores, keypoints)
        metrics_data.update_batch_index(i, cfg.postprocessing.confidence_thresh, cfg.postprocessing.NMS_thresh, image_size)


    groundtruths, detections = metrics_data.get_data()

    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics    
    
    
def _display_objdet_metrics(metrics, class_names):
    
    table = []
    classes = list(metrics.keys())    
    for c in sorted(classes):
        table.append([
            class_names[c],
            round(100 * metrics[c].pre, 1),
            round(100 * metrics[c].rec, 1),
            round(100 * metrics[c].ap, 1)])
            
    print()
    headers = ["Class name", "Precision %", "  Recall %", "   AP %  "]
    print()
    print(tabulate(table, headers=headers, tablefmt="pipe", numalign="center"))

    mpre, mrec, mAP = calculate_average_metrics(metrics)
    
    print("\nAverages over classes %:")
    print("-----------------------")
    print(" Mean precision: {:.1f}".format(100 * mpre))
    print(" Mean recall:    {:.1f}".format(100 * mrec))
    print(" Mean AP (mAP):  {:.1f}".format(100 * mAP))


def _plot_precision_versus_recall(metrics, class_names, plots_dir):
    """
    Plot the precision versus recall curves. AP values are the areas under these curves.
    """

    # Create the directory where plots will be saved
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)

    for c in list(metrics.keys()):
        
        # Plot the precision versus recall curve
        figure = plt.figure(figsize=(10, 10))
        plt.xlabel("recall")
        plt.ylabel("interpolated precision")
        plt.title("Class '{}' (AP = {:.2f})".
                    format(class_names[c], metrics[c].ap * 100))
        plt.plot(metrics[c].interpolated_precision, metrics[c].interpolated_recall)
        plt.grid()

        # Save the plot in the plots directory
        plt.savefig(f"{plots_dir}/{class_names[c]}.png")
        plt.close(figure)


def evaluate(cfg: DictConfig, model: tf.keras.Model = None) -> None:
    """
    Evaluate an object detection model (Keras, TFLite, or ONNX) and log metrics.

    Args:
        cfg (DictConfig): Configuration object containing evaluation parameters.
        model (tf.keras.Model or TFLite Interpreter or ONNX InferenceSession): Model to evaluate.
        num_classes (int, optional): Number of classes in the dataset.

    Returns:
        None
    """
    output_dir = HydraConfig.get().runtime.output_dir
    cpp = cfg.postprocessing
    class_names=cfg.dataset.class_names
    num_classes = len(class_names) if class_names else None
#    model_type=cfg.model.model_type
    print("Metrics calculation parameters:")
    print("  confidence threshold:", cpp.confidence_thresh)
    print("  NMS IoU threshold:", cpp.NMS_thresh)
    print("  max detection boxes:", cpp.max_detection_boxes)
    print("  metrics IoU threshold:", cpp.IoU_eval_thresh)

    model_path = getattr(model, 'model_path', None)
#    model_type = "float" if Path(model_path).suffix in ['.h5','.keras'] else "quantized"
    start_time = timer()
    if isinstance(model, tf.keras.Model):
        count_h5_parameters(output_dir=output_dir, model=model)  # model_path not needed
        metrics = evaluate_keras_model(cfg, model, num_classes=num_classes)
        model_type = "float"
    elif 'Interpreter' in str(type(model)):
        metrics = _evaluate_tflite_quantized_model(cfg, model, num_classes=num_classes, output_dir=output_dir)
        model_type = "quantized"
    elif isinstance(model, onnxruntime.InferenceSession):
        model_type = "quantized" if model_is_quantized(model_path) else "float"
        metrics = _evaluate_onnx_model(cfg, model, num_classes=num_classes)
    else:
        raise RuntimeError(
            "Evaluation internal error: unsupported model type "
            f"{type(model)}. Please provide a tf.keras.Model, a TFLite Interpreter, or an ONNX InferenceSession.")

    end_time = timer()
    eval_run_time = int(end_time - start_time)
    print("Evaluation run time: " + str(timedelta(seconds=eval_run_time)))

    _display_objdet_metrics(metrics, class_names)            
    # Log metrics in the stm32ai_main.log file
    log_to_file(output_dir, f"{model_type} model dataset used: {cfg.dataset.dataset_name}")
    mpre, mrec, mAP = calculate_average_metrics(metrics)
    log_to_file(output_dir, "{}_model_mpre: {:.1f}".format(model_type, 100 * mpre))
    log_to_file(output_dir, "{}_model_mrec: {:.1f}".format(model_type, 100 * mrec))
    log_to_file(output_dir, "{}_model_map: {:.1f}".format(model_type, 100 * mAP))
    
    # Log metrics in mlflow
    mlflow.log_metric(f"{model_type}_model_mpre", round(100 * mpre, 2))
    mlflow.log_metric(f"{model_type}_model_mrec", round(100 * mrec, 2))
    mlflow.log_metric(f"{model_type}_model_mAP", round(100 * mAP, 2))

    if cfg.postprocessing.plot_metrics:
        print("\nPlotting precision versus recall curves")
        plots_dir = os.path.join(output_dir, "precision_vs_recall_curves", os.path.basename(model_path))
        print("Plots directory:", plots_dir)
        
        output_dir = HydraConfig.get().runtime.output_dir
        _plot_precision_versus_recall(metrics, class_names, plots_dir)

