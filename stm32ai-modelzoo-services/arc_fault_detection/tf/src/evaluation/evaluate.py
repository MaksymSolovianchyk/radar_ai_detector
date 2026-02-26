# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import warnings
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import onnx
import onnxruntime
from sklearn.metrics import accuracy_score, confusion_matrix
from common.utils.visualize_utils import plot_confusion_matrix
from common.evaluation import model_is_quantized, predict_onnx

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from typing import Optional, Tuple

from common.utils import tf_dataset_to_np_array,\
                         count_h5_parameters, plot_confusion_matrix, log_to_file

def _compute_confusion_matrix(test_set: tf.data.Dataset = None, model: tf.keras.models.Model = None) -> Tuple[np.ndarray, np.float32]:
    """
    Computes the confusion matrix and logs it as an image summary.

    Args:
        test_set (tf.data.Dataset): The test dataset to evaluate the model on.
        model (tf.keras.models.Model): The trained model to evaluate.
    Returns:
        confusion_matrix and accuracy
    """
    test_pred = []
    test_labels = []
    for data in test_set:
        test_pred_score = model.predict_on_batch(data[0])
        test_pred.append(np.argmax(test_pred_score, axis=-1))
        batch_labels = data[1]
        test_labels.append(batch_labels)

    labels = np.concatenate(test_labels, axis=0).flatten()
    logits = np.concatenate(test_pred, axis=0).flatten()
    acc_score = round(accuracy_score(labels, logits) * 100 , 2)
    # Calculate the confusion matrix.
    cm = confusion_matrix(labels, logits)
    return cm, acc_score

def _quantize_input(x: np.ndarray, input_details: dict) -> np.ndarray:
    if input_details["dtype"] in (np.int8, np.uint8):
        scale, zp = input_details["quantization"]
        x_q = (x / scale + zp).round()
        info = np.iinfo(input_details["dtype"])
        x_q = np.clip(x_q, info.min, info.max).astype(input_details["dtype"])
        return x_q
    return x.astype(input_details["dtype"])

def _dequantize_output(raw: np.ndarray, output_details: dict) -> np.ndarray:
    """Convert int8/uint8 quantized output back to float probabilities."""
    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zp = output_details["quantization"]
        if scale and scale > 0:
            return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)

def _sanitize_onnx_opset_imports(onnx_model_path: str,
                                target_opset: int):
    '''
    remove all the un-necessary opset imports from an onnx model resulting due to tf2onnx opperation
    Inputs
    ------
    input_model_path : str 
        Path to the model file which has to be cleaned
    target_opset : int, the target onnx opset '''
    onnx_model = onnx.load(onnx_model_path)
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ''
    opset.version = target_opset
    onnx.save(onnx_model, onnx_model_path)

def evaluate_h5_model(model_path: str = None,
                      eval_ds: tf.data.Dataset = None,
                      class_names: list = None,
                      output_dir: str = None,
                      name_ds: Optional[str] = 'test_set') -> float:
    """
    Evaluates a trained Keras model saved in .h5 format on the provided dataset.

    Args:
        model_path (str): The file path to the .h5 model.
        eval_ds (tf.data.Dataset): Dataset to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.

    Returns:
        float: The accuracy of the provided model on eval_ds

    """

    # Load the .h5 model
    model = tf.keras.models.load_model(model_path)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss, metrics=['accuracy'])
    # Evaluate the model on the test data
    tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
    loss, accuracy = model.evaluate(eval_ds)

    # Calculate the confusion matrix.
    cm, test_accuracy = _compute_confusion_matrix(test_set=eval_ds, model=model)
    # Log the confusion matrix as an image summary.
    model_name = f"float_model_confusion_matrix_{name_ds}"
    plot_confusion_matrix(cm=cm, class_names=class_names, model_name=model_name,
                          title=f'{model_name}\naccuracy: {test_accuracy}', output_dir=output_dir)
    print(f"[INFO] : Accuracy of float model = {test_accuracy}%")
    print(f"[INFO] : Loss of float model = {loss}")
    mlflow.log_metric(f"float_acc_{name_ds}", test_accuracy)
    mlflow.log_metric(f"float_loss_{name_ds}", loss)
    log_to_file(output_dir, f"Float model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of float model : {test_accuracy} %")
    log_to_file(output_dir, f"Loss of float model : {round(loss,2)} ")

    return accuracy

def _evaluate_tflite_quantized_model(cfg: DictConfig = None,
                                    quantized_model_path: str = None, eval_ds: tf.data.Dataset = None,
                                    class_names: list = None,
                                    output_dir: str = None, name_ds: Optional[str] = 'test_set',
                                    num_threads: Optional[int] = 1):
    """
    Evaluates the accuracy of a quantized TensorFlow Lite model using tflite.interpreter and plots the confusion matrix.

    Args:
        cfg (config): The configuration file.
        quantized_model_path (str): The file path to the quantized TensorFlow Lite model.
        eval_ds (tf.data.Dataset): The test dataset to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
        num_threads (int): number of threads for the tflite interpreter
    Returns:
        float: The accuracy of the provided model on eval_ds
    """
        
    tf.print(f'[INFO] : Evaluating the quantized model using {name_ds}...')

    features = []
    labels = []

    for x, y in eval_ds.as_numpy_iterator():
        features.append(x)
        labels.append(y)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()
  
    interpreter = tf.lite.Interpreter(model_path=quantized_model_path, num_threads=num_threads)

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    expected_shape = list(input_detail["shape"])
    expected_shape[0] = features.shape[0]
    # Get shape of a batch
    interpreter.resize_tensor_input(input_detail["index"], expected_shape)
    interpreter.allocate_tensors()

    tf.print(f"[INFO] : Quantization input details : {input_detail['quantization']}")
    tf.print(f"[INFO] : Dtype input details : {input_detail['dtype']}")

    X_proc = _quantize_input(features, input_detail)
    interpreter.set_tensor(input_detail["index"], X_proc)
    interpreter.invoke()
    raw_out = interpreter.get_tensor(output_detail["index"])
    probs = _dequantize_output(raw_out, output_detail)

    pred_idx = np.argmax(probs, axis=-1).flatten()

    # Compute the accuracy
    acc_score = round(accuracy_score(labels, pred_idx) * 100 , 2)

    # Print metrics & log in MLFlow
    print(f"[INFO] : Accuracy of quantized model = {acc_score}%")
    mlflow.log_metric(f"quant_acc_{name_ds}", acc_score)

    log_to_file(output_dir,  "" + f"Quantized model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of quantized model : {acc_score} %")
    # Compute and plot the confusion matrices
    
    cm = confusion_matrix(labels, pred_idx)

    confusion_matrix_title = ("Quantized model confusion matrix \n"
                             f"On dataset : {name_ds} \n"
                             f"Quantized model accuracy : {np.round(acc_score * 100, decimals=2)}")

    plot_confusion_matrix(cm=cm,
                            class_names=class_names,
                            title=confusion_matrix_title,
                            model_name=f"quant_model_confusion_matrix_{name_ds}",
                            output_dir=output_dir)

    return acc_score

def _evaluate_onnx_model(input_samples:np.ndarray,
                            input_labels:np.ndarray,
                            input_model_path:str,
                            class_labels:list,
                            output_dir:str,
                            name_ds:str):
    """
    Evaluates an ONNX model on a validation dataset.

    Args:
        input_samples (np.ndarray): data to run evaluation on.
        input_labels (np.ndarray): labels for the evaluation samples.
        input_model_path (str): The path to the onnx model to be evaluated.
        class_labels (List[str]): The list of the class labels.
        output_dir(str): The name of the output directory where confusion matrix and the logs are to be saved.
        name_ds (str) : Name of the dataset.
    Returns:
        accuracy : float, accuracy of the provided model on eval_ds.
    """

    # fixing the opset of the input model
    _sanitize_onnx_opset_imports(onnx_model_path = input_model_path,
                                target_opset = 17)

    sess = onnxruntime.InferenceSession(input_model_path)
    model_type = 'Quantized' if model_is_quantized(input_model_path) else 'Float'
    # Evaluate the model on the input data
    preds = predict_onnx(sess, input_samples)
    y_pred = np.argmax(preds, axis=-1).flatten()  
    # Compute the accuracy
    input_labels = input_labels.flatten() 
    eval_accuracy = round(accuracy_score(input_labels, y_pred) * 100, 2)
    print(f'[INFO] : {model_type} accuracy: {eval_accuracy} %')
    log_file_name = f"{output_dir}/stm32ai_main.log"
    with open(log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'{model_type} ONNX model\n accuracy: {eval_accuracy} %\n')
    acc_metric_name = f"quant_acc_{name_ds}" if model_is_quantized(input_model_path) else f"float_acc_{name_ds}"
    mlflow.log_metric(acc_metric_name, eval_accuracy)
    # Calculate the confusion matrix.
    cm = confusion_matrix(input_labels, y_pred)
    # Log the confusion matrix as an image summary.
    confusion_matrix_title = (f"{model_type} confusion matrix \n"
                               f"On dataset : {name_ds} \n"
                               f"accuracy : {eval_accuracy}")
    plot_confusion_matrix(cm=cm,
                            class_names=class_labels,
                            title=confusion_matrix_title,
                            model_name=f"{model_type.lower()}_confusion_matrix_{name_ds}",
                            output_dir=output_dir)

    return eval_accuracy

def evaluate(cfg: DictConfig = None, model_to_evaluate = None, dataloaders: dict = None) -> None:
    """
    Evaluates and benchmarks a TensorFlow Lite or Keras model, and generates a Config header file if specified.

    Args:
        cfg (config): The configuration file.
        model_to_evaluate: The model object to be evaluated.
        dataloaders (dict): A dictionary containing the datasets for evaluation.

    Returns:
        None
    """
    output_dir = HydraConfig.get().runtime.output_dir
    class_names = cfg.dataset.class_names
    model_path = model_to_evaluate.model_path
    name_ds = 'test_set' if dataloaders.get('test') is not None else 'valid_set'
    eval_ds = dataloaders[name_ds[:-4]]  # removing '_set' suffix to get the key in dataloaders
    try:
        file_extension = Path(model_path).suffix
        if file_extension == '.tflite':
            # Evaluate quantized TensorFlow Lite model
            _evaluate_tflite_quantized_model(cfg=cfg, 
                                            quantized_model_path=model_path,
                                            eval_ds=eval_ds,
                                            class_names=class_names,
                                            output_dir=output_dir,
                                            name_ds=name_ds,
                                            num_threads=cfg.general.num_threads_tflite)

        # Check if the model is a Keras model
        elif file_extension in ['.h5', '.keras']:
            count_h5_parameters(output_dir=output_dir, 
                                model=model_to_evaluate)
            # Evaluate Keras model
            evaluate_h5_model(model_path=model_path,
                            eval_ds=eval_ds,
                            class_names=class_names,
                            output_dir=output_dir,
                            name_ds=name_ds)
        elif file_extension == '.onnx':
            data, labels = tf_dataset_to_np_array(eval_ds, nchw=False)
            _evaluate_onnx_model(input_samples=data,
                                    input_labels=labels,
                                    input_model_path=model_path,
                                    class_labels=class_names,
                                    output_dir=output_dir,
                                    name_ds=name_ds)

    except Exception as e:
        raise ValueError(f"Model evaluation failed\n Received model path: {model_path}. Exception was {e}")
