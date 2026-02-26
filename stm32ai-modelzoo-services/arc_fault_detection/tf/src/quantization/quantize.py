# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
import numpy as np
import tensorflow as tf
import tqdm
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Optional
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from common.optimization import model_formatting_ptq_per_tensor
import onnx
import onnxruntime

from common.evaluation import model_is_quantized
from common.onnx_utils import onnx_model_converter

def _representative_data_gen(configs: DictConfig, quantization_ds: Optional[tf.data.Dataset] = None, quantization_split: float = 1.0):
    """
    Generates representative data samples for post-training quantization.
    This generator yields input data samples, either randomly generated or from a provided dataset,
    to be used during quantization calibration.
    Args:
        configs (DictConfig): Configuration object containing model parameters, including input shape.
        quantization_ds (Optional[tf.data.Dataset]): Dataset to use for representative data. If None, random data is generated.
        quantization_split (float): Fraction of the dataset to use for quantization. If 1.0, uses the entire dataset.
            If 0.0, uses dummy random data.
        np.ndarray: A numpy array representing a single input sample, shaped according to the model's input.
    """
    if quantization_ds is None or quantization_split == 0.0:
        print("[INFO] : Quantizing using dummy data")
        if configs.quantization.quantizer == "TFlite_converter":
            for _ in tqdm.tqdm(range(5)):
                data = np.random.rand(1,*configs.model.input_shape)
                yield [data.astype(np.float32)]
        else:
            for _ in tqdm.tqdm(range(5)):
                data = np.random.rand(1,*configs.model.input_shape)
                yield data.astype(np.float32)

    else:
        if not quantization_split or quantization_split == 1.0:
            print("[INFO] : Quantizing by using the provided dataset fully, this will take a while.")
            print(f"[INFO] : Using {len(quantization_ds)} patches")
            if configs.quantization.quantizer == "TFlite_converter":
                for patches, _ in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
                    for patch in patches:
                        yield [tf.cast(patch[np.newaxis, ...], tf.float32)]
            else:
                for patches, _ in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
                        for patch in patches:
                            yield tf.cast(patch[np.newaxis, ...], tf.float32).numpy()
        else:
            print(f'[INFO] : Quantizing by using {quantization_split * 100} % of the provided dataset...')
            split_ds = quantization_ds.take(int(len(quantization_ds) * float(quantization_split)))
            if configs.quantization.quantizer == "TFlite_converter":
                for patches, _ in tqdm.tqdm(split_ds, total=len(split_ds)):
                    for patch in patches:
                        yield [tf.cast(patch[np.newaxis, ...], tf.float32)]
            else:
                for patches, _ in tqdm.tqdm(split_ds, total=len(split_ds)):
                    for patch in patches:
                        yield tf.cast(patch[np.newaxis, ...], tf.float32).numpy()

def quantize_onnx(configs: DictConfig, model_path: str = None, quantization_ds=None):
    """
    Quantizes an ONNX model using onnx-runtime.

    Args:
        configs (DictConfig): Configuration dictionary containing quantization and model settings.
        model_path (str, optional): Path to the ONNX model file.
        quantization_ds: Calibration/representative dataset as a numpy array (optional).

    Returns:
        onnxruntime.InferenceSession: Quantized model session with model_path attribute.
    """

    # Create the output directory (like in your TFLite logic)
    output_dir = HydraConfig.get().runtime.output_dir   
    onnx_models_dir = Path(output_dir) / configs.quantization.export_dir
    onnx_models_dir.mkdir(exist_ok=True, parents=True)
    quantized_model_path = onnx_models_dir / "quantized_model.onnx"

    # Define a CalibrationDataReader
    class NumpyDataReader(CalibrationDataReader):
        def __init__(self, configs, quantization_ds, quantization_split, model_path):
            # Create the generator inside the class
            self.enum_data = iter(
                _representative_data_gen(
                    configs=configs,
                    quantization_ds=quantization_ds,
                    quantization_split=quantization_split
                )
            )
            self.input_name = onnx.load(model_path).graph.input[0].name

        def get_next(self):
            try:
                return {self.input_name: next(self.enum_data)}  
            except StopIteration:
                return None

    # Usage:
    quantization_split = configs.dataset.quantization_split if configs.dataset.quantization_split is not None else 1.0
    data_reader = NumpyDataReader(
        configs=configs,
        quantization_ds=quantization_ds,
        quantization_split=quantization_split,
        model_path=model_path
    )
    # Perform static quantization
    quantize_static(
        model_input=model_path,
        model_output=str(quantized_model_path),
        calibration_data_reader=data_reader,
        per_channel=(configs.quantization.granularity == "per_channel"),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    # Load the modified model using ONNX Runtime Check if the model is valid
    model = onnxruntime.InferenceSession(quantized_model_path)
    try:
        model.get_inputs()
    except Exception as e:
        print(f"[ERROR] : An error occurred while quantizing the model: {e}")
        return
    setattr(model, 'model_path', quantized_model_path)
    print(f"[INFO] : Quantized model saved at {quantized_model_path}")
    return model


def _tflite_ptq_quantizer(configs: DictConfig = None, model: tf.keras.Model = None,
                            quantization_ds: tf.data.Dataset = None) -> tf.lite.Interpreter:
    """
    Perform post-training quantization on a TensorFlow Lite model.

    Args:
        configs (DictConfig): Configuration dictionary containing quantization and model settings.  
        model (tf.keras.Model): The TensorFlow model to be quantized.
        quantization_ds (tf.data.Dataset): The quantization dataset if it's provided by the user else the training dataset. Defaults to None
    Returns:
        tf.lite.Interpreter: The quantized TFLite model as an Interpreter object.
    """
    # Create the output directory
    output_dir = HydraConfig.get().runtime.output_dir
    tflite_models_dir = Path(output_dir) / configs.quantization.export_dir
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_quantized_file = tflite_models_dir / "quantized_model.tflite"

    # Create the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set the quantization types for the input and output
    if configs.quantization_input_type == 'int8':
        converter.inference_input_type = tf.int8
    elif configs.quantization_input_type == 'uint8':
        converter.inference_input_type = tf.uint8

    if configs.quantization_output_type == 'int8':
        converter.inference_output_type = tf.int8
    elif configs.quantization_output_type == 'uint8':
        converter.inference_output_type = tf.uint8

    # Set the optimizations and representative dataset generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantization_split = configs.dataset.quantization_split if configs.dataset.quantization_split is not None else 1.0
    converter.representative_dataset = lambda: _representative_data_gen(configs=configs, quantization_ds=quantization_ds,
                                                               quantization_split=quantization_split)
    
    if configs.quantization_granularity == 'per_tensor':
        converter._experimental_disable_per_channel = True

    # Convert the model to a quantized TFLite model
    tflite_model_quantized = converter.convert()
    tflite_model_quantized_file.write_bytes(tflite_model_quantized)

    # Return the quantized TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quantized_file))
    interpreter.allocate_tensors()
    setattr(interpreter, 'model_path', str(tflite_model_quantized_file))
    print(f"[INFO] : Quantized model saved at {tflite_model_quantized_file}")
    return interpreter

def quantize(cfg: DictConfig = None, float_model: object = None, dataloaders: dict = None):
    """
    Quantize a model using ONNX or TFLite post-training quantization, based on configuration.

    Args:
        cfg (DictConfig): The configuration dictionary.
        float_model: TensorFlow Keras model or ONNX InferenceSession.
        dataloaders: Dictionary containing datasets for training, validation, testing, quantization and prediction.

    Returns:
        Quantized model object (ONNX InferenceSession or TFLite Interpreter) with 'model_path' attribute.
    """

    model_path = float_model.model_path
    output_dir = HydraConfig.get().runtime.output_dir
    file_extension = Path(model_path).suffix
     # check if the batch dimension is included in the input shape and remove it if present
    if len (cfg.model.input_shape) == 4 : setattr(cfg.model, 'input_shape', cfg.model.input_shape[1:])
    if dataloaders.get('quantization') is not None:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantization_ds = dataloaders['quantization']
    elif dataloaders.get('train') is not None:
        quantization_ds = dataloaders['train']
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
    else:
        quantization_ds = None
        print('[INFO] : Neither quantization dataset nor training set are provided! Using fake data to quantize the model. '
              'The model performances will not be accurate.')

    if cfg.quantization.quantizer.lower() == "onnx_quantizer" and cfg.quantization.quantization_type == "PTQ":
        if file_extension in [".h5",".keras"]:
            # Convert the model to ONNX first
            input_shape = float_model.input_shape  # include batch dimension
            print(f"Converting model to ONNX, with static input shape {input_shape}")
            converted_model_path = os.path.join(output_dir, 'converted_model', 'converted_model.onnx')
            target_opset = cfg.quantization.target_opset if cfg.quantization.target_opset else 17
            onnx_model_converter(input_model_path=model_path, target_opset=target_opset, 
                                 output_dir=converted_model_path, static_input_shape=input_shape,
                                 input_channels_last=True)
            quantized_model = quantize_onnx(configs=cfg, quantization_ds=quantization_ds, model_path=converted_model_path)
            return quantized_model
        
        elif file_extension == '.onnx':
            if model_is_quantized(model_path):
                print('[INFO] : The input model is already quantized!\n\tReturning the same model!')
                return float_model
            quantized_model = quantize_onnx(configs=cfg, quantization_ds=quantization_ds, model_path=model_path)
            return quantized_model
        else:
            raise ValueError("Unsupported model format for ONNX quantization. Supported formats are .h5, .keras, and .onnx")
    else:
        if cfg.quantization.quantizer == "TFlite_converter" and cfg.quantization.quantization_type == "PTQ":
            if file_extension not in  [".h5",".keras"]:
                raise ValueError("For TFLite quantization, the model format must be either .h5 or .keras.")
            float_model = tf.keras.models.load_model(model_path)            
            # if per-tensor quantization is required some optimizations are possible on the float model
            if cfg.quantization.granularity == 'per_tensor' and cfg.quantization.optimize:    
                print("[INFO] : Optimizing the model for improved per_tensor quantization...")
                float_model = model_formatting_ptq_per_tensor(model_origin=float_model)
                optimized_model_path = os.path.join(output_dir, cfg.quantization.export_dir, "optimized_model.keras")
                float_model.save(optimized_model_path)

            # Quantize the model
            quantized_model = _tflite_ptq_quantizer(configs=cfg, model=float_model, quantization_ds=quantization_ds)            
            return quantized_model
        else:
            raise NotImplementedError("Quantizer and quantization type not supported yet!")
