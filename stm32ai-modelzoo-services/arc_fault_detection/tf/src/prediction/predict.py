# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tabulate import tabulate
import onnx
import onnxruntime
from common.evaluation import predict_onnx, model_is_quantized

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
    Remove all the un-necessary opset imports from an onnx model resulting due to tf2onnx operation.
    Inputs
    ------
    onnx_model_path : str 
        Path to the model file which has to be cleaned
    target_opset : int, the target onnx opset '''
    onnx_model = onnx.load(onnx_model_path)
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ''
    opset.version = target_opset
    onnx.save(onnx_model, onnx_model_path)


def predict(cfg, dataloaders) -> None:
    """
    Predicts the class for the given input data using the specified model.
    Args:
        cfg: Configuration object containing model and dataset info.
        dataloaders: Dictionary containing the 'predict' dataset (numpy array or tf.data.Dataset).
    Returns:
        None. Prints a table of predictions.
    """
    X = dataloaders['predict']
    model_path = cfg.model.model_path
    class_names = list(cfg.dataset.class_names)
    ext = Path(model_path).suffix.lower()

    if ext in [".h5", ".keras"]:
        model = tf.keras.models.load_model(model_path)
        probs = model.predict(X, verbose=0)
    elif ext == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        expected_shape = list(input_detail["shape"])
        expected_shape[0] = X.shape[0]
        interpreter.resize_tensor_input(input_detail["index"], expected_shape)
        interpreter.allocate_tensors()
        X_proc = _quantize_input(X, input_detail)
        interpreter.set_tensor(input_detail["index"], X_proc)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_detail["index"])
        probs = _dequantize_output(raw_out, output_detail)
    elif ext == '.onnx':
        # Fixing the opset of the input model
        _sanitize_onnx_opset_imports(onnx_model_path=model_path, target_opset=17)
        sess = onnxruntime.InferenceSession(model_path)
        probs = predict_onnx(sess, X)
    else:
        raise TypeError(f"Unsupported model extension: {ext}")

    # Handle multi-channel output: probs shape (n_samples, n_channels, n_classes)
    n_samples, n_channels, n_classes = probs.shape

    # Build the cell content for each sample
    body = []
    for i in range(n_samples):
        row = []
        for ch in range(n_channels):
            ch_probs = probs[i, ch]
            idx = int(np.argmax(ch_probs))
            one_hot = [1 if c == idx else 0 for c in range(n_classes)]
            row.extend([
                class_names[idx] if idx < len(class_names) else str(idx),
                one_hot,
                np.round(ch_probs, 2),
            ])
        body.append(row)

    # Now build a **tabulate‑compatible** table: prepend header rows as data rows
    # Row 0: channel labels (Channel 1, "", "", Channel 2, "", "", ...)
    # Row 1: types (Prediction, One-hot, Scores, Prediction, One-hot, Scores, ...)
    header_channel = []
    header_type = []
    for ch in range(n_channels):
        header_channel.extend(["", f"Channel {ch+1}", ""])
        header_type.extend(["Prediction", "One-hot", "Scores"])

    # Combine headers + body into one table
    # Tabulate will treat everything as rows
    full_table = [header_channel, header_type] + body

    print(tabulate(full_table, tablefmt="grid", showindex=False))
