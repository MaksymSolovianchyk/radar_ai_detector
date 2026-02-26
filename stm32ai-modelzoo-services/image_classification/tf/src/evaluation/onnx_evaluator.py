# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
import sys
from pathlib import Path
import warnings
import sklearn
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict
import numpy as np

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import onnxruntime
import tensorflow as tf
import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Import utility functions
from image_classification.tf.src.utils import ai_runner_invoke
from common.evaluation import model_is_quantized, predict_onnx_batch
from common.utils import (
    tf_dataset_to_np_array, display_figures, ai_runner_interp, ai_interp_input_quant,
    ai_interp_outputs_dequant, plot_confusion_matrix, torch_dataset_to_np_array
)  # Common utilities for evaluation and visualization


# Define a class for evaluating ONNX models
class ONNXModelEvaluator:
    """
    A class to evaluate ONNX models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The ONNX model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig, model: object, 
                 dataloaders: dict = None):
        self.cfg = cfg
        self.input_model = model
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.class_names = cfg.dataset.class_names
        self.display_figures = cfg.general.display_figures
        input_chpos = getattr(cfg.evaluation, 'input_chpos', 'chlast') if hasattr(cfg, 'evaluation') else 'chlast'
        if self.cfg.model.framework == "tf":
            # Dataloader is channel last with TF
            if input_chpos=="chfirst" or self._get_target() == 'host':
                self.nchw = True
            else:
                self.nchw = False
        else:
            # Dataloader is already channel first with Torch
            if input_chpos=="chfirst":
                self.nchw = False
            else:
                self.nchw = True
        
        self.eval_ds = None
        self.name_ds = None

    def _prepare_evaluation(self):
        """
        Prepares the evaluation process by selecting the appropriate dataset.
        """
        # Use the test dataset if available; otherwise, use the validation dataset
        if self.test_ds:
            self.eval_ds = self.test_ds
            self.name_ds = "test_set"
        else:
            self.eval_ds = self.valid_ds
            self.name_ds = "validation_set"

    def _ensure_output_dir(self):
        """
        Ensures that the output directory exists.
        """
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_target(self):
        """
        Retrieves the evaluation target from the configuration.
        Returns:
            str: the target on which evaluation will be done, by default 'host'
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_model_type(self):
        """
        Determines whether the model is quantized or float.
        Returns:
            str: 'quantized' or 'float' depending on model quantization status
        """
        return 'quantized' if model_is_quantized(self.input_model.model_path) else 'float'

    def _get_ai_runner_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        Args:
            target (str): target on which evaluation is to be performed
        Returns: ai_runner interpreter correctly parametrized
        """
        name_model = os.path.basename(self.input_model.model_path)
        return ai_runner_interp(target, name_model)

    def _predict(self, ai_runner_interpreter, model_type, target):
        """
        Runs predictions on the evaluation dataset.

        Args:
            ai_runner_interpreter: AI runner interpreter for inference.
            model_type (str): Type of the model ('float' or 'quantized').
            target (str): Evaluation target.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and input labels.
        """
        if model_type == 'float' or target == 'host':
            # Use ONNX runtime for predictions
            prd_labels, input_labels = predict_onnx_batch(sess=self.input_model, 
                                                          data=self.eval_ds,
                                                          nchw=self.nchw)
            prd_labels = prd_labels.argmax(axis=1)
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p'] and model_type == 'quantized':
            # Use AI runner for predictions
            prd_labels = []
            if self.cfg.model.framework == "tf":
                # Convert the tf dataset to NumPy array as dataloader was based on TF framework
                input_samples, input_labels = tf_dataset_to_np_array(input_ds=self.eval_ds, 
                                                                     nchw=self.nchw)
            else: #if self.cfg.model.framework == "torch":
                input_samples, input_labels = torch_dataset_to_np_array(input_loader=self.eval_ds,
                                                                        nchw=self.nchw)

            for i in tqdm.tqdm(range(input_samples.shape[0])):
                data = ai_interp_input_quant(ai_runner_interpreter, input_samples[i][None], '.onnx')
                prd_label = ai_runner_invoke(data, ai_runner_interpreter)
                prd_label = ai_interp_outputs_dequant(ai_runner_interpreter, [prd_label])[0]
                prd_label = prd_label.argmax(axis=1)
                prd_labels.append(prd_label)
            prd_labels = np.array(prd_labels, dtype=np.float32)
        else:
            raise TypeError("Only supported targets are \"host\", \"stedgeai_host\" or \"stedgeai_n6\". "
                            "Check the \"evaluation\" section of your configuration file.")
        return prd_labels, input_labels

    def _run_evaluate(self):
        """
        Runs the evaluation process and computes metrics.

        Returns:
            Tuple[float, np.ndarray]: Accuracy and confusion matrix.
        """
        self._ensure_output_dir()       # Ensure the output directory exists
        target = self._get_target()     # Get the evaluation target
        model_type = self._get_model_type() # Determine the model type
        ai_runner_interpreter = self._get_ai_runner_interpreter(target=target)   # Get the AI runner interpreter

        # Run predictions
        prd_labels, input_labels = self._predict(ai_runner_interpreter, model_type, target)
        accuracy = round(accuracy_score(input_labels, prd_labels) * 100, 2)
        print(f'[INFO] : Evaluation accuracy on {self.name_ds}: {accuracy} %')

        # Log evaluation results to a file
        log_file_name = f"{self.output_dir}/stm32ai_main.log"
        with open(log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'{model_type} onnx model\nEvaluation accuracy: {accuracy} %\n')

        # Compute the confusion matrix
        cm = confusion_matrix(input_labels, prd_labels)
        acc_metric_name = f"int_acc_{self.name_ds}" if model_type == 'quantized' else f"float_acc_{self.name_ds}"
        mlflow.log_metric(acc_metric_name, accuracy)

        # Plot and display the confusion matrix if enabled
        if self.display_figures:
            model_name = f'{model_type}_onnx_model_{self.name_ds}'
            plot_confusion_matrix(
                cm=cm,
                class_names=self.class_names,
                model_name=model_name,
                title=f'{model_name}\naccuracy: {accuracy}',
                output_dir=self.output_dir
            )
            display_figures(self.cfg)
        return accuracy, cm

    def evaluate(self):
        """
        Executes the full evaluation process.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        self._prepare_evaluation()      # Prepare the evaluation process
        acc, cm = self._run_evaluate()  # Run the evaluation
        print('[INFO] : Evaluation complete.')
        return acc   # Return the accuracy

