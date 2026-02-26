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

# Import utility functions
from image_classification.tf.src.preprocessing import postprocess_output, preprocess_input
from image_classification.tf.src.utils import ai_runner_invoke
from common.utils import (
    ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant,
    plot_confusion_matrix, log_to_file, display_figures
)  # Common utilities for evaluation and visualization


# Define a class for evaluating TFLite quantized models
class TFLiteQuantizedModelEvaluator:
    """
    A class to evaluate TensorFlow Lite (TFLite) quantized models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The quantized TFLite model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig, model: object, 
                 dataloaders: dict = None):
        self.cfg = cfg
        self.quantized_model = model
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.class_names = cfg.dataset.class_names
        self.display_figures = cfg.general.display_figures
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

    def _get_target(self):
        """
        Retrieves the evaluation target from the configuration.
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        Args: 
            target (str): target on which we intend to evaluate
        Returns:
            ai runner interpreter correctly parametrized
        """
        name_model = os.path.basename(self.quantized_model.model_path)
        return ai_runner_interp(target, name_model)

    def _run_evaluate(self):
        """
        Runs the evaluation process and computes metrics.

        Returns:
            float: Accuracy of the quantized model on the evaluation dataset.
        """
        tf.print(f'[INFO] : Evaluating the quantized model using {self.name_ds}...')
        target = self._get_target()     # Get the evaluation target
        ai_runner_interpreter = self._get_interpreter(target=target)    # Get the AI runner interpreter
        interpreter_quant = self.quantized_model    # Quantized TFLite model
        input_details = interpreter_quant.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_index_quant = interpreter_quant.get_output_details()[0]["index"]
        output_details = interpreter_quant.get_output_details()[0]
        predictions_all = []    # Placeholder for all predictions
        test_pred = []      # Placeholder for predicted labels
        test_labels = []    # Placeholder for ground truth labels
        images_full = []    # Placeholder for processed input images

        # Iterate over the evaluation dataset
        for images, labels in tqdm.tqdm(self.eval_ds, total=len(self.eval_ds)):
            for image, label in zip(images, labels):
                # Preprocess the input image
                image_processed = preprocess_input(image, input_details)
                if "evaluation" in self.cfg and self.cfg.evaluation:
                    if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input == True:
                        images_full.append(image_processed)
                
                # Perform inferences
                if target == 'host':
                    interpreter_quant.set_tensor(input_index_quant, image_processed)
                    interpreter_quant.invoke()
                    test_pred_score = interpreter_quant.get_tensor(output_index_quant)
                elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                    image_preproc = ai_interp_input_quant(ai_runner_interpreter, image[None].numpy(), '.tflite')
                    test_pred_score = ai_runner_invoke(image_preproc, ai_runner_interpreter)
                    test_pred_score = ai_interp_outputs_dequant(ai_runner_interpreter, [test_pred_score])[0]
                    test_pred_score = np.reshape(test_pred_score, [1, -1])

                # Save predictions if configured
                if "evaluation" in self.cfg and self.cfg.evaluation:
                    if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output == True:
                        predictions_all.append(test_pred_score)

                # Postprocess the output and store predictions
                predicted_label = postprocess_output(test_pred_score, output_details)
                test_pred.append(predicted_label)
                test_labels.append(label.numpy())

        # Save evaluation dataset in a .npy file if configured
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input == True:
                npy_in_name = getattr(self.cfg.evaluation, "npy_in_name", "unknown_npy_in_name")
                images_full = np.concatenate(images_full, axis=0)
                print("[INFO] : Shape of npy input dataset = {}".format(images_full.shape))
                np.save(os.path.join(self.output_dir, f"{npy_in_name}.npy"), images_full)

        # Save model output in a .npy file if configured
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output == True:
                npy_out_name = getattr(self.cfg.evaluation, "npy_out_name", "unknown_npy_out_name")
                predictions_all = np.concatenate(predictions_all, axis=0)
                print("[INFO] : Shape of npy predicted scores = {}".format(predictions_all.shape))
                np.save(os.path.join(self.output_dir, f"{npy_out_name}.npy"), predictions_all)

        # Compute the confusion matrix and accuracy
        labels = np.array(test_labels)
        logits = np.concatenate(test_pred, axis=0)
        logits = np.squeeze(logits)
        cm = sklearn.metrics.confusion_matrix(labels, logits)
        accuracy = round((np.sum(labels == logits) * 100) / len(test_labels), 2)

        # Log evaluation results
        print(f"[INFO] : Accuracy of quantized model on {self.name_ds} = {accuracy}%")
        log_to_file(self.output_dir,  f"Quantized model {self.name_ds}:")
        log_to_file(self.output_dir, f"Accuracy of quantized model : {accuracy} %")
        mlflow.log_metric(f"int_acc_{self.name_ds}", accuracy)

        # Plot and display the confusion matrix if enabled
        if self.display_figures:
            model_name = f"quantized_model_confusion_matrix_{self.name_ds}"
            plot_confusion_matrix(cm, class_names=self.class_names, model_name=model_name,
                                  title=f'{model_name}\naccuracy: {accuracy}', output_dir=self.output_dir)
            display_figures(self.cfg)

        return accuracy

    def evaluate(self):
        """
        Executes the full evaluation process.

        Returns:
            float: Accuracy of the quantized model on the evaluation dataset.
        """
        self._prepare_evaluation()      # Prepare the evaluation process
        acc = self._run_evaluate()      # Run the evaluation
        print('[INFO] : Evaluation complete.')
        return acc  # Return the accuracy
    

