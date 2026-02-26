# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import warnings
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tqdm

from common.utils import (
    ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant, log_to_file) 
from semantic_segmentation.tf.src.evaluation import prediction_accuracy_on_batch, iou_per_class
from semantic_segmentation.tf.src.preprocessing import preprocess_input, postprocess_output_values
from semantic_segmentation.tf.src.utils import ai_runner_invoke


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
        self.quantized_model.allocate_tensors()
        input_details = self.quantized_model.get_input_details()[0]
        input_index = input_details["index"]
        output_details = self.quantized_model.get_output_details()[0]
        output_index = output_details["index"]

        accuracy_list = []
        iou_global_window = []
        num_classes = len(self.class_names)
        images_full = []
        predictions_all = []
        for (images, mask) in tqdm.tqdm(self.eval_ds, total=len(self.eval_ds)):
            for img, msk in zip(images, mask):
                if target == "host":
                    # Preprocess the image for applying quantization parameters
                    img = preprocess_input(img, input_details=input_details)
                    
                    if "evaluation" in self.cfg and self.cfg.evaluation:
                        if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input == True:
                            images_full.append(img)

                    # Run inference
                    self.quantized_model.set_tensor(input_index, img)
                    self.quantized_model.invoke()
                    out = self.quantized_model.get_tensor(output_index)[0]
                    out = postprocess_output_values(output=out, output_details=output_details)
                elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                    data = ai_interp_input_quant(ai_runner_interpreter,img.numpy()[None],
                                                '.tflite')
                    out  = ai_runner_invoke(data, ai_runner_interpreter)
                    out  = ai_interp_outputs_dequant(ai_runner_interpreter, out)[0][0]

                if "evaluation" in self.cfg and self.cfg.evaluation:
                    if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output == True:
                        predictions_all.append(out)

                pred_mask = np.argmax(out, axis=-1)
                true_mask = tf.squeeze(msk, axis=-1).numpy()
                accuracy = prediction_accuracy_on_batch(pred_mask, true_mask)
                accuracy_list.append(accuracy)

                ious_per_image = iou_per_class(pred_mask, true_mask, num_classes)
                if ious_per_image:
                    for iou in ious_per_image:
                        iou_global_window.append(iou)

        # Saves evaluation dataset in a .npy
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input == True:
                if "npy_in_name" in self.cfg.evaluation and self.cfg.evaluation.npy_in_name:
                    npy_in_name = self.cfg.evaluation.npy_in_name
                else:
                    npy_in_name = "unknown_npy_in_name"
                images_full = np.concatenate(images_full, axis=0)
                print("[INFO] : Shape of npy input dataset = {}".format(images_full.shape))
                np.save(os.path.join(self.output_dir, f"{npy_in_name}.npy"), images_full)

        # Saves model output in a .npy
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output == True:
                if "npy_out_name" in self.cfg.evaluation and self.cfg.evaluation.npy_out_name:
                    npy_out_name = self.cfg.evaluation.npy_out_name
                else:
                    npy_out_name = "unknown_npy_out_name"
                predictions_all = np.concatenate(predictions_all, axis=0)
                np.save(os.path.join(self.output_dir, f"{npy_out_name}.npy"), predictions_all)

        avg_accuracy = np.mean(accuracy_list)
        avg_iou = np.mean(iou_global_window) if iou_global_window else 0  # Handle case with no IoU scores

        print(f"[INFO] : Quantized model accuracy on {self.name_ds} = {round(avg_accuracy * 100, 2)}%")
        print(f"[INFO] : Quantized model average IoU on all classes on {self.name_ds} = {round(avg_iou * 100, 2)}%")
        mlflow.log_metric(f"quantized_acc_{self.name_ds}", round(avg_accuracy * 100, 2))
        mlflow.log_metric(f"quantized_avg_iou_{self.name_ds}", round(avg_iou * 100, 2))
        log_to_file(self.output_dir, f"Tflite quantized model {self.name_ds}:")
        log_to_file(self.output_dir, f"Quantized model accuracy : {round(avg_accuracy * 100, 2)}%")
        log_to_file(self.output_dir, f"Quantized model average IoU on all classes : {round(avg_iou * 100, 2)}% ")

        return avg_accuracy, avg_iou

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
    

