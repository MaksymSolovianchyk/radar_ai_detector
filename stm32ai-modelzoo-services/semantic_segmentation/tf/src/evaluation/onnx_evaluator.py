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
import warnings
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tqdm

from semantic_segmentation.tf.src.utils import ai_runner_invoke, get_batch_data_and_masks
from common.evaluation import model_is_quantized, predict_onnx
from semantic_segmentation.tf.src.evaluation import prediction_accuracy_on_batch, iou_per_class
from common.utils import (
    tf_dataset_to_np_array, ai_runner_interp, ai_interp_input_quant,
    ai_interp_outputs_dequant, log_to_file)  



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
            self.nchw = False

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
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_model_type(self):
        """
        Determines whether the model is quantized or float.
        """
        return 'quantized' if model_is_quantized(self.input_model.model_path) else 'float'

    def _get_ai_runner_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        """
        name_model = os.path.basename(self.input_model.model_path)
#        name_model = str(self.input_model.model_path)
#        name_model = self.input_model.get_inputs()[0].name
        return ai_runner_interp(target, name_model)


    def _run_evaluate(self):
        """
        Runs the evaluation process and computes metrics.

        Returns:
            Tuple[float, np.ndarray]: Accuracy and confusion matrix.
        """
        num_classes = len(self.class_names)
        self._ensure_output_dir()       # Ensure the output directory exists
        target = self._get_target()     # Get the evaluation target
        model_type = self._get_model_type() # Determine the model type
        ai_runner_interpreter = self._get_ai_runner_interpreter(target=target)   # Get the AI runner interpreter
        accuracy_list = []
        iou_global_window = []

        if target == "host":
                unbatched_dataset = self.eval_ds.unbatch()
                val_ds = unbatched_dataset.batch(1)
                for images, masks in tqdm.tqdm(val_ds):
                    batch_data, batch_masks = get_batch_data_and_masks(images, masks, nchw=self.nchw)
                    out = predict_onnx(self.input_model, batch_data)
                    # Handle both channel-first and channel-last
                    if out.shape[1] == num_classes:
                        pred_mask = np.argmax(out, axis=1)
                    elif out.shape[-1] == num_classes:
                        pred_mask = np.argmax(out, axis=-1)
                    else:
                        raise ValueError("Unexpected output shape for ONNX model.")
                    true_mask = np.squeeze(batch_masks, axis=1) if batch_masks.shape[1] == 1 else batch_masks
                    accuracy = prediction_accuracy_on_batch(pred_mask, true_mask)
                    accuracy_list.append(accuracy)

                    ious_per_image = iou_per_class(pred_mask, true_mask, num_classes)
                    if ious_per_image:
                        for iou in ious_per_image:
                            iou_global_window.append(iou)

        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            unbatched_dataset = self.eval_ds.unbatch()
            val_ds = unbatched_dataset.batch(1)
            for images, masks in tqdm.tqdm(val_ds):
                batch_data, batch_masks = get_batch_data_and_masks(images, masks, nchw=self.nchw)
                for img, msk in zip(batch_data, batch_masks):
                    img = np.expand_dims(img, axis=0)
                    data = ai_interp_input_quant(ai_runner_interpreter, img, '.onnx')
                    out = ai_runner_invoke(data, ai_runner_interpreter)
                    out = ai_interp_outputs_dequant(ai_runner_interpreter, out)[0]
                    # Handle both channel-first and channel-last
                    if out.shape[1] == num_classes:
                        pred_mask = np.argmax(out, axis=1)
                    elif out.shape[-1] == num_classes:
                        pred_mask = np.argmax(out, axis=-1)
                    else:
                        raise ValueError("Unexpected output shape for ONNX model.")
                    true_mask = msk
                    accuracy = prediction_accuracy_on_batch(pred_mask, true_mask)
                    accuracy_list.append(accuracy)

                    ious_per_image = iou_per_class(pred_mask, true_mask, num_classes)
                    if ious_per_image:
                        for iou in ious_per_image:
                            iou_global_window.append(iou)

        avg_accuracy = np.mean(accuracy_list)
        avg_iou = np.mean(iou_global_window) if iou_global_window else 0  # Handle case with no IoU scores

        print(f"[INFO] : {model_type} model accuracy on {self.name_ds} = {round(avg_accuracy*100, 2)}%")
        print(f"[INFO] : {model_type} model average IoU (all classes) on {self.name_ds} = {round(avg_iou*100, 2)}%")
        mlflow.log_metric(f"{model_type.lower()}_acc_{self.name_ds}", round(avg_accuracy*100, 2))
        mlflow.log_metric(f"{model_type.lower()}_avg_iou_{self.name_ds}", round(avg_iou*100, 2))
        log_to_file(self.output_dir, f"ONNX {model_type} model {self.name_ds}:")
        log_to_file(self.output_dir, f"{model_type} model accuracy : {round(avg_accuracy*100, 2)}%")
        log_to_file(self.output_dir, f"{model_type} model average IoU (all classes) : {round(avg_iou*100, 2)}% ")

        return avg_accuracy, avg_iou

    def evaluate(self):
        """
        Executes the full evaluation process.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        self._prepare_evaluation()      
        acc, avg_iou = self._run_evaluate()  
        print('[INFO] : Evaluation complete.')
        return acc, avg_iou

