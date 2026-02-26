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
import tensorflow as tf
import tqdm

from semantic_segmentation.tf.src.evaluation import prediction_accuracy_on_batch, iou_per_class
from common.utils import count_h5_parameters, log_to_file 


# Define a class for evaluating Keras models
class KerasModelEvaluator:
    """
    A class to evaluate TensorFlow Keras models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The Keras model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig = None, model: object = None, 
                 dataloaders: dict = None):
        self.cfg = cfg
        self.model = model
        self.test_ds = dataloaders['test']
        self.val_ds = dataloaders['valid']
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
            self.eval_ds = self.val_ds
            self.name_ds = "validation_set"
    

    def _run_evaluate(self):
        """
        Runs the evaluation process and computes metrics.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        # Count the number of parameters in the model and log them
        count_h5_parameters(output_dir=self.output_dir, 
                            model=self.model)      

        accuracy_batch = []
        iou_global_window = []
        num_classes = len(self.class_names)
        for (image, mask) in tqdm.tqdm(self.eval_ds, total=len(self.eval_ds)):
            # Run inference, image/mask are already preprocessed in tf.dataset
            out = self.model.predict(image)

            pred_mask = np.argmax(out, axis=-1)
            if mask.shape[-1] == 1:
                true_mask = tf.squeeze(mask, axis=-1).numpy()
            else:
                # If the last dimension is not 1, do not squeeze or handle accordingly
                true_mask = mask.numpy()
            accuracy_on_batch = prediction_accuracy_on_batch(pred_mask, true_mask)
            accuracy_batch.append(accuracy_on_batch)

            # Calculate IoU for each class and per image
            for p_msk, t_msk in zip(pred_mask, true_mask):
                ious_per_image = iou_per_class(p_msk, t_msk, num_classes)
                # Calculate mean IoU for this sample all class for which we have an IoU included
                if ious_per_image:
                    for iou in ious_per_image:
                        iou_global_window.append(iou)

        avg_accuracy = np.mean(accuracy_batch)
        avg_iou = np.mean(iou_global_window) if iou_global_window else 0  # Handle case with no IoU scores

        print(f"[INFO] : Accuracy of float model on {self.name_ds} = {round(avg_accuracy*100, 2)}%")
        print(f"[INFO] : Average IoU of float model (all classes) on {self.name_ds} = {round(avg_iou*100, 2)}%")
        mlflow.log_metric(f"float_acc_{self.name_ds}", round(avg_accuracy*100, 2))
        mlflow.log_metric(f"float_avg_iou_{self.name_ds}", round(avg_iou*100, 2))
        log_to_file(self.output_dir, f"TF/Keras Float model {self.name_ds}:")
        log_to_file(self.output_dir, f"Accuracy of float model : {round(avg_accuracy*100, 2)}%")
        log_to_file(self.output_dir, f"Average IoU of float model (all classes) : {round(avg_iou*100, 2)}% ")

        return avg_accuracy, avg_iou

    def evaluate(self):
        """
        Executes the full evaluation process.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        self._prepare_evaluation()  # Prepare the evaluation process
        acc = self._run_evaluate()  # Run the evaluation
        print('[INFO] : Evaluation complete.')
        return acc  # Return the accuracy
    
