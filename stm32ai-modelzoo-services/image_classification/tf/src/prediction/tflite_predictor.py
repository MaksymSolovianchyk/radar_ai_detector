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
import numpy as np
import tensorflow as tf
import cv2

# Suppress warnings and TensorFlow logs for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import utility functions for AI runner and ONNX prediction
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from image_classification.tf.src.utils import ai_runner_invoke
from image_classification.tf.src.preprocessing import preprocess_input


class TFLiteQuantizedModelPredictor:
    """
    A class to handle predictions using a TFLite quantized model. This class includes methods for:
    - Loading and preprocessing images
    - Running inference on the TFLite model
    - Annotating and saving prediction results
    - Displaying results in a tabular format
    """
    def __init__(self, cfg, model, dataloaders):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg: Configuration object containing settings for the predictor.
            model: The TFLite model to use for predictions.
            dataloaders: A dictionary containing the prediction dataset.
        """
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']
        self.class_names = cfg.dataset.class_names
        self.prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')
        os.makedirs(self.prediction_result_dir, exist_ok=True)
        self.results_table = []
        self.target = getattr(cfg.prediction, 'target', 'host') if hasattr(cfg, 'prediction') else 'host'
        self.model_name = os.path.basename(model.model_path)
        self.display_figures = cfg.general.display_figures

        # Initialize the TFLite interpreter for the quantized model
        self.interpreter_quant = tf.lite.Interpreter(model_path=model.model_path)
        self.interpreter_quant.allocate_tensors()
        self.input_details = self.interpreter_quant.get_input_details()[0]
        self.input_index_quant = self.input_details["index"]
        self.output_details = self.interpreter_quant.get_output_details()[0]
        self.output_index_quant = self.output_details["index"]

        # Initialize the AI runner interpreter for edge devices
        self.ai_runner_interpreter = ai_runner_interp(self.target, self.model_name)

    def _load_image(self, img_path):
        """
        Load an image from the given path and convert it to RGB format.

        Args:
            img_path: Path to the image file.

        Returns:
            The loaded image in RGB format, or None if the image could not be loaded.
        """
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] : Could not load image {img_path}")
            return None
        if len(image.shape) != 3:   # If the image is grayscale, convert it to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _get_scores(self, img):
        """
        Perform inference on the input image and get prediction scores.

        Args:
            img: The input image.

        Returns:
            The prediction scores.
        """
        if self.target == 'host':
            print("img.shape",img.shape)
            image_processed = preprocess_input(img, self.input_details)
            image_processed = np.squeeze(image_processed, axis=0)   # Remove batch dimension
            print("image_processed.shape",image_processed.shape)
            self.interpreter_quant.set_tensor(self.input_index_quant, image_processed)
            self.interpreter_quant.invoke()
            scores = self.interpreter_quant.get_tensor(self.output_index_quant)
            if self.output_details['dtype'] in [np.uint8, np.int8]:
                scores = (np.float32(scores) - self.output_details['quantization'][1]) * self.output_details['quantization'][0]
        elif self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            imagee = ai_interp_input_quant(self.ai_runner_interpreter, img, '.tflite')
#            imagee = ai_interp_input_quant(self.ai_runner_interpreter, img[None].numpy(), '.tflite')
            scores = ai_runner_invoke(imagee, self.ai_runner_interpreter)
            scores = ai_interp_outputs_dequant(self.ai_runner_interpreter, [scores])[0]
        else:
            raise ValueError(f"Unknown target: {self.target}")
        return np.squeeze(scores)   # Remove single-dimensional entries

    def _get_prediction(self, scores):
        """
        Get the predicted label and score from the model's output.

        Args:
            scores: The prediction scores.

        Returns:
            A tuple containing the predicted label and the prediction score.
        """
        if scores.shape == ():  # Handle the case where scores is a scalar
            scores = [scores]
        max_score_index = np.argmax(scores)
        prediction_score = 100 * scores[max_score_index]
        predicted_label = self.class_names[max_score_index]
        print("predicted_label", predicted_label)
        print("prediction_score", prediction_score)
        return predicted_label, prediction_score

    def _annotate_and_save(self, image, pred_text, img_path):
        """
        Annotate the image with the prediction and save it.

        Args:
            image: The RGB image to annotate.
            pred_text: The prediction text to overlay on the image.
            img_path: The original image path (used for naming the saved file).
        """
        height, width, _ = image.shape
        thick = int(0.6 * (height + width) / 600)   # Calculate text thickness
        # Draw a rectangle for the text background
        cv2.rectangle(
            image,
            pt1=(int(0.2*width//2) - int(0.037*width), int(0.2*height//2) - int(2*0.037*height)),
            pt2=(int(0.2*width//2) + int(len(pred_text)*0.037*width), int(0.2*height//2) + int(0.5*0.037*height)),
            color=[0, 0, 0],
            thickness=-1
        )
        # Overlay the prediction text
        cv2.putText(
            image,
            pred_text,
            (int(0.2*width//2), int(0.2*height//2)),
            cv2.FONT_HERSHEY_COMPLEX,
            width/500,
            (255, 255, 255),
            thick,
            lineType=cv2.LINE_AA
        )
        # Convert the image back to BGR and save it
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_name = os.path.splitext(img_path)[0]
        pred_res_filename = os.path.join(self.prediction_result_dir, f"{os.path.basename(img_name)}.png")
        cv2.imwrite(pred_res_filename, image_bgr)
        if self.display_figures:
            cv2.imshow('image', image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def predict(self):
        """
        Perform predictions on the dataset and save the results.
        """
        for img, img_path in self.predict_ds:   # Iterate over the prediction dataset
            image_path = img_path[0] if isinstance(img_path, tuple) else img_path.numpy()[0].decode()   # Decode the image path 
            # image_path = img_path.numpy()[0].decode()
            image_rgb = self._load_image(image_path)
            if image_rgb is None:
                continue
            img = img.numpy()
            # img = np.squeeze(img, axis=0)   # Remove batch dimension
            scores = self._get_scores(img)
            predicted_label, prediction_score = self._get_prediction(scores)
            self.results_table.append([predicted_label, f"{prediction_score:.1f}", image_path])
            pred_text = f"{predicted_label}: {prediction_score:.1f}%"
            self._annotate_and_save(image_rgb, pred_text, image_path)
        print('[INFO] : Prediction complete.')
