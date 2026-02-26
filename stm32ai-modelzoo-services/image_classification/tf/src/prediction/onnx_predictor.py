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
import onnxruntime
import cv2

# Suppress warnings and TensorFlow logs for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import utility functions for AI runner and ONNX prediction
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import predict_onnx
from image_classification.tf.src.utils import ai_runner_invoke


class ONNXModelPredictor:
    """
    A class to handle predictions using an ONNX model. This class includes methods for:
    - Loading and preprocessing images
    - Running inference on the ONNX model
    - Annotating and saving prediction results
    - Displaying results in a tabular format
    """
    def __init__(self, cfg, model, dataloaders):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg: Configuration object containing settings for the predictor.
            model: The ONNX model to use for predictions.
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
        self.input_chpos = getattr(cfg.prediction, 'input_chpos', 'chlast') if hasattr(cfg, 'prediction') else 'chlast'

        # Initialize ONNX runtime session and AI runner interpreter
        self.sess = onnxruntime.InferenceSession(model.model_path)
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

    def _preprocess(self, img):
        """
        Preprocess the image for ONNX model inference.

        Args:
            img: The input image.

        Returns:
            The preprocessed image.
        """
        if self.cfg.model.framework == "tf":
            # Dataloader is channel last with TF
            if self.input_chpos == "chfirst" or self.target == 'host':
                # The transpose is doing chlast -> chfirst as the host model is onnx channel first
                image_processed = np.transpose(img, [0, 3, 1, 2])
            else:
                image_processed = img.numpy()
        else:
            # Dataloader is already channel first with Torch
            if self.input_chpos == "chfirst" or self.target == 'host':
                image_processed = img
                image_processed = image_processed.cpu().numpy()
            else:
                image_processed = np.transpose(img, [0, 2, 3, 1])

        return image_processed

    def _get_scores(self, image_processed):
        """
        Perform inference on the preprocessed image and get prediction scores.

        Args:
            image_processed: The preprocessed image.

        Returns:
            The prediction scores.
        """
        if self.target == 'host':   # If running on the host
            scores = predict_onnx(self.sess, image_processed)   # Use ONNX runtime for inference
        elif self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:   # If running on an N6 device
            imagee = ai_interp_input_quant(self.ai_runner_interpreter, image_processed, '.onnx')
            scores = ai_runner_invoke(imagee, self.ai_runner_interpreter)   # Run inference
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
        # Optionally display the image if configured
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
            image_rgb = self._load_image(image_path)
            if image_rgb is None:
                continue
            image_processed = self._preprocess(img)
            scores = self._get_scores(image_processed)
            predicted_label, prediction_score = self._get_prediction(scores)
            self.results_table.append([predicted_label, f"{prediction_score:.1f}", image_path])
            pred_text = f"{predicted_label}: {prediction_score:.1f}%"
            self._annotate_and_save(image_rgb, pred_text, image_path)
        print('[INFO] : Prediction complete.')