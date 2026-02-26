import os
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from semantic_segmentation.tf.src.utils import vis_segmentation
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
#from image_classification.tf.src.utils import ai_runner_invoke
from semantic_segmentation.tf.src.utils import ai_runner_invoke

def _generate_output_image(image_path, raw_output, input_size, nchw=False, cfg=None):
    if nchw and raw_output.ndim == 4:
        raw_output = raw_output.transpose(0, 2, 3, 1)
    seg_map = tf.argmax(tf.image.resize(raw_output, size=input_size), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    vis_segmentation(image_path=image_path, seg_map=seg_map, cfg=cfg, input_size=input_size)

class TFLiteQuantizedModelPredictor:
    def __init__(self, cfg: DictConfig, model: tf.lite.Interpreter, dataloaders: list):
        self.cfg = cfg
        self.interpreter = model
        self.predict_ds = dataloaders['predict']  # (image, path) batched
        if self.predict_ds is None:
            raise ValueError("Prediction dataset is None")
        self.target = getattr(cfg.prediction, 'target', 'host')
        self.model_name = os.path.basename(cfg.model.model_path)
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        shape = inp['shape']
        if shape[0] in (None, 1):
            shape = shape[1:]
        self.height, self.width = int(shape[0]), int(shape[1])
        self.in_index = inp['index']
        self.out_details = self.interpreter.get_output_details()[0]
        self.out_index = self.out_details['index']
        self.ai_runner = ai_runner_interp(self.target, self.model_name)

    def predict(self):
        print("[INFO] : TFLite segmentation prediction started")
        for sample in self.predict_ds:
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                img = sample[0]
                path_t = sample[1]
                raw = path_t.numpy()[0]
                image_path = raw.decode() if isinstance(raw, bytes) else str(raw)
            else:
                img = sample[0] if isinstance(sample, (tuple, list)) else sample
                image_path = None

            if len(img.shape) == 3:
                img = tf.expand_dims(img, 0)

            if img.shape[1] != self.height or img.shape[2] != self.width:
                img = tf.image.resize(img, (self.height, self.width))

            if self.target == 'host':
                img_cast = tf.cast(img, self.interpreter.get_input_details()[0]['dtype'])
                self.interpreter.set_tensor(self.in_index, img_cast.numpy())
                self.interpreter.invoke()
                raw_pred = self.interpreter.get_tensor(self.out_index)
            elif self.target in ['stedgeai_n6', 'stedgeai_h7p', 'stedgeai_host']:
                arr = tf.cast(img, tf.float32).numpy()
                q_in = ai_interp_input_quant(self.ai_runner, arr, '.tflite')
                q_out = ai_runner_invoke(q_in, self.ai_runner)
                raw_pred = ai_interp_outputs_dequant(self.ai_runner, [q_out])[0]
            else:
                raise ValueError(f"Unknown target {self.target}")

            if raw_pred.shape[1] != self.height or raw_pred.shape[2] != self.width:
                raw_pred = tf.image.resize(raw_pred, (self.height, self.width)).numpy()

            _generate_output_image(image_path=image_path if image_path and os.path.isfile(image_path) else None,
                                   raw_output=raw_pred,
                                   input_size=[self.height, self.width],
                                   nchw=False,
                                   cfg=self.cfg)
        print("[INFO] : prediction complete")