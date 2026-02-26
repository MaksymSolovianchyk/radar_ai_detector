import os
import numpy as np
import tensorflow as tf
import onnxruntime
from omegaconf import DictConfig
from semantic_segmentation.tf.src.utils import vis_segmentation
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
#from image_classification.tf.src.utils import ai_runner_invoke
from semantic_segmentation.tf.src.utils import ai_runner_invoke

def _generate_output_image(image_path, raw_output, input_size, nchw=False, cfg=None):
#    if nchw and raw_output.ndim == 4:
#        raw_output = raw_output.transpose(0, 2, 3, 1)
    seg_map = tf.argmax(tf.image.resize(raw_output, size=input_size), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    vis_segmentation(image_path=image_path, seg_map=seg_map, cfg=cfg, input_size=input_size)

class ONNXModelPredictor:
    def __init__(self, cfg: DictConfig, model: onnxruntime.InferenceSession, dataloaders: list):
        self.cfg = cfg
        self.sess = model
        self.predict_ds = dataloaders['predict']  # (image, path) batched
        if self.predict_ds is None:
            raise ValueError("Prediction dataset is None")
        self.target = getattr(cfg.prediction, 'target', 'host')
        self.model_name = os.path.basename(model.model_path)
        raw_shape = self.sess.get_inputs()[0].shape  # e.g. ('unk__584', 3, 512, 512)
        dims = [int(d) if isinstance(d, (int, np.integer)) else None for d in raw_shape]
        
        input_chpos = getattr(cfg.prediction, 'input_chpos', 'chlast') if hasattr(cfg, 'prediction') else 'chlast'
        if self.cfg.model.framework == "tf":
            # Dataloader is channel last with TF
            if input_chpos=="chfirst" or self.target == 'host':
                self.nchw = True
            else:
                self.nchw = False
        else:
            # Dataloader is already channel first with Torch
            self.nchw = False
        self.height, self.width = dims[2], dims[3]
        self.input_name = self.sess.get_inputs()[0].name
        self.ai_runner = ai_runner_interp(self.target, self.model_name)

    def predict(self):
        print("[INFO] : ONNX segmentation prediction started")
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

            if self.cfg.model.framework == "tf":
                # Channel last dataloader...
                if self.height and self.width and (img.shape[1] != self.height or img.shape[2] != self.width):
                    img = tf.image.resize(img, (self.height, self.width))
            else:
                # Channel first dataloader...
                if self.height and self.width and (img.shape[2] != self.height or img.shape[3] != self.width):
                    img = tf.image.resize(img, (self.height, self.width))                

            arr = tf.cast(img, tf.float32).numpy()
            if self.nchw:
                arr = arr.transpose(0, 3, 1, 2)

            if self.target == 'host':
                raw_pred = self.sess.run(None, {self.input_name: arr})[0]
            elif self.target in ['stedgeai_n6', 'stedgeai_h7p', 'stedgeai_host']:
                q_in = ai_interp_input_quant(self.ai_runner, arr, '.onnx')
                print("q_in :", q_in.shape)
                q_out = ai_runner_invoke(q_in, self.ai_runner)
                print("q_out :", q_out.shape)
                raw_pred = ai_interp_outputs_dequant(self.ai_runner, [q_out])[0]
            else:
                raise ValueError(f"Unknown target {self.target}")

            if self.nchw:
               raw_pred = raw_pred.transpose(0, 2, 3, 1)
            print("raw_pred :", raw_pred.shape)
            _generate_output_image(image_path=image_path if image_path and os.path.isfile(image_path) else None,
                                   raw_output=raw_pred,
                                   input_size=[self.height, self.width],
                                   nchw=False,  # already converted to NHWC for overlay
                                   cfg=self.cfg)
        print("[INFO] : prediction complete")