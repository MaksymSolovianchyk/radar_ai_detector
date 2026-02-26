# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

from face_detection.tf.src.models import model_family


def ssd_generate_anchors(opts):
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)



def sort_and_combine(arrays):
    squeezed_arrays = [np.squeeze(arr, axis=0) for arr in arrays]
    sorted_arrays = sorted(squeezed_arrays, key=lambda arr: (arr.shape[0], arr.shape[1]), reverse=True)
    out_1 = np.concatenate((sorted_arrays[0], sorted_arrays[1]), axis=1)
    out_2 = np.concatenate((sorted_arrays[2], sorted_arrays[3]), axis=1)
    final_array = np.concatenate((out_1, out_2), axis=0)
    return  final_array

def _decode_boxes(raw_boxes,input_shape,anchors):
    # width == height so scale is the same across the board
    scale = input_shape
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes

def sigmoid(data):
    return 1 / (1 + np.exp(-data))

def _get_sigmoid_scores(raw_scores):
    """Extracted loop from ProcessCPU (line 327) in
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
    # # this lower limit is safe for use with the sigmoid functions and float32
    RAW_SCORE_LIMIT = 80
    # just a single class ("face"), which simplifies this a lot
    # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
    #    causes overflow with IEEE single precision floats (max ~10e38)
    raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
    raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
    # 2) apply sigmoid function on clipped confidence scores
    return sigmoid(raw_scores)

def decode_facedetect_front_predictions(predictions,image_size):
    SSD_OPTIONS_FRONT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0}
    anchors=ssd_generate_anchors(SSD_OPTIONS_FRONT)
    out = sort_and_combine(predictions)
    decoded_boxes = _decode_boxes(out[:,0:16],image_size[0],anchors) #xmin, ymin, xmax, ymax
    d_boxes = decoded_boxes[:,:2,:]  #boxes xmin, ymin, xmax, ymax
    d_points = decoded_boxes[:,2:,:]  #keypoints (not used here)
    reshaped_d_boxes = d_boxes.reshape(-1, 4)
    reshaped_dpoints = d_points.reshape(-1, 12)
    decoded_scores = _get_sigmoid_scores(out[:,-1])
    reshaped_d_scores = np.expand_dims(decoded_scores, axis=1)
    boxes = np.expand_dims(reshaped_d_boxes, axis=0)
    scores = np.expand_dims(reshaped_d_scores, axis=0)
    key_points = np.expand_dims(reshaped_dpoints, axis=0)
    return boxes, scores, key_points


def generate_yunet_anchor(input_size, strides):
    centers = []
    for stride in strides:
        anchor_centers = np.stack(np.mgrid[:(input_size[1] // stride), :(input_size[0] // stride)][::-1], axis=-1)
        anchor_centers = (anchor_centers * stride).astype(np.float32).reshape(-1, 2)
        centers.append(anchor_centers)
    return centers

def decode_yunet_predictions(cfg,predictions,image_size):
    post_proc_supported_chpos = "chfirst"

    strides = [32,16,8] #model strides
    NK = 5 #number of keypoints
    predictions = [p.numpy() if isinstance(p, tf.Tensor) else p for p in predictions]
    opm=None
    if cfg.operation_mode == "prediction":
        opm = cfg.prediction
    elif cfg.operation_mode in ["chain_eqe","evaluation","chain_eqeb"]:
        opm = cfg.evaluation

    if opm.target != "host":
        if opm.output_chpos != post_proc_supported_chpos:
            for p in range(len(predictions)):
                if len(predictions[p].shape)==3:
                    predictions[p] = np.transpose(predictions[p],[0,2,1])
                elif len(predictions[p].shape)==4:
                    predictions[p] = np.transpose(predictions[p],[0,2,3,1])

    nets_out_shapes_sorted = sorted(predictions, key=lambda x: x.shape)

    model_anchor_centers = generate_yunet_anchor(image_size, strides)

    scores, bboxes, kpss = [], [], []
    for idx, stride in enumerate(strides):
        cls_pred = nets_out_shapes_sorted[4 * idx + 0].reshape(-1, 1)
        obj_pred = nets_out_shapes_sorted[4 * idx + 1].reshape(-1, 1)
        reg_pred = nets_out_shapes_sorted[4 * idx + 2].reshape(-1, 4)
        kps_pred = nets_out_shapes_sorted[4 * idx + 3].reshape(-1, NK * 2)

        # decode bboxes
        bbox_cxy = reg_pred[:, :2] * stride + model_anchor_centers[idx]
        bbox_wh = np.exp(reg_pred[:, 2:]) * stride

        tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
        tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
        br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
        br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

        bboxes.append(np.stack([tl_x, tl_y, br_x, br_y], -1))
        per_kps = np.concatenate(
            [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + model_anchor_centers[idx]) for i in range(NK)], axis=-1)
        kpss.append(per_kps)

        scores.append(cls_pred * obj_pred)

    scores = np.concatenate(scores, axis=0).reshape(-1)
    bboxes = np.concatenate(bboxes, axis=0)
    kpss = np.concatenate(kpss, axis=0)

    # Normalize predictions to [0, 1] (relative to resized image)
    bboxes[:, [0, 2]] /= image_size[0]  # x coordinates
    bboxes[:, [1, 3]] /= image_size[1]  # y coordinates
    for i in range(kpss.shape[1] // 2):
        kpss[:, 2 * i] /= image_size[0]   # x
        kpss[:, 2 * i + 1] /= image_size[1] # y

    reshaped_d_scores = np.expand_dims(scores, axis=1)
    boxes = np.expand_dims(bboxes, axis=0)
    scores = np.expand_dims(reshaped_d_scores, axis=0)
    key_points = np.expand_dims(kpss, axis=0)
    key_points_padded = np.pad(key_points, ((0, 0), (0, 0), (0, 2)), mode='constant', constant_values=0)
    
    return boxes, scores, key_points_padded



def blind_nms(boxes_scores,max_output_size,iou_threshold,score_threshold):

    boxes = boxes_scores[...,:4] # shape (anchors,classes,4)
    keypoints = boxes_scores[...,4:16] # shape (anchors,classes,1)
    scores = boxes_scores[...,16] # shape (anchors,classes,12)

    boxes = tf.reshape(boxes,[-1,4]) # shape (anchors*classes,4)
    keypoints = tf.reshape(keypoints,[-1,12]) # shape (anchors*classes,12)
    scores = tf.reshape(scores,[-1]) # shape (anchors*classes)

    selected_indices,valid_outputs = tf.raw_ops.NonMaxSuppressionV4(boxes=boxes,
                                                                    scores=scores,
                                                                    max_output_size=max_output_size,
                                                                    iou_threshold=iou_threshold,
                                                                    score_threshold=score_threshold,
                                                                    pad_to_max_output_size=True)

    nmsed_boxes = tf.gather(boxes,selected_indices)   # shape (max_output_size,4)
    nmsed_scores = tf.gather(scores,selected_indices) # shape (max_output_size)
    nmsed_keypoints = tf.gather(keypoints,selected_indices) # shape (max_output_size,12)
    valid_outputs = tf.cast(tf.range(max_output_size)<tf.cast(valid_outputs,tf.int32),tf.float32) # shape (max_output_size)

    return tf.concat(values=[nmsed_boxes,nmsed_keypoints,nmsed_scores[...,None],valid_outputs[...,None]],axis=-1) # shape (max_output_size,6)

def st_combined_nms(boxes_scores,max_output_size,iou_threshold,score_threshold):

    boxes_scores = tf.transpose(boxes_scores,[1,0,2]) # shape (classes,anchors,5+12=17) FLOAT32

    args = {'max_output_size':max_output_size,
            'iou_threshold':iou_threshold,
            'score_threshold':score_threshold}

    classed_value = tf.map_fn(lambda x : blind_nms(x,**args), boxes_scores) # shape (classes,max_output_size,6) FLOAT32
    classed_value = tf.reshape(classed_value,[-1,18])                        # shape (classes*max_output_size,6) FLOAT32

    classed_valid =  classed_value[:,-1]                                    # shape (classes*max_output_size)   FLOAT32
    classed_scores = classed_value[:,-2]*classed_valid                      # shape (classes*max_output_size)   FLOAT32
    classed_boxes  = classed_value[:,:4]*classed_valid[...,None]           # shape (classes*max_output_size,4) FLOAT32
    classed_keypoints  = classed_value[:,4:16]*classed_valid[...,None]      # shape (classes*max_output_size,12) FLOAT32

    cls_nb = tf.shape(classed_scores)[0] // max_output_size # number of classes INT32

    nmsed_scores,classed_indices = tf.raw_ops.TopKV2(input=classed_scores,k=max_output_size,sorted=True,index_type=tf.int32) # shape (max_output_size) FLOAT32, (max_output_size) INT32

    nmsed_cls = tf.cast(classed_indices//max_output_size,tf.float32) # shape (max_output_size) FLOAT32

    nmsed_boxes = tf.gather(classed_boxes,classed_indices) # shape (max_output_size,4) FLOAT32
    nmsed_keypoints = tf.gather(classed_keypoints,classed_indices) # shape (max_output_size,12) FLOAT32

    return tf.concat(values=[nmsed_boxes,nmsed_keypoints,nmsed_scores[...,None],nmsed_cls[...,None]],axis=-1) # shape (max_output_size,6)

def st_combined_non_max_suppression(boxes,scores,keypoints,max_total_size,iou_threshold,score_threshold):

    boxes_scores = tf.concat(values=[boxes,keypoints,scores[...,None]],axis=-1) # shape (batch,anchors,classes,5)

    args = {'max_output_size':max_total_size,
            'iou_threshold':iou_threshold,
            'score_threshold':score_threshold}

    nmsed_values = tf.map_fn(lambda x : st_combined_nms(x,**args), boxes_scores)

    #nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints = nmsed_values[...,:-2],nmsed_values[...,-2],nmsed_values[...,-1]
    nmsed_boxes, nmsed_keypoints, nmsed_scores, nmsed_classes = nmsed_values[...,0:4],nmsed_values[...,4:16],nmsed_values[...,-2],nmsed_values[...,-1]

    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints


def nms_box_keypoints_filtering(
                boxes: tf.Tensor,
                scores: tf.Tensor,
                keypoints: tf.Tensor,
                max_boxes: int = None,
                score_threshold: float = None,
                iou_threshold: float = None,
                clip_boxes: bool = True) -> tuple:
    
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    num_classes = tf.shape(scores)[-1]
    
    # Convert box coordinates from (x1, y1, x2, y2) to (y1, x1, y2, x2)
    boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

    # NMS is run by class, so we need to replicate the boxes num_classes times.
    boxes_t = tf.tile(boxes, [1, 1, num_classes])
    nms_input_boxes = tf.reshape(boxes_t, [batch_size, num_boxes, num_classes, 4])

    keypoints_t = tf.tile(keypoints, [1, 1, num_classes])
    nms_input_keypoints = tf.reshape(keypoints_t, [batch_size, num_boxes, num_classes, 12])

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints = st_combined_non_max_suppression(
                                                     boxes=nms_input_boxes,
                                                     scores=scores,
                                                     keypoints=nms_input_keypoints,
                                                     max_total_size=max_boxes,
                                                     iou_threshold=iou_threshold,
                                                     score_threshold=score_threshold)


    # Convert coordinates of NMSed boxes to (x1, y1, x2, y2)
    nmsed_boxes = tf.stack([nmsed_boxes[..., 1], nmsed_boxes[..., 0],
                            nmsed_boxes[..., 3], nmsed_boxes[..., 2]],
                            axis=-1)

    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints


def get_nmsed_detections(cfg, predictions, image_size):

    num_classes = len(cfg.dataset.class_names)
    cpp = cfg.postprocessing
    
    if model_family(cfg.model.model_type) == "facedetect_front":
        boxes, scores, keypoints = decode_facedetect_front_predictions(predictions,image_size)

    elif model_family(cfg.model.model_type) == "yunet":
        boxes, scores, keypoints = decode_yunet_predictions(cfg,predictions,image_size)

    else:
        raise ValueError("Unsupported model type")
        
    # NMS the detections
    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints = nms_box_keypoints_filtering(
                    boxes,
                    scores,
                    keypoints,
                    max_boxes=cpp.max_detection_boxes,
                    score_threshold=cpp.confidence_thresh,
                    iou_threshold=cpp.NMS_thresh)

    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints

def get_detections(cfg, predictions, image_size):


    if model_family(cfg.model.model_type) == "facedetect_front":
        boxes, scores, keypoints = decode_facedetect_front_predictions(predictions,image_size)
    elif model_family(cfg.model.model_type) == "yunet":
        boxes, scores, keypoints = decode_yunet_predictions(cfg,predictions,image_size)
    else:
        raise ValueError("Unsupported model type")

    return boxes, scores, keypoints