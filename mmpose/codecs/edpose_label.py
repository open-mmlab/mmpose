# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class EDPoseLabel(BaseKeypointCodec):
    r"""Generate keypoint and label coordinates for `ED-Pose`_ by
    Yang J. et al (2023).

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        - area (np.ndarray): Area in shape (N)
        - bbox (np.ndarray): Bbox in shape (N, 4)

    Args:
        num_select (int): The number of candidate keypoints
        num_body_points (int): The Number of keypoints
        not_to_xyxy (bool): Whether convert bbox from cxcy to
            xyxy.
    """

    auxiliary_encode_keys = {'area', 'bboxes', 'img_shape'}

    def __init__(self,
                 num_select: int = 100,
                 num_body_points: int = 17,
                 not_to_xyxy: bool = False):
        super().__init__()

        self.num_select = num_select
        self.num_body_points = num_body_points
        self.not_to_xyxy = not_to_xyxy

    def encode(
        self,
        img_shape,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        area: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
    ) -> dict:
        """Encoding keypoints 、area、bbox from input image space to normalized
        space.

        Args:
            - keypoints (np.ndarray): Keypoint coordinates in
                shape (N, K, D).
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
            - area (np.ndarray):
            - bboxes (np.ndarray):

        Returns:
            encoded (dict): Contains the following items:

                - keypoint_labels (np.ndarray): The processed keypoints in
                    shape like (N, K, D).
                - keypoints_visible (np.ndarray): Keypoint visibility in shape
                    (N, K, D)
                - area_labels (np.ndarray): The processed target
                    area in shape (N).
                - bboxes_labels: The processed target bbox in
                    shape (N, 4).
        """
        w, h = img_shape

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if bboxes is not None:
            bboxes = self.box_xyxy_to_cxcywh(bboxes)
            bboxes_labels = bboxes / np.array([w, h, w, h], dtype=np.float32)

        if area is not None:
            area_labels = area / (
                np.array(w, dtype=np.float32) * np.array(h, dtype=np.float32))

        if keypoints is not None:
            keypoint_labels = keypoints / np.array([w, h], dtype=np.float32)

        encoded = dict(
            keypoint_labels=keypoint_labels,
            area_labels=area_labels,
            bboxes_labels=bboxes_labels,
            keypoints_visible=keypoints_visible)

        return encoded

    def decode(self, input_shapes: np.ndarray, pred_logits: np.ndarray,
               pred_boxes: np.ndarray, pred_keypoints: np.ndarray):
        """Select the final top-k keypoints, and decode the results from
        normalize size to origin input size.

        Args:
            input_shapes (Tensor): The size of input image resize.
            test_cfg (ConfigType): Config of testing.
            pred_logits (Tensor): The result of score.
            pred_boxes (Tensor): The result of bbox.
            pred_keypoints (Tensor): The result of keypoints.

        Returns:
        """

        num_body_points = self.num_body_points

        prob = pred_logits

        prob_reshaped = prob.reshape(-1)
        topk_indexes = np.argsort(-prob_reshaped)[:self.num_select]
        topk_values = np.take_along_axis(prob_reshaped, topk_indexes, axis=0)

        scores = np.tile(topk_values[:, np.newaxis], [1, num_body_points])

        # bbox
        topk_boxes = topk_indexes // pred_logits.shape[1]
        if self.not_to_xyxy:
            boxes = pred_boxes
        else:
            x_c, y_c, w, h = np.split(pred_boxes, 4, axis=-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w),
                 (y_c + 0.5 * h)]
            boxes = np.concatenate(b, axis=1)

        boxes = np.take_along_axis(
            boxes, np.tile(topk_boxes[:, np.newaxis], [1, 4]), axis=0)

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = np.split(input_shapes, 2, axis=0)
        scale_fct = np.hstack([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[np.newaxis, :]

        # keypoints
        topk_keypoints = topk_indexes // pred_logits.shape[1]
        keypoints = np.take_along_axis(
            pred_keypoints,
            np.tile(topk_keypoints[:, np.newaxis], [1, num_body_points * 3]),
            axis=0)

        Z_pred = keypoints[:, :(num_body_points * 2)]
        V_pred = keypoints[:, (num_body_points * 2):]
        Z_pred = Z_pred * np.tile(
            np.hstack([img_w, img_h]), [num_body_points])[np.newaxis, :]
        keypoints_res = np.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        keypoint = keypoints_res.reshape(-1, num_body_points, 3)[:, :, :2]

        return keypoint, scores, boxes

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        return np.stack(b, dim=-1)
