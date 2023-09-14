# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from mmpose.structures import bbox_cs2xyxy, bbox_xyxy2cs
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
        num_select (int): The number of candidate instances
        num_keypoints (int): The Number of keypoints
    """

    auxiliary_encode_keys = {'area', 'bboxes', 'img_shape'}

    def __init__(self, num_select: int = 100, num_keypoints: int = 17):
        super().__init__()

        self.num_select = num_select
        self.num_keypoints = num_keypoints

    def encode(
        self,
        img_shape,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
        area: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
    ) -> dict:
        """Encoding keypoints, area and bbox from input image space to
        normalized space.

        Args:
            - img_shape (Sequence[int]): The shape of image in the format
                of (width, height).
            - keypoints (np.ndarray): Keypoint coordinates in
                shape (N, K, D).
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K)
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
            bboxes = np.concatenate(bbox_xyxy2cs(bboxes), axis=-1)
            bboxes_labels = bboxes / np.array([w, h, w, h], dtype=np.float32)

        if area is not None:
            area_labels = area / float(w * h)

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
            tuple: Decoded keypoints, scores, and boxes.
        """

        num_keypoints = self.num_keypoints

        prob = pred_logits

        prob = prob.reshape(-1)
        topk_indexes = np.argsort(-prob)[:self.num_select]
        topk_values = np.take_along_axis(prob, topk_indexes, axis=0)

        scores = np.tile(topk_values[:, np.newaxis], [1, num_keypoints])

        # bbox
        topk_boxes = topk_indexes // pred_logits.shape[1]

        boxes = bbox_cs2xyxy(*np.split(pred_boxes, [2], axis=-1))
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
            np.tile(topk_keypoints[:, np.newaxis], [1, num_keypoints * 3]),
            axis=0)

        Z_pred = keypoints[:, :(num_keypoints * 2)]
        V_pred = keypoints[:, (num_keypoints * 2):]
        Z_pred = Z_pred * np.tile(np.hstack([img_w, img_h]),
                                  [num_keypoints])[np.newaxis, :]
        keypoints_res = np.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        keypoint = keypoints_res.reshape(-1, num_keypoints, 3)[:, :, :2]

        return keypoint, scores, boxes
