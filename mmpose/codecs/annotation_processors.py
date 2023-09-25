# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec

INF = 1e6
NEG_INF = -1e6


class BaseAnnotationProcessor(BaseKeypointCodec):
    """Base class for annotation processors."""

    def decode(self, *args, **kwargs):
        pass


@KEYPOINT_CODECS.register_module()
class YOLOXPoseAnnotationProcessor(BaseAnnotationProcessor):
    """Convert dataset annotations to the input format of YOLOX-Pose.

    This processor expands bounding boxes and converts category IDs to labels.

    Args:
        expand_bbox (bool, optional): Whether to expand the bounding box
            to include all keypoints. Defaults to False.
        input_size (tuple, optional): The size of the input image for the
            model, formatted as (h, w). This argument is necessary for the
            codec in deployment but is not used indeed.
    """

    auxiliary_encode_keys = {'category_id', 'bbox'}
    label_mapping_table = dict(
        bbox='bboxes',
        bbox_labels='labels',
        keypoints='keypoints',
        keypoints_visible='keypoints_visible',
        area='areas',
    )

    def __init__(self,
                 expand_bbox: bool = False,
                 input_size: Optional[Tuple] = None):
        super().__init__()
        self.expand_bbox = expand_bbox

    def encode(self,
               keypoints: Optional[np.ndarray] = None,
               keypoints_visible: Optional[np.ndarray] = None,
               bbox: Optional[np.ndarray] = None,
               category_id: Optional[List[int]] = None
               ) -> Dict[str, np.ndarray]:
        """Encode keypoints, bounding boxes, and category IDs.

        Args:
            keypoints (np.ndarray, optional): Keypoints array. Defaults
                to None.
            keypoints_visible (np.ndarray, optional): Visibility array for
                keypoints. Defaults to None.
            bbox (np.ndarray, optional): Bounding box array. Defaults to None.
            category_id (List[int], optional): List of category IDs. Defaults
                to None.

        Returns:
            Dict[str, np.ndarray]: Encoded annotations.
        """
        results = {}

        if self.expand_bbox and bbox is not None:
            # Handle keypoints visibility
            if keypoints_visible.ndim == 3:
                keypoints_visible = keypoints_visible[..., 0]

            # Expand bounding box to include keypoints
            kpts_min = keypoints.copy()
            kpts_min[keypoints_visible == 0] = INF
            bbox[..., :2] = np.minimum(bbox[..., :2], kpts_min.min(axis=1))

            kpts_max = keypoints.copy()
            kpts_max[keypoints_visible == 0] = NEG_INF
            bbox[..., 2:] = np.maximum(bbox[..., 2:], kpts_max.max(axis=1))

            results['bbox'] = bbox

        if category_id is not None:
            # Convert category IDs to labels
            bbox_labels = np.array(category_id).astype(np.int8) - 1
            results['bbox_labels'] = bbox_labels

        return results
