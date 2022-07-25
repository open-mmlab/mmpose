# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class RegressionLabel(BaseKeypointCodec):
    r"""Generate keypoint coordinates.

    Note:

        - input image size: [w, h]

    Args:
        input_size (tuple): Input image size in [w, h]

    """

    def __init__(self, input_size: Tuple[int, int]) -> None:
        super().__init__()

        self.input_size = input_size

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encoding keypoints from input image space to normalized space."""

        w, h = self.input_size
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        reg_labels = keypoints / np.array([w, h])
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        return reg_labels, keypoint_weights

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K, 1).
                It usually represents the confidence of the keypoint prediction
        """

        if encoded.shape[-1] == 2:
            N, K, _ = encoded.shape
            normalized_coords = encoded.copy()
            scores = np.ones((N, K, 1), dtype=np.float32)
        elif encoded.shape[-1] == 4:
            # split coords and sigma if outputs contain output_sigma
            normalized_coords = encoded[..., :2].copy()
            output_sigma = encoded[..., 2:4].copy()
            scores = (1 - output_sigma).mean(axis=2, keepdims=True)

        w, h = self.input_size
        keypoints = normalized_coords * np.array([w, h])

        # Unsqueeze the instance dimension for single-instance results
        keypoints = keypoints[None]
        scores = scores[None]

        return keypoints, scores
