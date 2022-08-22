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
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - reg_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        w, h = self.input_size
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        reg_labels = (keypoints / np.array([w, h])).astype(np.float32)
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        return reg_labels, keypoint_weights

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        if encoded.shape[-1] == 2:
            N, K, _ = encoded.shape
            normalized_coords = encoded.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif encoded.shape[-1] == 4:
            # split coords and sigma if outputs contain output_sigma
            normalized_coords = encoded[..., :2].copy()
            output_sigma = encoded[..., 2:4].copy()

            scores = (1 - output_sigma).mean(axis=-1)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {encoded.shape[-1]}')

        w, h = self.input_size
        keypoints = normalized_coords * np.array([w, h])

        return keypoints, scores
