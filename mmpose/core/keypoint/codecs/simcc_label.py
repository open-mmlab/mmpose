# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class SimCCLabel(BaseKeypointCodec):
    r"""Generate keypoint representation via "SimCC" approach.
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    Note:

        - input image size: [w, h]

    Args:
        input_size (tuple): Input image size in [w, h]

    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float = 6.0,
                 simdr_split_ratio: float = 2.0) -> None:
        super().__init__()

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.simdr_split_ratio = simdr_split_ratio

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC with Gaussian label smoothing.

        Note that the original keypoint coordinates should be in the input
        image space.
        """

        w, h = self.input_size
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        reg_label = keypoints / np.array([w, h])
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        return reg_label, keypoint_weights

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

    def _generate_sa_simdr(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encoding keypoints into."""
