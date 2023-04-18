# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class ReshapedKeypoints(BaseKeypointCodec):
    r"""Generate reshaped keypoint coordinates.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Encoded:

        - keypoints (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        num_keypoints (int): The number of keypoints in the dataset.
    """

    def __init__(self, num_keypoints: int) -> None:
        super().__init__()

        self.num_keypoints = num_keypoints

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The reshaped regression keypoints
                in shape (K * D, N) where D is 2 for 2d coordinates
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        valid = (keypoints >= 0).all(axis=-1) & (keypoints_visible > 0.5)

        assert keypoints.ndim in {2, 3}
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]

        N = keypoints.shape[0]
        reshaped_keypoints = keypoints.transpose(1, 2, 0).reshape(-1, N)
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        encoded = dict(
            keypoint_labels=reshaped_keypoints,
            keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (K * D, N)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """
        N = encoded.shape[-1]
        keypoints = np.reshape(
            encoded, (N, -1),
            order='F').transpose(1, 0).reshape(N, self.num_keypoints, -1)
        D = keypoints.shape[-1]
        if D in {2, 3}:
            N, K, _ = keypoints.shape
            normalized_coords = keypoints.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif D in {4, 6}:
            # split coords and sigma if outputs contain output_sigma
            normalized_coords = keypoints[..., :D].copy()
            output_sigma = keypoints[..., D:].copy()

            scores = (1 - output_sigma).mean(axis=-1)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {keypoints.shape[-1]}')

        return normalized_coords, scores
