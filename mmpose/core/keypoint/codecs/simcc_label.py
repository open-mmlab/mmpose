# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

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
                 simcc_type: str = 'gaussian',
                 sigma: float = 6.0,
                 simcc_split_ratio: float = 2.0) -> None:
        super().__init__()

        self.input_size = input_size
        self.simcc_type = simcc_type
        self.sigma = sigma
        self.simcc_split_ratio = simcc_split_ratio

        if self.simcc_type not in {'gaussian', 'one-hot'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `simcc_type` value'
                f'{self.simcc_type}. Should be one of '
                '{"gaussian", "one-hot"}')

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]]:
        """Encoding keypoints into SimCC labels. Note that the original
        keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - simcc_x (np.ndarray): The generated SimCC label in shape
                (K, Wx) where Wx equals to w * `simcc_split_ratio`.
            - simcc_y (np.ndarray): The generated SimCC label in shape
                (K, Wy) where Wx equals to h * `simcc_split_ratio`
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """

        if self.simcc_type == 'gaussian':
            return self._generate_gaussian(keypoints, keypoints_visible)
        elif self.simcc_type == 'one-hot':
            return self._generate_onehot(keypoints, keypoints_visible)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `simcc_type` value'
                f'{self.simcc_type}. Should be one of '
                '{"gaussian", "one-hot"}')

    def decode(self,
               encoded: Tuple[np.ndarray,
                              np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x and y

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

    def _generate_onehot(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encoding keypoints into one-hot SimCC labels without Label
        Smoothing."""

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = w * self.simcc_split_ratio
        H = h * self.simcc_split_ratio

        assert N == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.simdr_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)

        keypoint_weights = np.ones((K, 1), dtype=np.float32)
        keypoint_weights[:, 0] = keypoints_visible[:, 0].copy()

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get center coordinates
            mu_x, mu_y = keypoints_split[n, k].astype(np.int64)

            # detect abnormal coords and assign the weight 0
            if mu_x >= W or mu_y >= H or mu_x < 0 or mu_y < 0:
                keypoint_weights[k] = 0
                continue

        return keypoints_split, keypoint_weights

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label
        Smoothing."""

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = w * self.simcc_split_ratio
        H = h * self.simcc_split_ratio,

        keypoints_split, keypoint_weights = self._generate_onehot(
            keypoints, keypoints_visible)

        target_x = np.zeros((K, W), dtype=np.float32)
        target_y = np.zeros((K, H), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)[:, None]

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            target_x[k] = (np.exp(-((x - mu_x)**2) / (2 * self.sigma**2))) / (
                self.sigma * np.sqrt(np.pi * 2))
            target_y[k] = (np.exp(-((y - mu_y)**2) / (2 * self.sigma**2))) / (
                self.sigma * np.sqrt(np.pi * 2))

        simcc_labels = np.concatenate([target_x, target_y], axis=-1)

        return simcc_labels, keypoint_weights
