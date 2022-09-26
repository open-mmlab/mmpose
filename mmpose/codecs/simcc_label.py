# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np

from mmpose.codecs.utils import get_simcc_maximum
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
        smoothing_type (str): The SimCC label smoothing strategy. Options are
        ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (str): The sigma value in the Gaussian SimCC label. Defaults to
            6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 smoothing_type: str = 'gaussian',
                 sigma: float = 6.0,
                 simcc_split_ratio: float = 2.0,
                 label_smooth_weight: float = 0.0,
                 normalize: bool = True) -> None:
        super().__init__()

        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.sigma = sigma
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize

        if self.smoothing_type not in {'gaussian', 'standard'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `smoothing_type` value'
                f'{self.smoothing_type}. Should be one of '
                '{"gaussian", "standard"}')

        if self.smoothing_type == 'gaussian' and self.label_smooth_weight > 0:
            raise ValueError('Attribute `label_smooth_weight` is only '
                             'used for `standard` mode.')

        if self.label_smooth_weight < 0.0 or self.label_smooth_weight > 1.0:
            raise ValueError('`label_smooth_weight` should be in range [0, 1]')

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]]:
        """Encoding keypoints into SimCC labels. Note that the original
        keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - simcc_x (np.ndarray): The generated SimCC label for x-axis.
                The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wx=w*simcc_split_ratio`
            - simcc_y (np.ndarray): The generated SimCC label for y-axis.
                The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wy=h*simcc_split_ratio`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.smoothing_type == 'gaussian':
            return self._generate_gaussian(keypoints, keypoints_visible)
        elif self.smoothing_type == 'standard':
            return self._generate_standard(keypoints, keypoints_visible)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `smoothing_type` value'
                f'{self.smoothing_type}. Should be one of '
                '{"gaussian", "standard"}')

    def decode(self,
               encoded: Tuple[np.ndarray,
                              np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis
                and y-axis

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        simcc_x, simcc_y = encoded
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)

        keypoints /= self.simcc_split_ratio

        # Unsqueeze the instance dimension for single-instance results
        if len(keypoints) == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        return keypoints, scores

    def _map_coordinates(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mapping keypoint coordinates into SimCC space."""

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = keypoints_visible.copy()

        return keypoints_split, keypoint_weights

    def _generate_standard(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Standard Label Smoothing
        strategy.

        Labels will be one-hot vectors if self.label_smooth_weight==0.0
        """

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get center coordinates
            mu_x, mu_y = keypoints_split[n, k].astype(np.int64)

            # detect abnormal coords and assign the weight 0
            if mu_x >= W or mu_y >= H or mu_x < 0 or mu_y < 0:
                keypoint_weights[n, k] = 0
                continue

            if self.label_smooth_weight > 0:
                target_x[n, k] = self.label_smooth_weight / (W - 1)
                target_y[n, k] = self.label_smooth_weight / (H - 1)

            target_x[n, k, mu_x] = 1.0 - self.label_smooth_weight
            target_y[n, k, mu_y] = 1.0 - self.label_smooth_weight

        return target_x, target_y, keypoint_weights

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label Smoothing
        strategy."""

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

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

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * self.sigma**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * self.sigma**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value
            target_y /= norm_value

        return target_x, target_y, keypoint_weights
