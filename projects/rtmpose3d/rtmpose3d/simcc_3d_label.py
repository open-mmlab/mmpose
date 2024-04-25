# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from mmpose.codecs.base import BaseKeypointCodec
from mmpose.registry import KEYPOINT_CODECS
from .utils import get_simcc_maximum


@KEYPOINT_CODECS.register_module()
class SimCC3DLabel(BaseKeypointCodec):
    r"""Generate keypoint representation via "SimCC" approach.
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
            The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wx=w*simcc_split_ratio`
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
            The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wy=h*simcc_split_ratio`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.
        use_dark (bool): Whether to use the DARK post processing. Defaults to
            False.
        decode_visibility (bool): Whether to decode the visibility. Defaults
            to False.
        decode_beta (float): The beta value for decoding visibility. Defaults
            to 150.0.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    auxiliary_encode_keys = {'keypoints_3d'}

    label_mapping_table = dict(
        keypoint_x_labels='keypoint_x_labels',
        keypoint_y_labels='keypoint_y_labels',
        keypoint_z_labels='keypoint_z_labels',
        keypoint_weights='keypoint_weights',
        weight_z='weight_z',
        with_z_label='with_z_label')

    instance_mapping_table = dict(
        bbox='bboxes',
        bbox_score='bbox_scores',
        bbox_scale='bbox_scales',
        lifting_target='lifting_target',
        lifting_target_visible='lifting_target_visible',
        camera_param='camera_params',
        root_z='root_z')

    def __init__(self,
                 input_size: Tuple[int, int, int],
                 smoothing_type: str = 'gaussian',
                 sigma: Union[float, int, Tuple[float]] = 6.0,
                 simcc_split_ratio: float = 2.0,
                 label_smooth_weight: float = 0.0,
                 normalize: bool = True,
                 use_dark: bool = False,
                 decode_visibility: bool = False,
                 decode_beta: float = 150.0,
                 root_index: Union[int, Tuple[int]] = 0,
                 z_range: Optional[int] = None,
                 sigmoid_z: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize
        self.use_dark = use_dark
        self.decode_visibility = decode_visibility
        self.decode_beta = decode_beta

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma, sigma])
        else:
            self.sigma = np.array(sigma)

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

        self.root_index = list(root_index) if isinstance(
            root_index, tuple) else [root_index]
        self.z_range = z_range if z_range is not None else 2.1744869
        self.sigmoid_z = sigmoid_z
        self.root_z = [5.14388]

    def encode(self,
               keypoints: np.ndarray,
               keypoints_3d: Optional[np.ndarray] = None,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)
        lifting_target = [None]
        root_z = self.root_z
        with_z_label = False
        if keypoints_3d is not None:
            lifting_target = keypoints_3d.copy()
            root_z = keypoints_3d[..., self.root_index, 2].mean(1)
            keypoints_3d[..., 2] -= root_z
            if self.sigmoid_z:
                keypoints_z = (1 / (1 + np.exp(-(3 * keypoints_3d[..., 2])))
                               ) * self.input_size[2]
            else:
                keypoints_z = (keypoints_3d[..., 2] / self.z_range + 1) * (
                    self.input_size[2] / 2)

            keypoints_3d = np.concatenate([keypoints, keypoints_z[..., None]],
                                          axis=-1)
            x, y, z, keypoint_weights = self._generate_gaussian(
                keypoints_3d, keypoints_visible)
            weight_z = keypoint_weights
            with_z_label = True
        else:
            if keypoints.shape != np.zeros([]).shape:
                keypoints_z = np.ones(
                    (keypoints.shape[0], keypoints.shape[1], 1),
                    dtype=np.float32)
                keypoints = np.concatenate([keypoints, keypoints_z], axis=-1)
                x, y, z, keypoint_weights = self._generate_gaussian(
                    keypoints, keypoints_visible)
            else:
                x, y, z = np.zeros((3, 1), dtype=np.float32)
                keypoint_weights = np.ones((1, ))
            weight_z = np.zeros_like(keypoint_weights)
            with_z_label = False

        encoded = dict(
            keypoint_x_labels=x,
            keypoint_y_labels=y,
            keypoint_z_labels=z,
            lifting_target=lifting_target,
            root_z=root_z,
            keypoint_weights=keypoint_weights,
            weight_z=weight_z,
            with_z_label=[with_z_label])

        return encoded

    def decode(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Decode SimCC labels into 3D keypoints.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis,
            y-axis and z-axis in shape (N, K, Wx), (N, K, Wy) and (N, K, Wz)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = get_simcc_maximum(x, y, z)

        # Unsqueeze the instance dimension for single-instance results
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        keypoints /= self.simcc_split_ratio
        keypoints_simcc = keypoints.copy()
        keypoints_2d = keypoints[..., :2]
        keypoints_z = keypoints[..., 2:3]
        if self.sigmoid_z:
            keypoints_z /= self.input_size[2]
            keypoints_z[keypoints_z <= 0] = 1e-8
            scores[(keypoints_z <= 0).squeeze(-1)] = 0
            keypoints[..., 2:3] = np.log(keypoints_z / (1 - keypoints_z)) / 3
        else:
            keypoints[...,
                      2:3] = (keypoints_z /
                              (self.input_size[-1] / 2) - 1) * self.z_range
        return keypoints_2d, keypoints, keypoints_simcc, scores

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

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label Smoothing
        strategy."""

        N, K, _ = keypoints.shape
        w, h, d = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)
        D = np.around(d * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)
        target_z = np.zeros((N, K, D), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)
        z = np.arange(0, D, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top, near = mu - radius
            right, bottom, far = mu + radius + 1

            if left >= W or top >= H or near >= D or right < 0 or bottom < 0 or far < 0:  # noqa: E501
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y, mu_z = mu

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * self.sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * self.sigma[1]**2))
            target_z[n, k] = np.exp(-((z - mu_z)**2) / (2 * self.sigma[2]**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]
            target_z /= norm_value[2]

        return target_x, target_y, target_z, keypoint_weights

    def _generate_standard(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> tuple[ndarray, ndarray, ndarray, Any]:
        """Encoding keypoints into SimCC labels with Standard Label Smoothing
        strategy.

        Labels will be one-hot vectors if self.label_smooth_weight==0.0
        """

        N, K, _ = keypoints.shape
        w, h, d = self.input_size
        w = np.around(w * self.simcc_split_ratio).astype(int)
        h = np.around(h * self.simcc_split_ratio).astype(int)
        d = np.around(d * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        x = np.zeros((N, K, w), dtype=np.float32)
        y = np.zeros((N, K, h), dtype=np.float32)
        z = np.zeros((N, K, d), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get center coordinates
            mu_x, mu_y, mu_z = keypoints_split[n, k].astype(np.int64)

            # detect abnormal coords and assign the weight 0
            if mu_x >= w or mu_y >= h or mu_x < 0 or mu_y < 0:
                keypoint_weights[n, k] = 0
                continue

            if self.label_smooth_weight > 0:
                x[n, k] = self.label_smooth_weight / (w - 1)
                y[n, k] = self.label_smooth_weight / (h - 1)
                z[n, k] = self.label_smooth_weight / (d - 1)

            x[n, k, mu_x] = 1.0 - self.label_smooth_weight
            y[n, k, mu_y] = 1.0 - self.label_smooth_weight
            z[n, k, mu_z] = 1.0 - self.label_smooth_weight

        return x, y, z, keypoint_weights
