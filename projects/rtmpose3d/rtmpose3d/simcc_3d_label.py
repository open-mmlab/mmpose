# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from mmpose.codecs.base import BaseKeypointCodec
from mmpose.codecs.utils.refinement import refine_simcc_dark
from mmpose.registry import KEYPOINT_CODECS
from .utils import get_simcc_maximum


@KEYPOINT_CODECS.register_module()
class SimCC3DLabel(BaseKeypointCodec):
    r"""Generate keypoint representation via "SimCC" approach.
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    We generate the SimCC label for 3D keypoint estimation.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
        - keypoint_z_labels (np.ndarray): The generated SimCC label for z-axis.
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.
        use_dark (bool): Whether to use the DARK post processing. Defaults to
            False.
        root_index (int | tuple): The index of the root keypoint. Defaults to
            0.
        z_range (float): The range of the z-axis. Defaults to None.

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
                 sigma: Union[float, int, Tuple[float]] = 6.0,
                 simcc_split_ratio: float = 2.0,
                 normalize: bool = True,
                 use_dark: bool = False,
                 root_index: Union[int, Tuple[int]] = 0,
                 z_range: Optional[int] = None) -> None:
        super().__init__()

        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize
        self.use_dark = use_dark

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma, sigma])
        else:
            self.sigma = np.array(sigma)

        self.root_index = list(root_index) if isinstance(
            root_index, tuple) else [root_index]

        # Mean value of the root z-axis of datasets
        # These values are statistics from the training set
        self.root_z = [5.14388]
        self.z_range = z_range if z_range is not None else 2.1744869

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
                # placeholder for empty keypoints
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

        if self.use_dark:
            x_blur = int((self.sigma[0] * 20 - 7) // 3)
            y_blur = int((self.sigma[1] * 20 - 7) // 3)
            z_blur = int((self.sigma[2] * 20 - 7) // 3)
            x_blur -= int((x_blur % 2) == 0)
            y_blur -= int((y_blur % 2) == 0)
            z_blur -= int((z_blur % 2) == 0)
            keypoints[:, :, 0] = refine_simcc_dark(keypoints[:, :, 0], x,
                                                   x_blur)
            keypoints[:, :, 1] = refine_simcc_dark(keypoints[:, :, 1], y,
                                                   y_blur)
            keypoints[:, :, 2] = refine_simcc_dark(keypoints[:, :, 2], z,
                                                   z_blur)

        keypoints /= self.simcc_split_ratio
        keypoints_simcc = keypoints.copy()
        keypoints_z = keypoints[..., 2:3]

        keypoints[..., 2:3] = (keypoints_z /
                               (self.input_size[-1] / 2) - 1) * self.z_range
        return keypoints, keypoints_simcc, scores

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
