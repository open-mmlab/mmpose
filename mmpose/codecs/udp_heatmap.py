# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_offset_heatmap, generate_udp_gaussian_heatmaps,
                    get_heatmap_maximum)


@KEYPOINT_CODECS.register_module()
class UDPHeatmap(BaseKeypointCodec):
    r"""Generate keypoint heatmaps by Unbiased Data Processing (UDP).
    See the paper: `The Devil is in the Details: Delving into Unbiased Data
    Processing for Human Pose Estimation`_ by Huang et al (2020) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        heatmap_type (str): The heatmap type to encode the keypoitns. Options
            are:

            - ``'gaussian'``: Gaussian heatmap
            - ``'combined'``: Combination of a binary label map and offset
                maps for X and Y axes.

        sigma (float): The sigma value of the Gaussian heatmap when
            ``heatmap_type=='gaussian'``. Defaults to 2.0
        radius_factor (float): The radius factor of the binary label
            map when ``heatmap_type=='combined'``. The positive region is
            defined as the neighbor of the keypoit with the radius
            :math:`r=radius_factor*max(W, H)`. Defaults to 0.0546875
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. Defaults to 11

    .. _`The Devil is in the Details: Delving into Unbiased Data Processing for
    Human Pose Estimation`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 heatmap_type: str = 'gaussian',
                 sigma: float = 2.,
                 radius_factor: float = 0.0546875,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = ((np.array(input_size) - 1) /
                             (np.array(heatmap_size) - 1)).astype(np.float32)

        if self.heatmap_type not in {'gaussian', 'combined'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - heatmap (np.ndarray): The generated heatmap in shape
                (C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """
        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.heatmap_type == 'gaussian':
            heatmaps, keypoint_weights = generate_udp_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        elif self.heatmap_type == 'combined':
            heatmaps, keypoint_weights = generate_offset_heatmap(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                radius_factor=self.radius_factor)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')

        return heatmaps, keypoint_weights

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()

        if self.heatmap_type == 'gaussian':
            keypoints, scores = get_heatmap_maximum(heatmaps)
            keypoints = self._postprocess_dark_udp(heatmaps, keypoints,
                                                   self.blur_kernel_size)
        elif self.heatmap_type == 'combined':
            _K, H, W = heatmaps.shape
            K = _K // 3
            for k in range(_K):
                if k % 3 == 0:
                    # for classification map
                    ks = 2 * self.blur_kernel_size + 1
                else:
                    # for offset map
                    ks = self.blur_kernel_size
                cv2.GaussianBlur(heatmaps[k], (ks, ks), 0, heatmaps[k])

            # valid radius
            radius = self.radius_factor * max(W, H)

            x_offset = heatmaps[1::3].flatten() * radius
            y_offset = heatmaps[2::3].flatten() * radius
            keypoints, scores = get_heatmap_maximum(heatmaps=heatmaps[::3])
            index = (keypoints[..., 0] + keypoints[..., 1] * W).flatten()
            index += W * H * np.arange(0, K)
            index = index.astype(int)
            keypoints += np.stack((x_offset[index], y_offset[index]), axis=-1)

        # Unsqueeze the instance dimension for single-instance results
        W, H = self.heatmap_size
        keypoints = keypoints[None] / [W - 1, H - 1] * self.input_size
        scores = scores[None]

        return keypoints, scores

    @staticmethod
    def _postprocess_dark_udp(heatmaps: np.ndarray, keypoints: np.ndarray,
                              kernel_size: int) -> np.ndarray:
        """Distribution aware post-processing for UDP.

        Args:
            heatmaps (np.ndarray): Heatmaps in shape (K, H, W)
            keypoints (np.ndarray): Keypoint coordinates in shape (K, D)
            kernel_size (int): The Gaussian blur kernel size of the heatmap
                modulation

        Returns:
            np.ndarray: Post-processed keypoint coordinates
        """
        K, H, W = heatmaps.shape

        for k in range(K):
            cv2.GaussianBlur(heatmaps[k], (kernel_size, kernel_size), 0,
                             heatmaps[k])

        np.clip(heatmaps, 0.001, 50., heatmaps)
        np.log(heatmaps, heatmaps)
        heatmaps_pad = np.pad(
            heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

        index = keypoints[..., 0] + 1 + (keypoints[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints -= np.einsum('imn,ink->imk', hessian, derivative).squeeze()

        return keypoints
