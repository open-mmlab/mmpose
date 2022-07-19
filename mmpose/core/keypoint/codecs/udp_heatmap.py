# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import get_heatmap_maximum


@KEYPOINT_CODECS.register_module()
class UDPHeatmap(BaseKeypointCodec):
    r"""Generate keypoint heatmap via "UDP" approach. See the paper: `The Devil
    is in the Details: Delving into Unbiased Data Processing for Human Pose
    Estimation`_ by Huang et al (2020) for more details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: C
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

        w, h = input_size
        W, H = heatmap_size
        self.scale_factor = (np.array([w - 1, h - 1]) /
                             [W - 1, H - 1]).astype(np.float32)

        if self.heatmap_type == 'gaussian':
            assert self.sigma is not None, (
                'The parameter `sigma` should be provided if '
                '`heatmap_type=="gaussian"`')
        elif self.heatmap_type == 'combined':
            assert self.sigma is not None, (
                'The parameter `radius_factor` should be provided if '
                '`heatmap_type=="combined"`')
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_typee}. Should be one of '
                '{"gaussian", "combined"}')

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode keypoints into heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
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
        if self.heatmap_type == 'gaussian':
            return self._encode_gaussian(keypoints, keypoints_visible)
        elif self.heatmap_type == 'combined':
            return self._encode_combined(keypoints, keypoints_visible)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_typee}. Should be one of '
                '{"gaussian", "combined"}')

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get keypoint coordinates from heatmaps.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, C)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction.
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
            index = keypoints[..., 0] + keypoints[..., 1] * W
            index += W * H * np.arange(0, K)
            index = index.astype(int)
            keypoints += np.stack((x_offset[index], y_offset[index]), axis=-1)

        # Unsqueeze the instance dimension for single-instance results
        keypoints = keypoints[None] * self.scale_factor
        scores = scores[None]

        return keypoints, scores

    def _encode_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode keypoints into Gaussian heatmaps."""

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size

        assert N == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        keypoint_weights = np.ones(K, dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]

        for k in range(K):
            # skip unlabled keypoints
            if keypoints_visible[0, k] < 0.5:
                keypoint_weights[k] = 0
                continue

            mu = (keypoints[0, k] / self.scale_factor + 0.5).astype(np.int64)
            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[k] = 0
                continue

            mu_ac = keypoints[0, k] / self.scale_factor
            x0 = y0 = gaussian_size // 2
            x0 += mu_ac[0] - mu[0]
            y0 += mu_ac[1] - mu[1]
            gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) /
                              (2 * self.sigma**2))

            # valid range in gaussian
            g_x1 = max(0, -left)
            g_x2 = min(W, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(H, bottom) - top

            # valid range in heatmap
            h_x1 = max(0, left)
            h_x2 = min(W, right)
            h_y1 = max(0, top)
            h_y2 = min(H, bottom)

            heatmaps[k, h_y1:h_y2, h_x1:h_x2] = gaussian[g_y1:g_y2, g_x1:g_x2]

        return heatmaps, keypoint_weights

    def _encode_combined(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode keypoints into Combined label and offset maps."""

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size

        assert N == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        heatmaps = np.zeros((K, 3, H, W), dtype=np.float32)
        keypoint_weights = np.ones(K, dtype=np.float32)

        # xy grid
        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)[:, None]

        # positive area radius in the classification map
        radius = self.radius_factor * max(W, H)

        for k in range(K):
            if keypoints_visible[0, k] < 0.5:
                keypoint_weights[k] = 0
                continue

            mu = keypoints[0, k] / self.scale_factor

            x_offset = (mu[0] - x) / radius
            y_offset = (mu[1] - y) / radius

            heatmaps[k, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1., 0.)
            heatmaps[k, 1] = x_offset
            heatmaps[k, 2] = y_offset

        # keep only valid region in offset maps
        heatmaps[:, 1:] *= heatmaps[:, :1]
        heatmaps = heatmaps.reshape(K * 3, H, W)

        return heatmaps, keypoint_weights

    @staticmethod
    def _postprocess_dark_udp(heatmaps: np.ndarray, keypoints: np.ndarray,
                              kernel_size: int) -> np.ndarray:
        """Distribution aware post-processing for UDP.

        Args:
            heatmaps (np.ndarray): Heatmaps in shape (K, H, W)
            keypoints (np.ndarray): Keypoint coordinates in shape (K, C)
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
