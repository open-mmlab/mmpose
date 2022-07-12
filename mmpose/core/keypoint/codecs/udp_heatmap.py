# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODEC
from .base import BaseKeypointCodec
from .utils import get_heatmap_maximum


@KEYPOINT_CODEC.register_module()
class UDPHeatmap(BaseKeypointCodec):

    def __init__(self,
                 image_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: Optional[float] = None,
                 radius_factor: Optional[float] = None,
                 heatmap_type: str = 'gaussian',
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size

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
                (N, C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
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
            encoded (np.ndarray): Heatmaps in shape (N, K, H, W)

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
            N, _K, H, W = heatmaps.shape
            K = _K // 3
            for n, k in product(range(N), range(_K)):
                if k % 3 == 0:
                    # for classification map
                    ks = 2 * self.blur_kernel_size + 1
                else:
                    # for offset map
                    ks = self.blur_kernel_size
                cv2.GaussianBlur(heatmaps[n, k], (ks, ks), 0, heatmaps[n, k])

            # valid radius
            radius = self.radius_factor * max(W, H)

            x_offset = heatmaps[:, 1::3].flatten() * radius
            y_offset = heatmaps[:, 2::3].flatten() * radius
            keypoints, scores = get_heatmap_maximum(heatmaps=heatmaps[:, ::3])
            index = keypoints[..., 0] + keypoints[..., 1] * W
            index += W * H * np.arange(0, N * K)
            index = index.astype(int).reshape(N, K)
            keypoints += np.stack((x_offset[index], y_offset[index]), axis=-1)

        return keypoints, scores

    def _encode_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size
        image_size = np.array(self.image_size)
        feat_stride = image_size / [W - 1, H - 1]

        heatmaps = np.zeros((N, K, H, W), dtype=np.float32)
        keypoint_weights = np.ones((N, K), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                keypoint_weights[n, k] = 0
                continue

            mu = (keypoints[n, k] / feat_stride + 0.5).astype(np.int64)
            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_ac = keypoints[n, k] / feat_stride
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

            heatmaps[n, k, h_y1:h_y2, h_x1:h_x2] = gaussian[g_y1:g_y2,
                                                            g_x1:g_x2]

        return heatmaps, keypoint_weights

    def _encode_combined(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size
        image_size = np.array(self.image_size)
        feat_stride = image_size / [W - 1, H - 1]

        heatmaps = np.zeros((N, K, 3, H, W), dtype=np.float32)
        keypoint_weights = np.ones((N, K), dtype=np.float32)

        # xy grid
        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)[:, None]

        # positive area radius in the classification map
        radius = self.radius_factor * max(W, H)

        for n, k in product(range(N), range(K)):
            if keypoints_visible[n, k] < 0.5:
                keypoint_weights[n, k] = 0
                continue

            mu = keypoints[n, k] / feat_stride

            x_offset = (mu[0] - x) / radius
            y_offset = (mu[1] - y) / radius

            heatmaps[n, k, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1.,
                                         0.)
            heatmaps[n, k, 1] = x_offset
            heatmaps[n, k, 2] = y_offset

        # keep only valid region in offset maps
        heatmaps[:, :, 1:] *= heatmaps[:, :, :1]
        heatmaps = heatmaps.reshape(N, K * 3, H, W)

        return heatmaps, keypoint_weights

    @staticmethod
    def _postprocess_dark_udp(heatmaps: np.ndarray, keypoints: np.ndarray,
                              kernel_size: int) -> np.ndarray:
        """Distribution aware post-processing for UDP.

        Args:
            heatmaps (np.ndarray): Heatmaps in shape (N, K, H, W)
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
            kernel_size (int): The Gaussian blur kernel size of the heatmap
                modulation

        Returns:
            np.ndarray: Post-processed keypoint coordinates
        """
        N, K, H, W = heatmaps.shape

        for n, k in product(range(N), range(K)):
            cv2.GaussianBlur(heatmaps[n, k], (kernel_size, kernel_size), 0,
                             heatmaps[n, k])

        np.clip(heatmaps, 0.001, 50., heatmaps)
        np.log(heatmaps, heatmaps)
        heatmaps_pad = np.pad(
            heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge').flatten()

        index = k[..., 0] + 1 + (k[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, N * K).reshape(-1, K)
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
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints -= np.einsum('ijmn,ijnk->ijmk', hessian,
                               derivative).squeeze()
        return keypoints
