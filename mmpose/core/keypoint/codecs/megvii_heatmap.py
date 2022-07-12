# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODEC
from .base import BaseKeypointCodec
from .utils import gaussian_blur, get_heatmap_maximum


@KEYPOINT_CODEC.register_module()
class MegviiHeatmap(BaseKeypointCodec):

    def __init__(
        self,
        image_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        kernel_size: int,
    ) -> None:

        super().__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.kernel_size = kernel_size

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size
        image_size = np.array(self.image_size)
        feat_stride = image_size / [W, H]

        heatmaps = np.zeros((N, K, H, W), dtype=np.float32)
        keypoint_weights = np.ones((N, K), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                keypoint_weights[n, k] = 0
                continue

            # get center coordinates
            kx, ky = (keypoints[n, k] / feat_stride).astype(np.int64)
            if kx < 0 or kx >= W or ky < 0 or ky >= H:
                keypoint_weights[n, k] = 0
                continue

            heatmaps[n, k, ky, kx] = 1.
            heatmaps[n, k] = cv2.GaussianBlur(heatmaps[n, k], self.kernel_size,
                                              0)

            # normalize the heatmap
            heatmaps[n, k] = heatmaps[n, k] / heatmaps[n, k, ky, kx] * 255.

        return heatmaps, keypoint_weights

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        heatmaps = gaussian_blur(encoded.copy(), self.kernel_size)
        N, K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        for n, k in product(range(N), range(K)):
            heatmap = heatmaps[n, k]
            px = int(keypoints[n, k, 0])
            py = int(keypoints[n, k, 1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                keypoints[n, k] += (np.sign(diff) * 0.25 + 0.5)

        scores = scores / 255.0 + 0.5

        return keypoints, scores
