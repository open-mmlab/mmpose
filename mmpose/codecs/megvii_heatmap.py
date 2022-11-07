# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import gaussian_blur, get_heatmap_maximum


@KEYPOINT_CODECS.register_module()
class MegviiHeatmap(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "Megvii" approach. See `MSPN`_
    (2019) and `CPN`_ (2018) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        kernel_size (tuple): The kernel size of the heatmap gaussian in
            [ks_x, ks_y]

    .. _`MSPN`: https://arxiv.org/abs/1901.00148
    .. _`CPN`: https://arxiv.org/abs/1711.07319
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        kernel_size: int,
    ) -> None:

        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.kernel_size = kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size

        assert N == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get center coordinates
            kx, ky = (keypoints[n, k] / self.scale_factor).astype(np.int64)
            if kx < 0 or kx >= W or ky < 0 or ky >= H:
                keypoint_weights[n, k] = 0
                continue

            heatmaps[k, ky, kx] = 1.
            kernel_size = (self.kernel_size, self.kernel_size)
            heatmaps[k] = cv2.GaussianBlur(heatmaps[k], kernel_size, 0)

            # normalize the heatmap
            heatmaps[k] = heatmaps[k] / heatmaps[k, ky, kx] * 255.

        encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (K, D)
            - scores (np.ndarray): The keypoint scores in shape (K,). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = gaussian_blur(encoded.copy(), self.kernel_size)
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        for k in range(K):
            heatmap = heatmaps[k]
            px = int(keypoints[k, 0])
            py = int(keypoints[k, 1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                keypoints[k] += (np.sign(diff) * 0.25 + 0.5)

        scores = scores / 255.0 + 0.5

        # Unsqueeze the instance dimension for single-instance results
        # and restore the keypoint scales
        keypoints = keypoints[None] * self.scale_factor
        scores = scores[None]

        return keypoints, scores
