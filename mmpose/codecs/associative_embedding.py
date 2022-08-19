# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Any, Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .functional import (generate_gaussian_heatmaps,
                         generate_udp_gaussian_heatmaps)


@KEYPOINT_CODECS.register_module()
class AssociativeEmbedding(BaseKeypointCodec):
    """Encode/decode keypoints with the method introduced in "Associative
    Embedding". This is an asymmetric codec, where the keypoints are
    represented as gaussian heatmaps and position indices during encoding, and
    reostred from predicted heatmaps and embedding vectors.

    See the paper `Associative Embedding: End-to-End Learning for Joint
    Detection and Grouping`_ by Newell et al (2017) for details

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
            encoding. See `Dark Pose`_ for details. Defaults to ``False``
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. The kernel size and sigma should follow
            the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
            Defaults to 11

    .. _`Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping`: https://arxiv.org/abs/1611.05424
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: Optional[float] = None,
                 use_udp: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.use_udp = use_udp

        if sigma is None:
            sigma = (heatmap_size[0] * heatmap_size[1])**0.5 / 64
        self.sigma = sigma

        if self.use_udp:
            self.scale_factor = ((np.array(heatmap_size) - 1) /
                                 (np.array(heatmap_size) - 1)).astype(
                                     np.float32)
        else:
            self.scale_factor = (np.array(input_size) /
                                 heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> Any:
        """Encode keypoints into heatmaps and position indices. Note that the
        original keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_indices (np.ndarray): The keypoint position indices
                in shape (N, K, 2). Each keypoint's index is [i, v], where i
                is the position index in the heatmap (:math:`i=y*w+x`) and v
                is the visibility
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor

        if self.use_udp:
            heatmaps, keypoints_weights = generate_udp_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoints_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        keypoint_indices = self._encode_keypoint_indices(
            heatmap_size=self.heatmap_size,
            keypoints=_keypoints,
            keypoints_visible=keypoints_visible)

        return heatmaps, keypoint_indices, keypoints_weights

    def _encode_keypoint_indices(self, heatmap_size: Tuple[int, int],
                                 keypoints: np.ndarray,
                                 keypoints_visible: np.ndarray) -> np.ndarray:
        w, h = heatmap_size
        N, K, _ = keypoints.shape
        keypoint_indices = np.zeros((N, K, 2), dtype=np.int64)

        for n, k in product(N, K):
            x, y = (keypoints[n, k] + 0.5).astype(np.int64)
            index = y * w + x
            vis = (keypoints_visible[n, k] > 0.5 and 0 <= x < w and 0 <= y < h)
            keypoint_indices[n, k] = [index, vis]

        return keypoint_indices

    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        pass
