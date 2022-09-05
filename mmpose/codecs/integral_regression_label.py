# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel


@KEYPOINT_CODECS.register_module()
class IntegralRegressionLabel(BaseKeypointCodec):
    """Generate keypoint coordinates and heatmaps. See the paper: `IPR`_ by
    Xiao et al(2018).

    Note:

        - input image size: [w, h]

    Args:
        input_size (tuple): Input image size in [w, h]


    .. _`IPR`: https://arxiv.org/abs/1711.08229
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()

        self.heatmap_codec = MSRAHeatmap(input_size, heatmap_size, sigma,
                                         unbiased, blur_kernel_size)
        self.keypoint_codec = RegressionLabel(input_size)

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encoding keypoints to regression labels and heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - reg_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        heatmaps, keypoint_weights = self.heatmap_codec.encode(
            keypoints, keypoints_visible)
        reg_labels, keypoint_weights = self.keypoint_codec.encode(
            keypoints, keypoint_weights)

        return heatmaps, reg_labels, keypoint_weights

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = self.keypoint_codec.decode(encoded)

        return keypoints, scores