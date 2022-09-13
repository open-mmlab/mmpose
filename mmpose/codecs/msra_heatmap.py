# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module()
class MSRAHeatmap(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "MSRA" approach. See the paper:
    `Simple Baselines for Human Pose Estimation and Tracking`_ by Xiao et al
    (2018) for details.

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

    .. _`Simple Baselines for Human Pose Estimation and Tracking`:
        https://arxiv.org/abs/1804.06208
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.unbiased = unbiased

        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

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
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.unbiased:
            heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

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
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores
