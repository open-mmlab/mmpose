# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (gaussian_blur, generate_gaussian_heatmaps,
                    get_heatmap_maximum)
from .utils.gaussian_heatmap import generate_unbiased_gaussian_heatmaps


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

        if self.unbiased:
            # Alleviate biased coordinate
            # Apply Gaussian distribution modulation.
            heatmaps = gaussian_blur(heatmaps, kernel=self.blur_kernel_size)
            heatmaps = np.log(np.maximum(heatmaps, 1e-10))
            for k in range(K):
                keypoints[k] = self._taylor_decode(
                    heatmap=heatmaps[k], keypoint=keypoints[k])
        else:
            # Add +/-0.25 shift to the predicted locations for higher acc.
            for k in range(K):
                heatmap = heatmaps[k]
                px = int(keypoints[k, 0])
                py = int(keypoints[k, 1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    keypoints[k] += np.sign(diff) * 0.25

        # Unsqueeze the instance dimension for single-instance results
        # and restore the keypoint scales
        keypoints = keypoints[None] * self.scale_factor
        scores = scores[None]

        return keypoints, scores

    @staticmethod
    def _taylor_decode(heatmap: np.ndarray,
                       keypoint: np.ndarray) -> np.ndarray:
        """Distribution aware coordinate decoding for a single keypoint.

        Note:
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmap (np.ndarray[H, W]): Heatmap of a particular keypoint type.
            keypoint (np.ndarray[2,]): Coordinates of the predicted keypoint.

        Returns:
            np.ndarray[2,]: Updated coordinates.
        """
        H, W = heatmap.shape[:2]
        px, py = int(keypoint[0]), int(keypoint[1])
        if 1 < px < W - 2 and 1 < py < H - 2:
            dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
            dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
            dxx = 0.25 * (
                heatmap[py][px + 2] - 2 * heatmap[py][px] +
                heatmap[py][px - 2])
            dxy = 0.25 * (
                heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
                heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
            dyy = 0.25 * (
                heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
                heatmap[py - 2 * 1][px])
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                keypoint += offset
        return keypoint
