# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import numpy as np


def generate_offset_heatmap(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    radius_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate offset heatmaps of keypoints, where each keypoint is
    represented by 3 maps: one pixel-level class label map (1 for keypoint and
    0 for non-keypoint) and 2 pixel-level offset maps for x and y directions
    respectively.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        radius_factor (float): The radius factor of the binary label
            map. The positive region is defined as the neighbor of the
            keypoint with the radius :math:`r=radius_factor*max(W, H)`

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

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, 3, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # xy grid
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)[:, None]

    # positive area radius in the classification map
    radius = radius_factor * max(W, H)

    for n, k in product(range(N), range(K)):
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = keypoints[n, k]

        x_offset = (mu[0] - x) / radius
        y_offset = (mu[1] - y) / radius

        heatmaps[k, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1., 0.)
        heatmaps[k, 1] = x_offset
        heatmaps[k, 2] = y_offset

    # keep only valid region in offset maps
    heatmaps[:, 1:] *= heatmaps[:, :1]
    heatmaps = heatmaps.reshape(K * 3, H, W)

    return heatmaps, keypoint_weights
