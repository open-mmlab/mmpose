# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import cv2
import numpy as np


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - locs (np.ndarray[K, 2]): locations of maximum heatmap responses
        - vals (np.ndarray[K, 1]): values of maximum heatmap responses
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3, 'heatmaps should be 3-ndim'

    K, H, W = heatmaps.shape
    heatmap_flatten = heatmaps.reshape(K, -1)
    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmap_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmap_flatten, axis=1)
    locs[vals <= 0.] = -1

    return locs, vals


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps
