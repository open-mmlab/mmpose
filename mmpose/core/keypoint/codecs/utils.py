# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import cv2
import numpy as np


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - maxlocs (np.ndarray[N, K, 2]): locations of maximum heatmap responses
        - maxvals (np.ndarray[N, K, 1]): values of maximum heatmap responses
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    maxlocs = np.argmax(heatmaps_reshaped, axis=2)
    maxvals = np.amax(heatmaps_reshaped, axis=2)
    maxlocs[maxvals <= 0.0] = -1

    maxlocs = np.stack((maxlocs % W, maxlocs // W), axis=-1)

    return maxlocs, maxvals


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, H, W = heatmaps.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(heatmaps[n, k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[n, k] = dr[border:-border, border:-border].copy()
        heatmaps[n, k] *= origin_max / np.max(heatmaps[n, k])
    return heatmaps
