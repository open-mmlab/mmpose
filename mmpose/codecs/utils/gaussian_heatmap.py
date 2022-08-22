# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import numpy as np


def generate_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value of the Gaussian heatmap

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]
    x0 = y0 = gaussian_size // 2

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        # get gaussian center coordinates
        mu = (keypoints[n, k] + 0.5).astype(np.int64)

        # check that the gaussian has in-bounds part
        left, top = (mu - radius).astype(np.int64)
        right, bottom = (mu + radius + 1).astype(np.int64)

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        # The gaussian is not normalized,
        # we want the center value to equal 1
        gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

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

        heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
        gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

        _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps, keypoint_weights


def generate_unbiased_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `Dark Pose`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    x = np.arange(0, W, 1, dtype=np.float32)
    y = np.arange(0, H, 1, dtype=np.float32)[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = keypoints[n, k]
        # check that the gaussian has in-bounds part
        left, top = mu - radius
        right, bottom = mu + radius + 1

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        gaussian = np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2))

        _ = np.maximum(gaussian, heatmaps[k], out=heatmaps[k])

    return heatmaps, keypoint_weights


def generate_udp_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `UDP`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value of the Gaussian heatmap

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = (keypoints[n, k] + 0.5).astype(np.int64)
        # check that the gaussian has in-bounds part
        left, top = (mu - radius).astype(np.int64)
        right, bottom = (mu + radius + 1).astype(np.int64)

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        mu_ac = keypoints[n, k]
        x0 = y0 = gaussian_size // 2
        x0 += mu_ac[0] - mu[0]
        y0 += mu_ac[1] - mu[1]
        gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

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

        heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
        gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

        _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps, keypoint_weights
