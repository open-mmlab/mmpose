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
            (K*3, H, W) where [W, H] is the `heatmap_size`
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

    heatmaps = heatmaps.reshape(K * 3, H, W)

    return heatmaps, keypoint_weights


def generate_displacement_heatmap(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    roots: np.ndarray,
    roots_visible: np.ndarray,
    diagonal_lengths: np.ndarray,
    radius: float,
):
    """Generate displacement heatmaps of keypoints, where each keypoint is
    represented by 3 maps: one pixel-level class label map (1 for keypoint and
    0 for non-keypoint) and 2 pixel-level offset maps for x and y directions
    respectively.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        roots (np.ndarray): Coordinates of instance centers in shape (N, D).
            The displacement fields of each instance will locate around its
            center.
        roots_visible (np.ndarray): Roots visibilities in shape (N,)
        diagonal_lengths (np.ndarray): Diaginal length of the bounding boxes
            of each instance in shape (N,)
        radius (float): The radius factor of the binary label
            map. The positive region is defined as the neighbor of the
            keypoint with the radius :math:`r=radius_factor*max(W, H)`

    Returns:
        tuple:
        - displacements (np.ndarray): The generated displacement map in
            shape (K*2, H, W) where [W, H] is the `heatmap_size`
        - displacement_weights (np.ndarray): The target weights in shape
            (K*2, H, W)
    """
    N, K, _ = keypoints.shape
    W, H = heatmap_size

    displacements = np.zeros((K * 2, H, W), dtype=np.float32)
    displacement_weights = np.zeros((K * 2, H, W), dtype=np.float32)
    instance_size_map = np.zeros((H, W), dtype=np.float32)

    for n in range(N):
        if (roots_visible[n] < 1 or (roots[n, 0] < 0 or roots[n, 1] < 0)
                or (roots[n, 0] >= W or roots[n, 1] >= H)):
            continue

        diagonal_length = diagonal_lengths[n]

        for k in range(K):
            if keypoints_visible[n, k] < 1 or keypoints[n, k, 0] < 0 \
                or keypoints[n, k, 1] < 0 or keypoints[n, k, 0] >= W \
                    or keypoints[n, k, 1] >= H:
                continue

            start_x = max(int(roots[n, 0] - radius), 0)
            start_y = max(int(roots[n, 1] - radius), 0)
            end_x = min(int(roots[n, 0] + radius), W)
            end_y = min(int(roots[n, 1] + radius), H)

            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if displacements[2 * k, y,
                                     x] != 0 or displacements[2 * k + 1, y,
                                                              x] != 0:
                        if diagonal_length > instance_size_map[y, x]:
                            # keep the gt displacement of smaller instance
                            continue

                    displacement_weights[2 * k:2 * k + 2, y,
                                         x] = 1 / diagonal_length
                    displacements[2 * k:2 * k + 2, y,
                                  x] = keypoints[n, k] - [x, y]
                    instance_size_map[y, x] = diagonal_length

    return displacements, displacement_weights
