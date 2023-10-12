# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np


def generate_3d_gaussian_heatmaps(
    heatmap_size: Tuple[int, int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
    image_size: Tuple[int, int],
    heatmap3d_depth_bound: float = 400.0,
    joint_indices: Optional[list] = None,
    max_bound: float = 1.0,
    use_different_joint_weights: bool = False,
    dataset_keypoint_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3d gaussian heatmaps of keypoints.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H, D]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float or List[float]): A list of sigma values of the Gaussian
            heatmap for each instance. If sigma is given as a single float
            value, it will be expanded into a tuple
        image_size (Tuple[int, int]): Size of input image.
        heatmap3d_depth_bound (float): Boundary for 3d heatmap depth.
            Default: 400.0.
        joint_indices (List[int], optional): Indices of joints used for heatmap
            generation. If None (default) is given, all joints will be used.
            Default: ``None``.
        max_bound (float): The maximal value of heatmap. Default: 1.0.
        use_different_joint_weights (bool): Whether to use different joint
            weights. Default: ``False``.
        dataset_keypoint_weights (np.ndarray, optional): Keypoints weight in
            shape (K, ).

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K * D, H, W) where [W, H, D] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """

    W, H, D = heatmap_size

    # select the joints used for target generation
    if joint_indices is not None:
        keypoints = keypoints[:, joint_indices, ...]
        keypoints_visible = keypoints_visible[:, joint_indices, ...]
    N, K, _ = keypoints.shape

    heatmaps = np.zeros([K, D, H, W], dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    if isinstance(sigma, (int, float)):
        sigma = (sigma, ) * N

    for n in range(N):
        # 3-sigma rule
        radius = sigma[n] * 3

        # joint location in heatmap coordinates
        mu_x = keypoints[n, :, 0] * W / image_size[0]  # (K, )
        mu_y = keypoints[n, :, 1] * H / image_size[1]
        mu_z = (keypoints[n, :, 2] / heatmap3d_depth_bound + 0.5) * D

        keypoint_weights[n, ...] = keypoint_weights[n, ...] * (mu_z >= 0) * (
            mu_z < D)
        if use_different_joint_weights:
            keypoint_weights[
                n] = keypoint_weights[n] * dataset_keypoint_weights
        # xy grid
        gaussian_size = 2 * radius + 1

        # get neighboring voxels coordinates
        x = y = z = np.arange(gaussian_size, dtype=np.float32) - radius
        zz, yy, xx = np.meshgrid(z, y, x)

        xx = np.expand_dims(xx, axis=0)
        yy = np.expand_dims(yy, axis=0)
        zz = np.expand_dims(zz, axis=0)
        mu_x = np.expand_dims(mu_x, axis=(-1, -2, -3))
        mu_y = np.expand_dims(mu_y, axis=(-1, -2, -3))
        mu_z = np.expand_dims(mu_z, axis=(-1, -2, -3))

        xx, yy, zz = xx + mu_x, yy + mu_y, zz + mu_z
        local_size = xx.shape[1]

        # round the coordinates
        xx = xx.round().clip(0, W - 1)
        yy = yy.round().clip(0, H - 1)
        zz = zz.round().clip(0, D - 1)

        # compute the target value near joints
        gaussian = np.exp(-((xx - mu_x)**2 + (yy - mu_y)**2 + (zz - mu_z)**2) /
                          (2 * sigma[n]**2))

        # put the local target value to the full target heatmap
        idx_joints = np.tile(
            np.expand_dims(np.arange(K), axis=(-1, -2, -3)),
            [1, local_size, local_size, local_size])
        idx = np.stack([idx_joints, zz, yy, xx],
                       axis=-1).astype(int).reshape(-1, 4)

        heatmaps[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = np.maximum(
            heatmaps[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]],
            gaussian.reshape(-1))

    heatmaps = (heatmaps * max_bound).reshape(-1, H, W)

    return heatmaps, keypoint_weights


def generate_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float or List[float]): A list of sigma values of the Gaussian
            heatmap for each instance. If sigma is given as a single float
            value, it will be expanded into a tuple

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

    if isinstance(sigma, (int, float)):
        sigma = (sigma, ) * N

    for n in range(N):
        # 3-sigma rule
        radius = sigma[n] * 3

        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]
        x0 = y0 = gaussian_size // 2

        for k in range(K):
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
            gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma[n]**2))

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

            _ = np.maximum(
                heatmap_region, gaussian_regsion, out=heatmap_region)

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
