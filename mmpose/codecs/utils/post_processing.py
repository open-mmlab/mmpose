# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def get_simcc_normalized(batch_pred_simcc, sigma=None):
    """Normalize the predicted SimCC.

    Args:
        batch_pred_simcc (torch.Tensor): The predicted SimCC.
        sigma (float): The sigma of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized SimCC.
    """
    B, K, _ = batch_pred_simcc.shape

    # Scale and clamp the tensor
    if sigma is not None:
        batch_pred_simcc = batch_pred_simcc / (sigma * np.sqrt(np.pi * 2))
    batch_pred_simcc = batch_pred_simcc.clamp(min=0)

    # Compute the binary mask
    mask = (batch_pred_simcc.amax(dim=-1) > 1).reshape(B, K, 1)

    # Normalize the tensor using the maximum value
    norm = (batch_pred_simcc / batch_pred_simcc.amax(dim=-1).reshape(B, K, 1))

    # Apply normalization
    batch_pred_simcc = torch.where(mask, norm, batch_pred_simcc)

    return batch_pred_simcc


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_x.ndim == simcc_y.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

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


def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc


def batch_heatmap_nms(batch_heatmaps: Tensor, kernel_size: int = 5):
    """Apply NMS on a batch of heatmaps.

    Args:
        batch_heatmaps (Tensor): batch heatmaps in shape (B, K, H, W)
        kernel_size (int): The kernel size of the NMS which should be
            a odd integer. Defaults to 5

    Returns:
        Tensor: The batch heatmaps after NMS.
    """

    assert isinstance(kernel_size, int) and kernel_size % 2 == 1, \
        f'The kernel_size should be an odd integer, got {kernel_size}'

    padding = (kernel_size - 1) // 2

    maximum = F.max_pool2d(
        batch_heatmaps, kernel_size, stride=1, padding=padding)
    maximum_indicator = torch.eq(batch_heatmaps, maximum)
    batch_heatmaps = batch_heatmaps * maximum_indicator.float()

    return batch_heatmaps
