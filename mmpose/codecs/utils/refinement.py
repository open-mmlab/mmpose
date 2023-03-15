# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product

import numpy as np

from .post_processing import gaussian_blur, gaussian_blur1d


def refine_keypoints(keypoints: np.ndarray,
                     heatmaps: np.ndarray) -> np.ndarray:
    """Refine keypoint predictions by moving from the maximum towards the
    second maximum by 0.25 pixel. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)

        if 1 < x < W - 1 and 0 < y < H:
            dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1]
        else:
            dx = 0.

        if 1 < y < H - 1 and 0 < x < W:
            dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x]
        else:
            dy = 0.

        keypoints[n, k] += np.sign([dx, dy], dtype=np.float32) * 0.25

    return keypoints


def refine_keypoints_dark(keypoints: np.ndarray, heatmaps: np.ndarray,
                          blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate
    decoding. See `Dark Pose`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
            dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])

            dxx = 0.25 * (
                heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] +
                heatmaps[k, y, x - 2])
            dxy = 0.25 * (
                heatmaps[k, y + 1, x + 1] - heatmaps[k, y - 1, x + 1] -
                heatmaps[k, y + 1, x - 1] + heatmaps[k, y - 1, x - 1])
            dyy = 0.25 * (
                heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] +
                heatmaps[k, y - 2, x])
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                keypoints[n, k, :2] += offset
    return keypoints


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints


def refine_simcc_dark(keypoints: np.ndarray, simcc: np.ndarray,
                      blur_kernel_size: int) -> np.ndarray:
    """SimCC version. Refine keypoint predictions using distribution aware
    coordinate decoding for UDP. See `UDP`_ for details. The operation is in-
    place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        simcc (np.ndarray): The heatmaps in shape (N, K, Wx)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N = simcc.shape[0]

    # modulate simcc
    simcc = gaussian_blur1d(simcc, blur_kernel_size)
    np.clip(simcc, 1e-3, 50., simcc)
    np.log(simcc, simcc)

    simcc = np.pad(simcc, ((0, 0), (0, 0), (2, 2)), 'edge')

    for n in range(N):
        px = (keypoints[n] + 2.5).astype(np.int64).reshape(-1, 1)  # K, 1

        dx0 = np.take_along_axis(simcc[n], px, axis=1)  # K, 1
        dx1 = np.take_along_axis(simcc[n], px + 1, axis=1)
        dx_1 = np.take_along_axis(simcc[n], px - 1, axis=1)
        dx2 = np.take_along_axis(simcc[n], px + 2, axis=1)
        dx_2 = np.take_along_axis(simcc[n], px - 2, axis=1)

        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

        offset = dx / dxx
        keypoints[n] -= offset.reshape(-1)

    return keypoints
