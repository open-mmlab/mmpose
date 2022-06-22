# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import cv2
import numpy as np


def generate_msra_heatmap(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    unbiased: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate keypoint heatmap via "MSRA" approach. See the paper: `Simple
    Baselines for Human Pose Estimation and Tracking`_ by Xiao et al (2018) for
    more details.

    Note:

        - keypoint number: K
        - keypoint dimension: C

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (K, C)
        keypoints_visible (np.ndarray): Keypoint visibility in shape (K, 1)
        sigma (float): The sigma value of the Gaussian heatmap
        image_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [w, h]
        unbiased (bool): Whether use unbiased method in ``'msra'`` encoding.
            See `Dark Pose`_ for details. Defaults to ``False``

    Returns:
        tuple:
        - heatmap (np.ndarray): The generated heatmap in shape (K, h, w) where
            [w, h] is the `heatmap_size`
        - keypoint_weight (np.ndarray): The target weights in shape (K, 1)

    .. _`Simple Baselines for Human Pose Estimation and Tracking`:
        https://arxiv.org/abs/1804.06208
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    num_keypoints = keypoints.shape[0]
    image_size = np.array(image_size)
    w, h = heatmap_size
    feat_stride = image_size / [w, h]

    heatmap = np.zeros((num_keypoints, h, w), dtype=np.float32)
    keypoint_weight = np.ones((num_keypoints, 1), dtype=np.float32)

    # 3-sigma rule
    radius = sigma * 3

    if unbiased:
        # xy grid
        x = np.arange(0, w, 1, dtype=np.float32)
        y = np.arange(0, h, 1, dtype=np.float32)[:, None]

        for i in range(num_keypoints):
            # skip unlabled keypoints
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            mu = keypoints[i] / feat_stride

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= w or top >= h or right < 0 or bottom < 0:
                keypoint_weight[i] = 0
                continue

            heatmap[i] = np.exp(-((x - mu[0])**2 + (y - mu[1])**2) /
                                (2 * sigma**2))

    else:
        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]

        for i in range(num_keypoints):
            # skip unlabled keypoints
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            # get integer center coordinates
            mu = (keypoints[i] / feat_stride + 0.5).astype(np.int64)

            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= w or top >= h or right < 0 or bottom < 0:
                keypoint_weight[i] = 0
                continue

            x0 = y0 = gaussian_size // 2
            # The gaussian is not normalized,
            # we want the center value to equal 1
            gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            # valid range in gaussian
            g_x1 = max(0, -left)
            g_x2 = min(w, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(h, bottom) - top

            # valid range in heatmap
            h_x1 = max(0, left)
            h_x2 = min(w, right)
            h_y1 = max(0, top)
            h_y2 = min(h, bottom)

            heatmap[i, h_y1:h_y2, h_x1:h_x2] = gaussian[g_y1:g_y2, g_x1:g_x2]

    return heatmap, keypoint_weight


def generate_megvii_heatmap(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    kernel_size: Tuple[int, int],
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate keypoint heatmap via "Megvii" approach. See `MSPN`_ (2019) and
    `CPN`_ (2018) for details.

    Note:

        - keypoint number: K
        - keypoint dimension: C

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (K, C)
        keypoints_visible (np.ndarray): Keypoint visibility in shape (K, 1)
        kernel_size (tuple): The kernel size of the heatmap gaussian in
            [ks_x, ks_y]
        image_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [w, h]

    Returns:
        tuple:
        - heatmap (np.ndarray): The generated heatmap in shape (K, h, w) where
            [w, h] is the `heatmap_size`
        - keypoint_weight (np.ndarray): The target weights in shape (K, 1)

    .. _`MSPN`: https://arxiv.org/abs/1901.00148
    .. _`CPN`: https://arxiv.org/abs/1711.07319
    """

    num_keypoints = keypoints.shape[0]
    image_size = np.array(image_size)
    w, h = heatmap_size
    feat_stride = image_size / [w, h]

    heatmap = np.zeros((num_keypoints, h, w), dtype=np.float32)
    keypoint_weight = np.ones((num_keypoints, 1), dtype=np.float32)

    for i in range(num_keypoints):
        # skip unlabled keypoints
        if keypoints_visible[i] < 0.5:
            keypoint_weight[i] = 0
            continue

        # get center coordinates
        kx, ky = (keypoints[i] / feat_stride).astype(np.int64)

        # if (mu[0] < 0 or mu[0]>=w or mu[1]<0 or mu[1]>=h):

        if kx < 0 or kx >= w or ky < 0 or ky >= h:
            keypoint_weight[i] = 0
            continue

        heatmap[i, ky, kx] = 1.
        heatmap[i] = cv2.GaussianBlur(heatmap[i], kernel_size, 0)

        # normalize the heatmap
        heatmap[i] = heatmap[i] / heatmap[i, ky, kx] * 255

    return heatmap, keypoint_weight


def generate_udp_heatmap(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    factor: float,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    combined_map: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate keypoint heatmap via "UDP" approach. See the paper: `The Devil
    is in the Details: Delving into Unbiased Data Processing for Human Pose
    Estimation`_ by Huang et al (2020) for more details.

    Note:

        - keypoint number: K
        - keypoint dimension: C

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (K, C)
        keypoints_visible (np.ndarray): Keypoint visibility in shape (K, 1)
        factor (float): The sigma value of the gaussian heatmap when
            ``combined_map==False``; or the valid radius factor of the binary
            heatmap map when ``combined_map==True``. See also ``combined_map``
        image_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [w, h]
        combined_map (bool): If ``True``, the generated map is a combination
            of a binary heatmap (for classification) and an offset map
            (for regression). Otherwise, the generated map is a gaussian
            heatmap. Defaults to ``False``

    Returns:
        tuple:
        - heatmap (np.ndarray): The generated heatmap in shape (K*3, h, w) if
            ``combined_map==True`` or (K, h, w) otherwise, where [w, h] is the
            `heatmap_size`
        - keypoint_weight (np.ndarray): The target weights in shape (K, 1)

    .. _`The Devil is in the Details: Delving into Unbiased Data Processing for
    Human Pose Estimation`: https://arxiv.org/abs/1911.07524
    """

    num_keypoints = keypoints.shape[0]
    image_size = np.array(image_size)
    w, h = heatmap_size
    feat_stride = (image_size - 1) / [w - 1, h - 1]
    keypoint_weight = np.ones((num_keypoints, 1), dtype=np.float32)

    if not combined_map:
        heatmap = np.zeros((num_keypoints, h, w), dtype=np.float32)

        # 3-sigma rule
        radius = factor * 3

        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]

        for i in range(num_keypoints):
            # skip unlabled keypoints
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            mu = (keypoints[i] / feat_stride + 0.5).astype(np.int64)
            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= w or top >= h or right < 0 or bottom < 0:
                keypoint_weight[i] = 0
                continue

            mu_ac = keypoints[i] / feat_stride
            x0 = y0 = gaussian_size // 2
            x0 += mu_ac[0] - mu[0]
            y0 += mu_ac[1] - mu[1]
            gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

            # valid range in gaussian
            g_x1 = max(0, -left)
            g_x2 = min(w, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(h, bottom) - top

            # valid range in heatmap
            h_x1 = max(0, left)
            h_x2 = min(w, right)
            h_y1 = max(0, top)
            h_y2 = min(h, bottom)

            heatmap[i, h_y1:h_y2, h_x1:h_x2] = gaussian[g_y1:g_y2, g_x1:g_x2]

    else:
        heatmap = np.zeros((num_keypoints, 3, h, w), dtype=np.float32)
        x = np.arange(0, w, 1, dtype=np.float32)
        y = np.arange(0, h, 1, dtype=np.float32)[:, None]

        # positive area radius in the classification map
        radius = factor * max(w, h)

        for i in range(num_keypoints):
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            mu = keypoints[i] / feat_stride

            x_offset = (x - mu[0]) / radius
            y_offset = (y - mu[1]) / radius

            heatmap[i, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1., 0.)
            heatmap[i, 1] = x_offset
            heatmap[i, 2] = y_offset

        # keep only valid region in offset maps
        heatmap[:, 1:] *= heatmap[:, :1]
        heatmap = heatmap.reshape(num_keypoints * 3, h, w)

    return heatmap, keypoint_weight
