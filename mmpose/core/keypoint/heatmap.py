# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.core.post_processing import transform_preds


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
        x0 = y0 = gaussian_size // 2

        for i in range(num_keypoints):
            # skip unlabled keypoints
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            # get gaussian center coordinates
            mu = (keypoints[i] / feat_stride + 0.5)

            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= w or top >= h or right < 0 or bottom < 0:
                keypoint_weight[i] = 0
                continue

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
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)[:, None]

        # positive area radius in the classification map
        radius = factor * max(w, h)

        for i in range(num_keypoints):
            if keypoints_visible[i] < 0.5:
                keypoint_weight[i] = 0
                continue

            mu = keypoints[i] / feat_stride

            x_offset = (mu[0] - x) / radius
            y_offset = (mu[1] - y) / radius

            heatmap[i, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1., 0.)
            heatmap[i, 1] = x_offset
            heatmap[i, 2] = y_offset

        # keep only valid region in offset maps
        heatmap[:, 1:] *= heatmap[:, :1]
        heatmap = heatmap.reshape(num_keypoints * 3, h, w)

    return heatmap, keypoint_weight


def get_max_preds(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def get_max_preds_3d(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get keypoint predictions from 3D score maps.

    Note:
        batch size: N
        num keypoints: K
        heatmap depth size: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 3]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), \
        ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 5, 'heatmaps should be 5-ndim'

    N, K, D, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.zeros((N, K, 3), dtype=np.float32)
    _idx = idx[..., 0]
    preds[..., 2] = _idx // (H * W)
    preds[..., 1] = (_idx // W) % H
    preds[..., 0] = _idx % W

    preds = np.where(maxvals > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
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
            coord += offset
    return coord


def _gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

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
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps: np.ndarray,
                            center: np.ndarray,
                            scale: np.ndarray,
                            unbiased: bool = False,
                            post_process: Optional[str] = 'default',
                            kernel: int = 11,
                            valid_radius_factor: float = 0.0546875,
                            use_udp: bool = False,
                            target_type: str = 'GaussianHeatmap'
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
            Default: ``False``.
        post_process (str, optional): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'. Default: ``'default'``.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2. Default: 11.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP. Default: 0.0546875.
        use_udp (bool): Use unbiased data processing. Default: ``False``.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
            Default: ``'GaussianHeatmap'``.

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals


def keypoints_from_heatmaps3d(heatmaps: np.ndarray, center: np.ndarray,
                              scale: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Get final keypoint predictions from 3d heatmaps and transform them back
    to the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap depth size: D
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 3]): Predicted 3d keypoint location \
            in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    N, K, D, H, W = heatmaps.shape
    preds, maxvals = get_max_preds_3d(heatmaps)
    # Transform back to the image
    for i in range(N):
        preds[i, :, :2] = transform_preds(preds[i, :, :2], center[i], scale[i],
                                          [W, H])
    return preds, maxvals


def post_dark_udp(coords: np.ndarray,
                  batch_heatmaps: np.ndarray,
                  kernel: int = 3) -> np.ndarray:
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords
