# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def flip_heatmaps(heatmaps: Tensor,
                  flip_indices: Optional[List[int]] = None,
                  flip_mode: str = 'heatmap',
                  shift_heatmap: bool = True):
    """Flip heatmaps for test-time augmentation.

    Args:
        heatmaps (Tensor): The heatmaps to flip. Should be a tensor in shape
            [B, C, H, W]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint. Defaults to ``None``
        flip_mode (str): Specify the flipping mode. Options are:

            - ``'heatmap'``: horizontally flip the heatmaps and swap heatmaps
                of symmetric keypoints according to ``flip_indices``
            - ``'udp_combined'``: similar to ``'heatmap'`` mode but further
                flip the x_offset values
            - ``'offset'``: horizontally flip the offset fields and swap
                heatmaps of symmetric keypoints according to
                ``flip_indices``. x_offset values are also reversed
        shift_heatmap (bool): Shift the flipped heatmaps to align with the
            original heatmaps and improve accuracy. Defaults to ``True``

    Returns:
        Tensor: flipped heatmaps in shape [B, C, H, W]
    """

    if flip_mode == 'heatmap':
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == heatmaps.shape[1]
            heatmaps = heatmaps[:, flip_indices]
    elif flip_mode == 'udp_combined':
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C // 3, 3, H, W)
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == C // 3
            heatmaps = heatmaps[:, flip_indices]
        heatmaps[:, :, 1] = -heatmaps[:, :, 1]
        heatmaps = heatmaps.view(B, C, H, W)

    elif flip_mode == 'offset':
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C // 2, -1, H, W)
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == C // 2
            heatmaps = heatmaps[:, flip_indices]
        heatmaps[:, :, 0] = -heatmaps[:, :, 0]
        heatmaps = heatmaps.view(B, C, H, W)

    else:
        raise ValueError(f'Invalid flip_mode value "{flip_mode}"')

    if shift_heatmap:
        # clone data to avoid unexpected in-place operation when using CPU
        heatmaps[..., 1:] = heatmaps[..., :-1].clone()

    return heatmaps


def flip_vectors(x_labels: Tensor, y_labels: Tensor, flip_indices: List[int]):
    """Flip instance-level labels in specific axis for test-time augmentation.

    Args:
        x_labels (Tensor): The vector labels in x-axis to flip. Should be
            a tensor in shape [B, C, Wx]
        y_labels (Tensor): The vector labels in y-axis to flip. Should be
            a tensor in shape [B, C, Wy]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
    """
    assert x_labels.ndim == 3 and y_labels.ndim == 3
    assert len(flip_indices) == x_labels.shape[1] and len(
        flip_indices) == y_labels.shape[1]
    x_labels = x_labels[:, flip_indices].flip(-1)
    y_labels = y_labels[:, flip_indices]

    return x_labels, y_labels


def flip_coordinates(coords: Tensor, flip_indices: List[int],
                     shift_coords: bool, input_size: Tuple[int, int]):
    """Flip normalized coordinates for test-time augmentation.

    Args:
        coords (Tensor): The coordinates to flip. Should be a tensor in shape
            [B, K, D]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        shift_coords (bool): Shift the flipped coordinates to align with the
            original coordinates and improve accuracy. Defaults to ``True``
        input_size (Tuple[int, int]): The size of input image in [w, h]
    """
    assert coords.ndim == 3
    assert len(flip_indices) == coords.shape[1]

    coords[:, :, 0] = 1.0 - coords[:, :, 0]

    if shift_coords:
        img_width = input_size[0]
        coords[:, :, 0] -= 1.0 / img_width

    coords = coords[:, flip_indices]
    return coords


def flip_visibility(vis: Tensor, flip_indices: List[int]):
    """Flip keypoints visibility for test-time augmentation.

    Args:
        vis (Tensor): The keypoints visibility to flip. Should be a tensor
            in shape [B, K]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
    """
    assert vis.ndim == 2

    vis = vis[:, flip_indices]
    return vis


def aggregate_heatmaps(heatmaps: List[Tensor],
                       size: Optional[Tuple[int, int]],
                       align_corners: bool = False,
                       mode: str = 'average'):
    """Aggregate multiple heatmaps.

    Args:
        heatmaps (List[Tensor]): Multiple heatmaps to aggregate. Each should
            be in shape (B, C, H, W)
        size (Tuple[int, int], optional): The target size in (w, h). All
            heatmaps will be resized to the target size. If not given, the
            first heatmap tensor's width and height will be used as the target
            size. Defaults to ``None``
        align_corners (bool): Whether align corners when resizing heatmaps.
            Defaults to ``False``
        mode (str): Aggregation mode in one of the following:

            - ``'average'``: Get average of heatmaps. All heatmaps mush have
                the same channel number
            - ``'concat'``: Concate the heatmaps at the channel dim
    """

    if mode not in {'average', 'concat'}:
        raise ValueError(f'Invalid aggregation mode `{mode}`')

    if size is None:
        h, w = heatmaps[0].shape[2:4]
    else:
        w, h = size

    for i, _heatmaps in enumerate(heatmaps):
        assert _heatmaps.ndim == 4
        if mode == 'average':
            assert _heatmaps.shape[:2] == heatmaps[0].shape[:2]
        else:
            assert _heatmaps.shape[0] == heatmaps[0].shape[0]

        if _heatmaps.shape[2:4] != (h, w):
            heatmaps[i] = F.interpolate(
                _heatmaps,
                size=(h, w),
                mode='bilinear',
                align_corners=align_corners)

    if mode == 'average':
        output = sum(heatmaps).div(len(heatmaps))
    elif mode == 'concat':
        output = torch.cat(heatmaps, dim=1)
    else:
        raise ValueError()

    return output
