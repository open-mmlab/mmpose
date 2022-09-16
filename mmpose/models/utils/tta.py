# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from torch import Tensor


def flip_heatmaps(heatmaps: Tensor,
                  flip_indices: List[int],
                  flip_mode: str = 'heatmap',
                  shift_heatmap: bool = True):
    """Flip heatmaps for test-time augmentation.

    Args:
        heatmaps (Tensor): The heatmaps to flip. Should be a tensor in shape
            [B, C, H, W]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        flip_mode (str): Specify the flipping mode. Options are:

            - ``'heatmap'``: horizontally flip the heatmaps and swap heatmaps
                of symmetric keypoints according to ``flip_indices``
            - ``'udp_combined'``: similar to ``'heatmap'`` mode but further
                flip the x_offset values
        shift_heatmap (bool): Shift the flipped heatmaps to align with the
            original heatmaps and improve accuracy. Defaults to ``True``

    Returns:
        Tensor: flipped heatmaps in shape [B, C, H, W]
    """

    if flip_mode == 'heatmap':
        assert len(flip_indices) == heatmaps.shape[1]
        heatmaps = heatmaps[:, flip_indices].flip(-1)

    elif flip_mode == 'udp_combined':
        B, C, H, W = heatmaps.shape
        assert len(flip_indices) * 3 == C
        heatmaps = heatmaps.view(B, C // 3, 3, H, W)
        heatmaps = heatmaps[:, flip_indices].flip(-1)
        heatmaps[:, :, 1] = -heatmaps[:, :, 1]
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
