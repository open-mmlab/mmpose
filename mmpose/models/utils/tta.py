# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

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
                of symmetric keypoints according to ``flip_pairs``
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
        heatmaps = heatmaps.view(B, 3, C // 3, H, W)
        heatmaps = heatmaps[:, flip_indices].flip(-1)
        heatmaps[:, 1] = -heatmaps[:, 1]
        heatmaps = heatmaps.view(B, C, H, W)
    else:
        raise ValueError(f'Invalid flip_mode value "{flip_mode}"')

    if shift_heatmap:
        heatmaps[..., 1:] = heatmaps[..., :-1]

    return heatmaps
