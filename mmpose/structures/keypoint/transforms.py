# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np


def flip_keypoints(keypoints: np.ndarray,
                   keypoints_visible: Optional[np.ndarray],
                   image_size: Tuple[int, int],
                   flip_indices: List[int],
                   direction: str = 'horizontal'
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flip keypoints in the given direction.

    Note:

        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): Keypoints in shape (..., K, D)
        keypoints_visible (np.ndarray, optional): The visibility of keypoints
            in shape (..., K, 1). Set ``None`` if the keypoint visibility is
            unavailable
        image_size (tuple): The image shape in [w, h]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        tuple:
        - keypoints_flipped (np.ndarray): Flipped keypoints in shape
            (..., K, D)
        - keypoints_visible_flipped (np.ndarray, optional): Flipped keypoints'
            visibility in shape (..., K, 1). Return ``None`` if the input
            ``keypoints_visible`` is ``None``
    """

    assert keypoints.shape[:-1] == keypoints_visible.shape, (
        f'Unmatched shapes of keypoints {keypoints.shape} and '
        f'keypoints_visible {keypoints_visible.shape}')

    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    # swap the symmetric keypoint pairs
    if direction == 'horizontal' or direction == 'vertical':
        keypoints = keypoints[..., flip_indices, :]
        if keypoints_visible is not None:
            keypoints_visible = keypoints_visible[..., flip_indices]

    # flip the keypoints
    w, h = image_size
    if direction == 'horizontal':
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
    elif direction == 'vertical':
        keypoints[..., 1] = h - 1 - keypoints[..., 1]
    else:
        keypoints = [w, h] - keypoints - 1

    return keypoints, keypoints_visible
