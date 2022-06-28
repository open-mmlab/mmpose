# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np


def flip_keypoints(keypoints: np.ndarray,
                   keypoints_visible: Optional[np.ndarray],
                   image_size: Tuple[int, int],
                   flip_pairs: List,
                   direction: str = 'horizontal'
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flip keypoints in the given direction.

    Note:

        - keypoint number: K
        - keypoint dimension: C

    Args:
        keypoints (np.ndarray): Keypoints in shape (..., K, C)
        keypoints_visible (np.ndarray, optional): The visibility of keypoints
            in shape (..., K, 1). Set ``None`` if the keypoint visibility is
            unavailable
        image_size (tuple): The image shape in [w, h]
        flip_pairs (list[tuple]): The list of symmetric keypoint pairs'
            indices
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        tuple:
        - keypoints_flipped (np.ndarray): Flipped keypoints in shape
            (..., K, C)
        - keypoints_visible_flipped (np.ndarray, optional): Flipped keypoints'
            visibility in shape (..., K, 1). Return ``None`` if the input
            ``keypoints_visible`` is ``None``
    """

    assert keypoints.shape[:-1] == keypoints_visible.shape[:-1], (
        f'Unmatched shapes of keypoints {keypoints.shape} and '
        f'keypoints_visible {keypoints_visible.shape}')

    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    keypoints_flipped = keypoints.copy()
    if keypoints_visible is None:
        keypoints_visible_flipped = None
    else:
        keypoints_visible_flipped = keypoints_visible.copy()

    # swap the symmetric keypoint pairs
    if direction == 'horizontal' or direction == 'vertical':
        for left, right in flip_pairs:
            keypoints_flipped[..., left, :] = keypoints[..., right, :]

        if keypoints_visible is not None:
            for left, right in flip_pairs:
                keypoints_visible_flipped[..., left, :] = keypoints_visible[
                    ..., right, :]

    # flip the keypoints
    # TODO: consider using "integer corner" coordinate system
    w, h = image_size
    if direction == 'horizontal':
        keypoints_flipped[..., 0] = w - 1 - keypoints_flipped[..., 0]
    elif direction == 'vertical':
        keypoints_flipped[..., 1] = h - 1 - keypoints_flipped[..., 1]
    else:
        keypoints_flipped = [w, h] - keypoints_flipped - 1

    return keypoints_flipped, keypoints_visible_flipped
