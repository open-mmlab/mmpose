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
            in shape (..., K, 1) or (..., K, 2). Set ``None`` if the keypoint
            visibility is unavailable
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
            visibility in shape (..., K, 1) or (..., K, 2). Return ``None`` if
            the input ``keypoints_visible`` is ``None``
    """

    ndim = keypoints.ndim
    assert keypoints.shape[:-1] == keypoints_visible.shape[:ndim - 1], (
        f'Mismatched shapes of keypoints {keypoints.shape} and '
        f'keypoints_visible {keypoints_visible.shape}')

    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    # swap the symmetric keypoint pairs
    if direction == 'horizontal' or direction == 'vertical':
        keypoints = keypoints.take(flip_indices, axis=ndim - 2)
        if keypoints_visible is not None:
            keypoints_visible = keypoints_visible.take(
                flip_indices, axis=ndim - 2)

    # flip the keypoints
    w, h = image_size
    if direction == 'horizontal':
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
    elif direction == 'vertical':
        keypoints[..., 1] = h - 1 - keypoints[..., 1]
    else:
        keypoints = [w, h] - keypoints - 1

    return keypoints, keypoints_visible


def flip_keypoints_custom_center(keypoints: np.ndarray,
                                 keypoints_visible: np.ndarray,
                                 flip_indices: List[int],
                                 center_mode: str = 'static',
                                 center_x: float = 0.5,
                                 center_index: int = 0):
    """Flip human joints horizontally.

    Note:
        - num_keypoint: K
        - dimension: D

    Args:
        keypoints (np.ndarray([..., K, D])): Coordinates of keypoints.
        keypoints_visible (np.ndarray([..., K])): Visibility item of keypoints.
        flip_indices (list[int]): The indices to flip the keypoints.
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:

            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)

            Defaults: ``'static'``.
        center_x (float): Set the x-axis location of the flip center. Only used
            when ``center_mode`` is ``'static'``. Defaults: 0.5.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when ``center_mode`` is
            ``'root'``. Defaults: 0.

    Returns:
        np.ndarray([..., K, C]): Flipped joints.
    """

    assert keypoints.ndim >= 2, f'Invalid pose shape {keypoints.shape}'

    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, 'Get invalid center_mode ' \
        f'{center_mode}, allowed choices are {allowed_center_mode}'

    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert keypoints.shape[-2] > center_index
        x_c = keypoints[..., center_index, 0]

    keypoints_flipped = keypoints.copy()
    keypoints_visible_flipped = keypoints_visible.copy()
    # Swap left-right parts
    for left, right in enumerate(flip_indices):
        keypoints_flipped[..., left, :] = keypoints[..., right, :]
        keypoints_visible_flipped[..., left] = keypoints_visible[..., right]

    # Flip horizontally
    keypoints_flipped[..., 0] = x_c * 2 - keypoints_flipped[..., 0]
    return keypoints_flipped, keypoints_visible_flipped


def keypoint_clip_border(keypoints: np.ndarray, keypoints_visible: np.ndarray,
                         shape: Tuple[int,
                                      int]) -> Tuple[np.ndarray, np.ndarray]:
    """Set the visibility values for keypoints outside the image border.

    Args:
        keypoints (np.ndarray): Input keypoints coordinates.
        keypoints_visible (np.ndarray): Visibility values of keypoints.
        shape (Tuple[int, int]): Shape of the image to which keypoints are
            being clipped in the format of (w, h).

    Note:
        This function sets the visibility values of keypoints that fall outside
            the specified frame border to zero (0.0).
    """
    width, height = shape[:2]

    # Create a mask for keypoints outside the frame
    outside_mask = ((keypoints[..., 0] > width) | (keypoints[..., 0] < 0) |
                    (keypoints[..., 1] > height) | (keypoints[..., 1] < 0))

    # Update visibility values for keypoints outside the frame
    if keypoints_visible.ndim == 2:
        keypoints_visible[outside_mask] = 0.0
    elif keypoints_visible.ndim == 3:
        keypoints_visible[outside_mask, 0] = 0.0

    return keypoints, keypoints_visible
