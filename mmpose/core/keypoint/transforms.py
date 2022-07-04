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

    assert keypoints.shape[:-1] == keypoints_visible.shape, (
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
                keypoints_visible_flipped[..., left] = keypoints_visible[...,
                                                                         right]

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


def keypoints_bbox2img(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords
