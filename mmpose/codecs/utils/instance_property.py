# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np


def get_instance_root(keypoints: np.ndarray,
                      keypoints_visible: Optional[np.ndarray] = None,
                      root_type: str = 'kpt_center',
                      output_size: Optional[Tuple[int,
                                                  int]] = None) -> np.ndarray:
    """Calculate the coordinates and visibility of instance roots.

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        tuple
        - roots_coordinate(np.ndarray): Coordinates of instance roots in
            shape [N, D]
        - roots_visible(np.ndarray): Visibility of instance roots in
            shape [N]
    """

    roots_coordinate = np.zeros((keypoints.shape[0], 2), dtype=np.float32)
    roots_visible = np.ones((keypoints.shape[0]), dtype=np.float32) * 2

    for i in range(keypoints.shape[0]):

        # collect visible keypoints
        if keypoints_visible is not None:
            visible_keypoints = keypoints[i][keypoints_visible[i] > 0]
        else:
            visible_keypoints = keypoints[i]
        if visible_keypoints.size == 0:
            roots_visible[i] = 0
            continue

        # compute the instance root with visible keypoints
        if root_type == 'kpt_center':
            roots_coordinate[i] = visible_keypoints.mean(axis=0)
        elif root_type == 'bbox_center':
            roots_coordinate[i] = (visible_keypoints.max(axis=0) +
                                   visible_keypoints.min(axis=0)) / 2.0
        else:
            raise ValueError(
                f'the value of `root_type` must be \'kpt_center\' or '
                f'\'bbox_center\', but got \'{root_type}\'')

        if output_size is not None:
            W, H = output_size
            # compute the visibility of roots
            if roots_coordinate[i][0] >= W or roots_coordinate[i][
                    1] >= H or roots_coordinate[i].min() < 0:
                roots_visible[i] = 0
    return roots_coordinate, roots_visible


def get_diagonal_lengths(keypoints: np.ndarray,
                         keypoints_visible: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """Calculate the diagonal length of instance bounding box from visible
    keypoints.

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        np.ndarray: bounding box diagonal length in [N]
    """
    diagonal_length = np.zeros((keypoints.shape[0]), dtype=np.float32)
    for i in range(keypoints.shape[0]):
        if keypoints_visible is not None:
            visible_keypoints = keypoints[i][keypoints_visible[i] > 0]
        else:
            visible_keypoints = keypoints[i]
        if visible_keypoints.size == 0:
            continue

        h_w_diff = visible_keypoints.max(axis=0) - visible_keypoints.min(
            axis=0)
        diagonal_length[i] = np.sqrt(np.power(h_w_diff, 2).sum())
    return diagonal_length
