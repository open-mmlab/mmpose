# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import numpy as np


def camera_to_image_coord(root_index: int, kpts_3d_cam: np.ndarray,
                          camera_param: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Project keypoints from camera space to image space and calculate factor.

    Args:
        root_index (int): Index for root keypoint.
        kpts_3d_cam (np.ndarray): Keypoint coordinates in camera space in
            shape (N, K, D).
        camera_param (dict): Parameters for the camera.

    Returns:
        tuple:
        - kpts_3d_image (np.ndarray): Keypoint coordinates in image space in
            shape (N, K, D).
        - factor (np.ndarray): The scaling factor that maps keypoints from
            image space to camera space in shape (N, ).
    """

    root = kpts_3d_cam[..., root_index, :]
    tl_kpt = root.copy()
    tl_kpt[..., :2] -= 1.0
    br_kpt = root.copy()
    br_kpt[..., :2] += 1.0
    tl_kpt = np.reshape(tl_kpt, (-1, 3))
    br_kpt = np.reshape(br_kpt, (-1, 3))
    fx, fy = camera_param['f'] / 1000.
    cx, cy = camera_param['c'] / 1000.

    tl2d = camera_to_pixel(tl_kpt, fx, fy, cx, cy)
    br2d = camera_to_pixel(br_kpt, fx, fy, cx, cy)

    rectangle_3d_size = 2.0
    kpts_3d_image = np.zeros_like(kpts_3d_cam)
    kpts_3d_image[..., :2] = camera_to_pixel(kpts_3d_cam.copy(), fx, fy, cx,
                                             cy)
    ratio = (br2d[..., 0] - tl2d[..., 0] + 0.001) / rectangle_3d_size
    factor = rectangle_3d_size / (br2d[..., 0] - tl2d[..., 0] + 0.001)
    kpts_3d_depth = ratio[:, None] * (
        kpts_3d_cam[..., 2] - kpts_3d_cam[..., root_index:root_index + 1, 2])
    kpts_3d_image[..., 2] = kpts_3d_depth
    return kpts_3d_image, factor


def camera_to_pixel(kpts_3d: np.ndarray,
                    fx: float,
                    fy: float,
                    cx: float,
                    cy: float,
                    shift: bool = False) -> np.ndarray:
    """Project keypoints from camera space to image space.

    Args:
        kpts_3d (np.ndarray): Keypoint coordinates in camera space.
        fx (float): x-coordinate of camera's focal length.
        fy (float): y-coordinate of camera's focal length.
        cx (float): x-coordinate of image center.
        cy (float): y-coordinate of image center.
        shift (bool): Whether to shift the coordinates by 1e-8.

    Returns:
        pose_2d (np.ndarray): Projected keypoint coordinates in image space.
    """
    if not shift:
        pose_2d = kpts_3d[..., :2] / kpts_3d[..., 2:3]
    else:
        pose_2d = kpts_3d[..., :2] / (kpts_3d[..., 2:3] + 1e-8)
    pose_2d[..., 0] *= fx
    pose_2d[..., 1] *= fy
    pose_2d[..., 0] += cx
    pose_2d[..., 1] += cy
    return pose_2d


def pixel_to_camera(kpts_3d: np.ndarray, fx: float, fy: float, cx: float,
                    cy: float) -> np.ndarray:
    """Project keypoints from camera space to image space.

    Args:
        kpts_3d (np.ndarray): Keypoint coordinates in camera space.
        fx (float): x-coordinate of camera's focal length.
        fy (float): y-coordinate of camera's focal length.
        cx (float): x-coordinate of image center.
        cy (float): y-coordinate of image center.
        shift (bool): Whether to shift the coordinates by 1e-8.

    Returns:
        pose_2d (np.ndarray): Projected keypoint coordinates in image space.
    """
    pose_2d = kpts_3d.copy()
    pose_2d[..., 0] -= cx
    pose_2d[..., 1] -= cy
    pose_2d[..., 0] /= fx
    pose_2d[..., 1] /= fy
    pose_2d[..., 0] *= kpts_3d[..., 2]
    pose_2d[..., 1] *= kpts_3d[..., 2]
    return pose_2d
