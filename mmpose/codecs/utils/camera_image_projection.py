# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def camera_to_image_coord(root_index, kpts_3d_cam, camera_param):
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


def camera_to_pixel(kpts_3d, fx, fy, cx, cy):
    pose_2d = kpts_3d[..., :2] / kpts_3d[..., 2:3]
    pose_2d[..., 0] *= fx
    pose_2d[..., 1] *= fy
    pose_2d[..., 0] += cx
    pose_2d[..., 1] += cy
    return pose_2d
