# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core import (affine_transform, flip_back, fliplr_joints,
                         fliplr_regression, get_affine_transform, rotate_point,
                         transform_preds)


def test_affine_transform():
    pt = np.array([0, 1])
    trans = np.array([[1, 0, 1], [0, 1, 0]])
    result = affine_transform(pt, trans)
    assert_array_almost_equal(result, np.array([1, 1]), decimal=4)
    assert isinstance(result, np.ndarray)


def test_rotate_point():
    src_point = np.array([0, 1])
    rot_rad = np.pi / 2.
    result = rotate_point(src_point, rot_rad)
    assert_array_almost_equal(result, np.array([-1, 0]), decimal=4)
    assert isinstance(result, list)


def test_fliplr_joints():
    joints = np.array([[0, 0, 0], [1, 1, 0]])
    joints_vis = np.array([[1], [1]])
    joints_flip, _ = fliplr_joints(joints, joints_vis, 5, [[0, 1]])
    res = np.array([[3, 1, 0], [4, 0, 0]])
    assert_array_almost_equal(joints_flip, res)


def test_flip_back():
    heatmaps = np.random.random([1, 2, 32, 32])
    flipped_heatmaps = flip_back(heatmaps, [[0, 1]])
    heatmaps_new = flip_back(flipped_heatmaps, [[0, 1]])
    assert_array_almost_equal(heatmaps, heatmaps_new)

    heatmaps = np.random.random([1, 2, 32, 32])
    flipped_heatmaps = flip_back(heatmaps, [[0, 1]])
    heatmaps_new = flipped_heatmaps[..., ::-1]
    assert_array_almost_equal(heatmaps[:, 0], heatmaps_new[:, 1])
    assert_array_almost_equal(heatmaps[:, 1], heatmaps_new[:, 0])

    ori_heatmaps = heatmaps.copy()
    # test in-place flip
    heatmaps = heatmaps[:, :, :, ::-1]
    assert_array_almost_equal(ori_heatmaps[:, :, :, ::-1], heatmaps)


def test_transform_preds():
    coords = np.random.random([2, 2])
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([100, 100])
    result = transform_preds(coords, center, scale, size)
    assert_array_almost_equal(coords, result)

    coords = np.random.random([2, 2])
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([101, 101])
    result = transform_preds(coords, center, scale, size, use_udp=True)
    assert_array_almost_equal(coords, result)


def test_get_affine_transform():
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([100, 100])
    result = get_affine_transform(center, scale, 0, size)
    trans = np.array([[1, 0, 0], [0, 1, 0]])
    assert_array_almost_equal(trans, result)


def test_flip_regression():
    coords = np.random.rand(3, 3)
    flip_pairs = [[1, 2]]
    root = coords[:1]
    coords_flipped = coords.copy()
    coords_flipped[1] = coords[2]
    coords_flipped[2] = coords[1]
    coords_flipped[..., 0] = 2 * root[..., 0] - coords_flipped[..., 0]

    # static mode
    res_static = fliplr_regression(
        coords, flip_pairs, center_mode='static', center_x=root[0, 0])
    assert_array_almost_equal(res_static, coords_flipped)

    # root mode
    res_root = fliplr_regression(
        coords, flip_pairs, center_mode='root', center_index=0)
    assert_array_almost_equal(res_root, coords_flipped)
