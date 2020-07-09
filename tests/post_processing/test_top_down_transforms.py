import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core import (flip_back, fliplr_joints, get_affine_transform,
                         transform_preds)


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
    ans = transform_preds(coords, center, scale, size)
    assert_array_almost_equal(coords, ans)


def test_get_affine_transform():
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([100, 100])
    ans = get_affine_transform(center, scale, 0, size)
    trans = np.array([[1, 0, 0], [0, 1, 0]])
    assert_array_almost_equal(trans, ans)
