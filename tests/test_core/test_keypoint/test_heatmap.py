# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmpose.core.keypoint.heatmap import (keypoints_from_heatmaps,
                                          keypoints_from_heatmaps3d)


def test_keypoints_from_heatmaps():
    heatmaps = np.ones((1, 1, 64, 64), dtype=np.float32)
    heatmaps[0, 0, 31, 31] = 2
    center = np.array([[127, 127]])
    scale = np.array([[64 / 200.0, 64 / 200.0]])

    udp_heatmaps = np.ones((32, 17, 64, 64), dtype=np.float32)
    udp_heatmaps[:, :, 31, 31] = 2
    udp_center = np.tile([127, 127], (32, 1))
    udp_scale = np.tile([32, 32], (32, 1))

    preds, maxvals = keypoints_from_heatmaps(heatmaps, center, scale)

    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    with pytest.raises(AssertionError):
        # kernel should > 0
        _ = keypoints_from_heatmaps(
            heatmaps, center, scale, post_process='unbiased', kernel=0)

    preds, maxvals = keypoints_from_heatmaps(
        heatmaps, center, scale, post_process='unbiased')
    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    # test for udp dimension problem
    preds, maxvals = keypoints_from_heatmaps(
        udp_heatmaps,
        udp_center,
        udp_scale,
        post_process='default',
        target_type='GaussianHeatMap',
        use_udp=True)
    assert_array_almost_equal(preds, np.tile([76, 76], (32, 17, 1)), decimal=0)
    assert_array_almost_equal(maxvals, np.tile([2], (32, 17, 1)), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    preds1, maxvals1 = keypoints_from_heatmaps(
        heatmaps,
        center,
        scale,
        post_process='default',
        target_type='GaussianHeatMap',
        use_udp=True)
    preds2, maxvals2 = keypoints_from_heatmaps(
        heatmaps,
        center,
        scale,
        post_process='default',
        target_type='GaussianHeatmap',
        use_udp=True)
    assert_array_almost_equal(preds1, preds2, decimal=4)
    assert_array_almost_equal(maxvals1, maxvals2, decimal=4)
    assert isinstance(preds2, np.ndarray)
    assert isinstance(maxvals2, np.ndarray)


def test_keypoints_from_heatmaps3d():
    heatmaps = np.ones((1, 1, 64, 64, 64), dtype=np.float32)
    heatmaps[0, 0, 10, 31, 40] = 2
    center = np.array([[127, 127]])
    scale = np.array([[64 / 200.0, 64 / 200.0]])
    preds, maxvals = keypoints_from_heatmaps3d(heatmaps, center, scale)

    assert_array_almost_equal(preds, np.array([[[135, 126, 10]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)
