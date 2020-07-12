import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmpose.core import keypoints_from_heatmaps, pose_pck_accuracy


def test_pose_pck_accuracy():
    output = np.zeros((1, 5, 64, 64))
    target = np.zeros((1, 5, 64, 64))
    # first channnel
    output[0, 0, 20, 20] = 1
    target[0, 0, 10, 10] = 1
    # second channel
    output[0, 1, 30, 30] = 1
    target[0, 1, 30, 30] = 1

    acc, avg_acc, cnt = pose_pck_accuracy(output, target)

    assert_array_almost_equal(acc, np.array([0, 1, -1, -1, -1]), decimal=4)
    assert abs(avg_acc - 0.5) < 1e-4
    assert abs(cnt - 2) < 1e-4


def test_keypoints_from_heatmaps():
    heatmaps = np.ones((1, 1, 64, 64))
    heatmaps[0, 0, 31, 31] = 2
    center = np.array([[127, 127]])
    scale = np.array([[64 / 200.0, 64 / 200.0]])

    preds, maxvals = keypoints_from_heatmaps(heatmaps, center, scale)

    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    with pytest.raises(AssertionError):
        # kernel should > 0
        _ = keypoints_from_heatmaps(
            heatmaps, center, scale, unbiased=True, kernel=0)

    preds, maxvals = keypoints_from_heatmaps(
        heatmaps, center, scale, unbiased=True)

    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)
