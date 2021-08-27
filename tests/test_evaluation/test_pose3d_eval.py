# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmpose.core import keypoint_3d_auc, keypoint_3d_pck


def test_keypoint_3d_pck():
    target = np.random.rand(2, 5, 3)
    output = np.copy(target)
    mask = np.ones((output.shape[0], output.shape[1]), dtype=bool)

    with pytest.raises(ValueError):
        _ = keypoint_3d_pck(output, target, mask, alignment='norm')

    pck = keypoint_3d_pck(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(pck, 100)

    output[0, 0, :] = target[0, 0, :] + 1
    pck = keypoint_3d_pck(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(pck, 90, 5)

    output = target * 2
    pck = keypoint_3d_pck(output, target, mask, alignment='scale')
    np.testing.assert_almost_equal(pck, 100)

    output = target + 2
    pck = keypoint_3d_pck(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(pck, 100)


def test_keypoint_3d_auc():
    target = np.random.rand(2, 5, 3)
    output = np.copy(target)
    mask = np.ones((output.shape[0], output.shape[1]), dtype=bool)

    with pytest.raises(ValueError):
        _ = keypoint_3d_auc(output, target, mask, alignment='norm')

    auc = keypoint_3d_auc(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)

    output = target * 2
    auc = keypoint_3d_auc(output, target, mask, alignment='scale')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)

    output = target + 2
    auc = keypoint_3d_auc(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)
