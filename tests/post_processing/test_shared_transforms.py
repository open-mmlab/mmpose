import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core import affine_transform, get_3rd_point, rotate_point


def test_get_3rd_point():
    a = np.array([0, 1])
    b = np.array([0, 0])
    ans = get_3rd_point(a, b)
    assert_array_almost_equal(ans, np.array([-1, 0]), decimal=4)
    assert isinstance(ans, np.ndarray)


def test_affine_transform():
    pt = np.array([0, 1])
    trans = np.array([[1, 0, 1], [0, 1, 0]])
    ans = affine_transform(pt, trans)
    assert_array_almost_equal(ans, np.array([1, 1]), decimal=4)
    assert isinstance(ans, np.ndarray)


def test_rotate_point():
    src_point = np.array([0, 1])
    rot_rad = np.pi / 2.
    ans = rotate_point(src_point, rot_rad)
    assert_array_almost_equal(ans, np.array([-1, 0]), decimal=4)
    assert isinstance(ans, list)
