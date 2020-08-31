import numpy as np

from mmpose.core.post_processing.nms import nms, oks_iou


def test_nms():
    result = nms(np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 8, 0.8]]), 0.5)
    assert result == [0]


def test_oks_iou():
    result = oks_iou(np.ones([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
    assert result == [1.]
