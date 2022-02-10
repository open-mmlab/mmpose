# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmpose.core.post_processing.nms import nms, oks_iou, oks_nms, soft_oks_nms


def test_soft_oks_nms():
    oks_thr = 0.9
    kpts = []
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.9
    })
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.4
    })
    kpts.append({
        'keypoints': np.tile(np.array([100, 100, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.7
    })

    keep = soft_oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2, 1])).all()

    keep = oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2])).all()

    kpts_with_score_joints = []
    kpts_with_score_joints.append({
        'keypoints':
        np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area':
        100,
        'score':
        np.tile(np.array([0.9]), 17)
    })
    kpts_with_score_joints.append({
        'keypoints':
        np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area':
        100,
        'score':
        np.tile(np.array([0.4]), 17)
    })
    kpts_with_score_joints.append({
        'keypoints':
        np.tile(np.array([100, 100, 0.9]), [17, 1]),
        'area':
        100,
        'score':
        np.tile(np.array([0.7]), 17)
    })
    keep = soft_oks_nms([
        kpts_with_score_joints[i] for i in range(len(kpts_with_score_joints))
    ],
                        oks_thr,
                        score_per_joint=True)
    assert (keep == np.array([0, 2, 1])).all()

    keep = oks_nms([
        kpts_with_score_joints[i] for i in range(len(kpts_with_score_joints))
    ],
                   oks_thr,
                   score_per_joint=True)
    assert (keep == np.array([0, 2])).all()


def test_func_nms():
    result = nms(np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 8, 0.8]]), 0.5)
    assert result == [0]


def test_oks_iou():
    result = oks_iou(np.ones([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
    assert result[0] == 1.
    result = oks_iou(np.zeros([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
    assert result[0] < 0.01
