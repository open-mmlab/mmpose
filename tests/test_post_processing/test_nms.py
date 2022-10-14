# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmpose.core.post_processing.nms import (nearby_joints_nms, nms, oks_iou,
                                             oks_nms, soft_oks_nms)


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


def test_nearby_joints_nms():

    kpts_db = []
    keep_pose_inds = nearby_joints_nms(
        kpts_db, 0.05, score_per_joint=True, max_dets=1)
    assert len(keep_pose_inds) == 0

    kpts_db = []
    for _ in range(5):
        kpts_db.append(
            dict(keypoints=np.random.rand(3, 2), score=np.random.rand(3)))
    keep_pose_inds = nearby_joints_nms(
        kpts_db, 0.05, score_per_joint=True, max_dets=1)
    assert len(keep_pose_inds) == 1
    assert keep_pose_inds[0] < 5

    kpts_db = []
    for _ in range(5):
        kpts_db.append(
            dict(keypoints=np.random.rand(3, 2), score=np.random.rand()))
    keep_pose_inds = nearby_joints_nms(kpts_db, 0.05, num_nearby_joints_thr=2)
    assert len(keep_pose_inds) <= 5 and len(keep_pose_inds) > 0

    with pytest.raises(AssertionError):
        _ = nearby_joints_nms(kpts_db, 0, num_nearby_joints_thr=2)

    with pytest.raises(AssertionError):
        _ = nearby_joints_nms(kpts_db, 0.05, num_nearby_joints_thr=3)
