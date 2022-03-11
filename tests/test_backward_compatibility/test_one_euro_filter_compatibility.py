# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmpose.core.post_processing.one_euro_filter import OneEuroFilter


def test_one_euro_filter():
    np.random.seed(1)

    kpts = []
    frames = 100
    for i in range(frames):
        kpts.append({
            'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
            'area': 100,
            'score': 0.9
        })
        kpts.append({
            'keypoints': np.tile(np.array([11, 11, 0.9]), [17, 1]),
            'area': 100,
            'score': 0.8
        })

    one_euro_filter = OneEuroFilter(
        kpts[0]['keypoints'][:, :2], min_cutoff=1.7, beta=0.3, fps=30)

    for i in range(1, len(kpts)):
        kpts[i]['keypoints'][:, :2] = one_euro_filter(
            kpts[i]['keypoints'][:, :2])

    one_euro_filter = OneEuroFilter(
        kpts[0]['keypoints'][:, :2], min_cutoff=1.7, beta=0.3)

    for i in range(1, len(kpts)):
        kpts[i]['keypoints'][:, :2] = one_euro_filter(
            kpts[i]['keypoints'][:, :2])
