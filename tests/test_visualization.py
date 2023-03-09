# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
import pytest

from mmpose.core import (imshow_bboxes, imshow_keypoints, imshow_keypoints_3d,
                         imshow_multiview_keypoints_3d)


def test_imshow_keypoints_2d():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts = np.array([[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1]],
                    dtype=np.float32)
    pose_result = [kpts]
    skeleton = [[0, 1], [1, 2], [2, 3]]
    # None: kpt or link is hidden
    pose_kpt_color = [None] + [(127, 127, 127)] * (len(kpts) - 1)
    pose_link_color = [(127, 127, 127)] * (len(skeleton) - 1) + [None]
    _ = imshow_keypoints(
        img,
        pose_result,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        show_keypoint_weight=True)


def test_imshow_keypoints_3d():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts_3d = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    pose_result_3d = [{'keypoints_3d': kpts_3d, 'title': 'test'}]
    skeleton = [[0, 1]]
    pose_kpt_color = [(127, 127, 127)] * len(kpts_3d)
    pose_link_color = [(127, 127, 127)] * len(skeleton)
    _ = imshow_keypoints_3d(
        pose_result_3d,
        img=img,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        vis_height=400)

    # multiview 3D keypoint
    pose_result_3d = [kpts_3d]
    _ = imshow_multiview_keypoints_3d(
        pose_result_3d,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        space_size=[8, 8, 8],
        space_center=[0, 0, 0],
        kpt_score_thr=0.0)


def test_imshow_bbox():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = np.array([[10, 10, 30, 30], [10, 50, 30, 80]], dtype=np.float32)
    labels = ['label 1', 'label 2']
    colors = ['red', 'green']

    with tempfile.TemporaryDirectory() as tmpdir:
        _ = imshow_bboxes(
            img,
            bboxes,
            labels=labels,
            colors=colors,
            show=False,
            out_file=f'{tmpdir}/out.png')

        # test case of empty bboxes
        _ = imshow_bboxes(
            img,
            np.zeros((0, 4), dtype=np.float32),
            labels=None,
            colors='red',
            show=False)

        # test unmatched bboxes and labels
        with pytest.raises(AssertionError):
            _ = imshow_bboxes(
                img,
                np.zeros((0, 4), dtype=np.float32),
                labels=labels[:1],
                colors='red',
                show=False)
