# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import numpy as np
import pytest

from mmpose.core import (apply_bugeye_effect, apply_sunglasses_effect,
                         imshow_bboxes, imshow_keypoints, imshow_keypoints_3d)


def test_imshow_keypoints():
    # 2D keypoint
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts = np.array([[1, 1, 1], [10, 10, 1]], dtype=np.float32)
    pose_result = [kpts]
    skeleton = [[0, 1]]
    pose_kpt_color = [(127, 127, 127)] * len(kpts)
    pose_link_color = [(127, 127, 127)] * len(skeleton)
    img_vis_2d = imshow_keypoints(
        img,
        pose_result,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        show_keypoint_weight=True)

    # 3D keypoint
    kpts_3d = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    pose_result_3d = [{'keypoints_3d': kpts_3d, 'title': 'test'}]
    _ = imshow_keypoints_3d(
        pose_result_3d,
        img=img_vis_2d,
        skeleton=skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        vis_height=400)


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


def test_effects():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts = np.array([[10., 10., 0.8], [20., 10., 0.8]], dtype=np.float32)
    bbox = np.array([0, 0, 50, 50], dtype=np.float32)
    pose_results = [dict(bbox=bbox, keypoints=kpts)]
    # sunglasses
    sunglasses_img = mmcv.imread('demo/resources/sunglasses.jpg')
    _ = apply_sunglasses_effect(
        img,
        pose_results,
        sunglasses_img,
        left_eye_index=1,
        right_eye_index=0,
        kpt_thr=0.5)
    _ = apply_sunglasses_effect(
        img,
        pose_results,
        sunglasses_img,
        left_eye_index=1,
        right_eye_index=0,
        kpt_thr=0.9)

    # bug-eye
    _ = apply_bugeye_effect(
        img, pose_results, left_eye_index=1, right_eye_index=0, kpt_thr=0.5)
    _ = apply_bugeye_effect(
        img, pose_results, left_eye_index=1, right_eye_index=0, kpt_thr=0.9)
