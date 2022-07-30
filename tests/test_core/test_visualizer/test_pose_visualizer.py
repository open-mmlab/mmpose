# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.data import InstanceData, PixelData

from mmpose.core import PoseDataSample, PoseLocalVisualizer


class TestPoseLocalVisualizer(TestCase):

    def test_add_datasample(self):
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.uint8)
        out_file = 'out_file.jpg'

        # None: kpt or link is hidden
        pose_kpt_color = [None] + [(127, 127, 127)] * 3
        pose_link_color = [(127, 127, 127)] * 2 + [None]
        dataset_meta = {
            'keypoint_colors': pose_kpt_color,
            'skeleton_link_colors': pose_link_color,
            'skeleton_links': [[0, 1], [1, 2], [2, 3]]
        }

        pose_local_visualizer = PoseLocalVisualizer(
            kpt_color=pose_kpt_color,
            link_color=pose_link_color,
            show_keypoint_weight=True)
        pose_local_visualizer.set_dataset_meta(dataset_meta)

        # setting keypoints
        gt_instances = InstanceData()
        gt_instances.keypoints = np.array([[[1, 1], [20, 20], [40, 40],
                                            [80, 80]]],
                                          dtype=np.float32)

        # setting bounding box
        gt_instances.bboxes = np.array([[20, 30, 50, 70]])

        # setting heatmap
        heatmap = torch.randn(10, 100, 100) * 0.05
        for i in range(10):
            heatmap[i][i * 10:(i + 1) * 10, i * 10:(i + 1) * 10] += 5
        gt_heatmap = PixelData()
        gt_heatmap.heatmaps = heatmap

        # test gt_sample
        pred_pose_data_sample = PoseDataSample()
        pred_pose_data_sample.gt_instances = gt_instances
        pred_pose_data_sample.gt_fields = gt_heatmap
        pred_instances = gt_instances.clone()
        pred_instances.scores = np.array([[0.9, 0.4, 1.7, -0.2]],
                                         dtype=np.float32)
        pred_pose_data_sample.pred_instances = pred_instances

        pose_local_visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_bbox=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        pose_local_visualizer.show_keypoint_weight = False
        pose_local_visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_pred=False,
            draw_heatmap=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, ((h * 2), w, 3))

        pose_local_visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_heatmap=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, ((h * 2), (w * 2), 3))

        return pose_local_visualizer

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)
