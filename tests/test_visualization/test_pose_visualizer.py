# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData

from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


class TestPoseLocalVisualizer(TestCase):

    def setUp(self):
        self.visualizer = PoseLocalVisualizer(show_keypoint_weight=True)

    def _get_dataset_meta(self):
        # None: kpt or link is hidden
        pose_kpt_color = [None] + [(127, 127, 127)] * 2 + ['red']
        pose_link_color = [(127, 127, 127)] * 2 + [None]
        skeleton_links = [[0, 1], [1, 2], [2, 3]]
        return {
            'keypoint_colors': pose_kpt_color,
            'skeleton_link_colors': pose_link_color,
            'skeleton_links': skeleton_links
        }

    def test_set_dataset_meta(self):
        dataset_meta = self._get_dataset_meta()
        self.visualizer.set_dataset_meta(dataset_meta)
        self.assertEqual(len(self.visualizer.kpt_color), 4)
        self.assertEqual(self.visualizer.kpt_color[-1], 'red')
        self.assertListEqual(self.visualizer.skeleton[-1], [2, 3])

        self.visualizer.dataset_meta = None
        self.visualizer.set_dataset_meta(dataset_meta)
        self.assertIsNotNone(self.visualizer.dataset_meta)

    def test_add_datasample(self):
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.uint8)
        out_file = 'out_file.jpg'

        dataset_meta = self._get_dataset_meta()
        self.visualizer.set_dataset_meta(dataset_meta)

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

        self.visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_bbox=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        self.visualizer.show_keypoint_weight = False
        self.visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_pred=False,
            draw_heatmap=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, ((h * 2), w, 3))

        self.visualizer.add_datasample(
            'image',
            image,
            data_sample=pred_pose_data_sample,
            draw_heatmap=True,
            out_file=out_file)
        self._assert_image_and_shape(out_file, ((h * 2), (w * 2), 3))

    def test_simcc_visualization(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        heatmap = torch.randn([17, 512, 512])
        pixelData = PixelData()
        pixelData.heatmaps = heatmap
        self.visualizer._draw_instance_xy_heatmap(pixelData, img, 10)

    def _assert_image_and_shape(self, out_file, out_shape):
        self.assertTrue(os.path.exists(out_file))
        drawn_img = cv2.imread(out_file)
        self.assertTupleEqual(drawn_img.shape, out_shape)
        os.remove(out_file)
