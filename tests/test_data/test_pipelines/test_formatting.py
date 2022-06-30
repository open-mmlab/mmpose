# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np
from mmengine.data import InstanceData, PixelData

from mmpose.core import PoseDataSample
from mmpose.datasets.pipelines2 import PackPoseInputs


class TestPackPoseInputs(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        # prepare dummy top-down data sample with COCO metainfo
        self.results_topdown = {
            'img_id': 1,
            'img_path': 'tests/data/coco/000000000785.jpg',
            'id': 1,
            'ori_shape': (425, 640),
            'img_shape': (425, 640, 3),
            'scale_factor': 2.0,
            'flip': False,
            'flip_direction': None,
            'img': np.zeros((425, 640, 3), dtype=np.uint8),
            'bbox': np.array([[0, 0, 100, 100]], dtype=np.float32),
            'bbox_center': np.array([[50, 50]], dtype=np.float32),
            'bbox_scale': np.array([[125, 125]], dtype=np.float32),
            'bbox_rotation': np.array([45], dtype=np.float32),
            'bbox_score': np.ones(1, dtype=np.float32),
            'keypoints': np.random.randint(0, 100,
                                           (1, 17, 2)).astype(np.float32),
            'keypoints_visible': np.full((1, 17), 1).astype(np.float32),
            'reg_label': np.random.randint(0, 100,
                                           (1, 17, 2)).astype(np.float32),
            'target_weight': np.full((1, 17), 1).astype(np.float32),
            'heatmap': np.random.random((17, 64, 48)).astype(np.float32),
        }
        self.meta_keys = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                          'scale_factor', 'flip', 'flip_direction')

    def test_transform(self):
        transform = PackPoseInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results_topdown))
        self.assertIn('inputs', results)
        self.assertIn('data_sample', results)
        self.assertIsInstance(results['data_sample'], PoseDataSample)
        self.assertIsInstance(results['data_sample'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_sample'].gt_fields, PixelData)
        self.assertEqual(len(results['data_sample'].gt_instances), 1)
        self.assertIsInstance(results['data_sample'].gt_fields.heatmaps,
                              np.ndarray)

    def test_repr(self):
        transform = PackPoseInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackPoseInputs(meta_keys={self.meta_keys})')
