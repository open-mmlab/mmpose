# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData

from mmpose.datasets.transforms import PackPoseInputs
from mmpose.structures import PoseDataSample


class TestPackPoseInputs(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        # prepare dummy top-down data sample with COCO metainfo
        self.results_topdown = {
            'img_id':
            1,
            'img_path':
            'tests/data/coco/000000000785.jpg',
            'id':
            1,
            'ori_shape': (425, 640),
            'img_shape': (425, 640, 3),
            'scale_factor':
            2.0,
            'flip':
            False,
            'flip_direction':
            None,
            'img':
            np.zeros((425, 640, 3), dtype=np.uint8),
            'bbox':
            np.array([[0, 0, 100, 100]], dtype=np.float32),
            'bbox_center':
            np.array([[50, 50]], dtype=np.float32),
            'bbox_scale':
            np.array([[125, 125]], dtype=np.float32),
            'bbox_rotation':
            np.array([45], dtype=np.float32),
            'bbox_score':
            np.ones(1, dtype=np.float32),
            'keypoints':
            np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            'keypoints_visible':
            np.full((1, 17), 1).astype(np.float32),
            'keypoint_weights':
            np.full((1, 17), 1).astype(np.float32),
            'heatmaps':
            np.random.random((17, 64, 48)).astype(np.float32),
            'keypoint_labels':
            np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            'keypoint_x_labels':
            np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            'keypoint_y_labels':
            np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            'transformed_keypoints':
            np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
        }
        self.meta_keys = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                          'scale_factor', 'flip', 'flip_direction')

    def test_transform(self):
        transform = PackPoseInputs(
            meta_keys=self.meta_keys, pack_transformed=True)
        results = transform(copy.deepcopy(self.results_topdown))
        self.assertIn('transformed_keypoints',
                      results['data_samples'].gt_instances)

        transform = PackPoseInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results_topdown))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (3, 425, 640))
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], PoseDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_samples'].gt_fields, PixelData)
        self.assertEqual(len(results['data_samples'].gt_instances), 1)
        self.assertIsInstance(results['data_samples'].gt_fields.heatmaps,
                              torch.Tensor)
        self.assertNotIn('transformed_keypoints',
                         results['data_samples'].gt_instances)

        # test when results['img'] is sequence of frames
        results = copy.deepcopy(self.results_topdown)
        len_seq = 5
        results['img'] = [
            np.random.randint(0, 255, (425, 640, 3), dtype=np.uint8)
            for _ in range(len_seq)
        ]
        results = transform(results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        # translate into 4-dim tensor: [len_seq, c, h, w]
        self.assertEqual(results['inputs'].shape, (len_seq, 3, 425, 640))

    def test_repr(self):
        transform = PackPoseInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackPoseInputs(meta_keys={self.meta_keys})')
