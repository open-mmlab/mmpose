# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.model.utils import revert_sync_batchnorm
from parameterized import parameterized

from mmpose.structures import PoseDataSample
from mmpose.testing import get_packed_inputs, get_pose_estimator_cfg
from mmpose.utils import register_all_modules

configs = [
    'wholebody_2d_keypoint/dwpose/ubody/'
    's1_dis/dwpose_l_dis_m_coco-ubody-256x192.py',
    'wholebody_2d_keypoint/dwpose/ubody/'
    's2_dis/dwpose_m-mm_coco-ubody-256x192.py',
    'wholebody_2d_keypoint/dwpose/coco-wholebody/'
    's1_dis/dwpose_l_dis_m_coco-256x192.py',
    'wholebody_2d_keypoint/dwpose/coco-wholebody/'
    's2_dis/dwpose_m-mm_coco-256x192.py',
]

configs_with_devices = [(config, ('cpu', 'cuda')) for config in configs]


class TestDWPoseDistiller(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    @parameterized.expand(configs)
    def test_init(self, config):
        dis_cfg = get_pose_estimator_cfg(config)
        model_cfg = get_pose_estimator_cfg(dis_cfg.student_cfg)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator
        model = build_pose_estimator(model_cfg)
        model = revert_sync_batchnorm(model)
        self.assertTrue(model.backbone)
        self.assertTrue(model.head)
        if model_cfg.get('neck', None):
            self.assertTrue(model.neck)

    @parameterized.expand(configs_with_devices)
    def test_forward_loss(self, config, devices):
        dis_cfg = get_pose_estimator_cfg(config)
        model_cfg = get_pose_estimator_cfg(dis_cfg.student_cfg)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)
            model = revert_sync_batchnorm(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2, num_keypoints=133)
            data = model.data_preprocessor(packed_inputs, training=True)
            losses = model.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand(configs_with_devices)
    def test_forward_predict(self, config, devices):
        dis_cfg = get_pose_estimator_cfg(config)
        model_cfg = get_pose_estimator_cfg(dis_cfg.student_cfg)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)
            model = revert_sync_batchnorm(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2, num_keypoints=133)
            model.eval()
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, training=True)
                batch_results = model.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], PoseDataSample)

    @parameterized.expand(configs_with_devices)
    def test_forward_tensor(self, config, devices):
        dis_cfg = get_pose_estimator_cfg(config)
        model_cfg = get_pose_estimator_cfg(dis_cfg.student_cfg)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)
            model = revert_sync_batchnorm(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2, num_keypoints=133)
            data = model.data_preprocessor(packed_inputs, training=True)
            batch_results = model.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, (tuple, torch.Tensor))
