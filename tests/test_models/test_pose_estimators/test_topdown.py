# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmpose.structures import PoseDataSample
from mmpose.testing import get_packed_inputs, get_pose_estimator_cfg
from mmpose.utils import register_all_modules

configs = [
    'body_2d_keypoint/topdown_heatmap/coco/'
    'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    'configs/body_2d_keypoint/topdown_regression/coco/'
    'td-reg_res50_8xb64-210e_coco-256x192.py',
    'configs/body_2d_keypoint/simcc/coco/'
    'simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192.py',
]

configs_with_devices = [(config, ('cpu', 'cuda')) for config in configs]


class TestTopdownPoseEstimator(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    @parameterized.expand(configs)
    def test_init(self, config):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator
        model = build_pose_estimator(model_cfg)
        self.assertTrue(model.backbone)
        self.assertTrue(model.head)
        if model_cfg.get('neck', None):
            self.assertTrue(model.neck)

    @parameterized.expand(configs_with_devices)
    def test_forward_loss(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2)
            data = model.data_preprocessor(packed_inputs, training=True)
            losses = model.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand(configs_with_devices)
    def test_forward_predict(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2)
            model.eval()
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, training=True)
                batch_results = model.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], PoseDataSample)

    @parameterized.expand(configs_with_devices)
    def test_forward_tensor(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2)
            data = model.data_preprocessor(packed_inputs, training=True)
            batch_results = model.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, (tuple, torch.Tensor))
