# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import unittest
from collections import defaultdict
from tempfile import TemporaryDirectory
from unittest import TestCase

import mmcv
import torch

from mmpose.apis.inferencers import Pose2DInferencer, Pose3DInferencer
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules


class TestPose3DInferencer(TestCase):

    def tearDown(self) -> None:
        register_all_modules(init_default_scope=True)
        return super().tearDown()

    def _get_det_model_weights(self):
        if platform.system().lower() == 'windows':
            # the default human/animal pose estimator utilizes rtmdet-m
            # detector through alias, which seems not compatible with windows
            det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
            det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/' \
                          'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/' \
                          'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        else:
            det_model, det_weights = None, None

        return det_model, det_weights

    def test_init(self):

        try:
            from mmdet.apis.det_inferencer import DetInferencer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return unittest.skip('mmdet is not installed')

        det_model, det_weights = self._get_det_model_weights()

        # 1. init with config path and checkpoint
        inferencer = Pose3DInferencer(
            model=  # noqa
            'configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m.py',  # noqa
            weights=  # noqa
            'https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',  # noqa
            pose2d_model='configs/body_2d_keypoint/simcc/coco/'
            'simcc_res50_8xb64-210e_coco-256x192.py',
            pose2d_weights='https://download.openmmlab.com/mmpose/'
            'v1/body_2d_keypoint/simcc/coco/'
            'simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.pose2d_model, Pose2DInferencer)

        # 2. init with config name
        inferencer = Pose3DInferencer(
            model='configs/body_3d_keypoint/pose_lift/h36m/pose-lift_'
            'videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m.py',
            pose2d_model='configs/body_2d_keypoint/simcc/coco/'
            'simcc_res50_8xb64-210e_coco-256x192.py',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.pose2d_model, Pose2DInferencer)

        # 3. init with alias
        inferencer = Pose3DInferencer(
            model='human3d',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.pose2d_model, Pose2DInferencer)

    def test_call(self):

        try:
            from mmdet.apis.det_inferencer import DetInferencer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return unittest.skip('mmdet is not installed')

        # top-down model
        det_model, det_weights = self._get_det_model_weights()
        inferencer = Pose3DInferencer(
            model='human3d',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)

        img_path = 'tests/data/coco/000000197388.jpg'
        img = mmcv.imread(img_path)

        # `inputs` is path to an image
        inputs = img_path
        results1 = next(inferencer(inputs, return_vis=True))
        self.assertIn('visualization', results1)
        self.assertIn('predictions', results1)
        self.assertIn('keypoints', results1['predictions'][0][0])
        self.assertEqual(len(results1['predictions'][0][0]['keypoints']), 17)

        # `inputs` is an image array
        inputs = img
        results2 = next(inferencer(inputs))
        self.assertEqual(
            len(results1['predictions'][0]), len(results2['predictions'][0]))
        self.assertSequenceEqual(results1['predictions'][0][0]['keypoints'],
                                 results2['predictions'][0][0]['keypoints'])
        results2 = next(inferencer(inputs, return_datasample=True))
        self.assertIsInstance(results2['predictions'][0], PoseDataSample)

        # `inputs` is path to a directory
        inputs = osp.dirname(img_path)

        with TemporaryDirectory() as tmp_dir:
            # only save visualizations
            for res in inferencer(inputs, vis_out_dir=tmp_dir):
                pass
            self.assertEqual(len(os.listdir(tmp_dir)), 4)
            # save both visualizations and predictions
            results3 = defaultdict(list)
            for res in inferencer(inputs, out_dir=tmp_dir):
                for key in res:
                    results3[key].extend(res[key])
            self.assertEqual(len(os.listdir(f'{tmp_dir}/visualizations')), 4)
            self.assertEqual(len(os.listdir(f'{tmp_dir}/predictions')), 4)
        self.assertEqual(len(results3['predictions']), 4)
        self.assertSequenceEqual(results1['predictions'][0][0]['keypoints'],
                                 results3['predictions'][3][0]['keypoints'])

        # `inputs` is path to a video
        inputs = 'https://user-images.githubusercontent.com/87690686/' \
            '164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4'
        with TemporaryDirectory() as tmp_dir:
            results = defaultdict(list)
            for res in inferencer(inputs, out_dir=tmp_dir):
                for key in res:
                    results[key].extend(res[key])
            self.assertIn('164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4',
                          os.listdir(f'{tmp_dir}/visualizations'))
            self.assertIn(
                '164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.json',
                os.listdir(f'{tmp_dir}/predictions'))
        self.assertTrue(inferencer._video_input)
