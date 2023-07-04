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
from mmengine.infer.infer import BaseInferencer

from mmpose.apis.inferencers import Pose2DInferencer
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules


class TestPose2DInferencer(TestCase):

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
        inferencer = Pose2DInferencer(
            model='configs/body_2d_keypoint/simcc/coco/'
            'simcc_res50_8xb64-210e_coco-256x192.py',
            weights='https://download.openmmlab.com/mmpose/'
            'v1/body_2d_keypoint/simcc/coco/'
            'simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.detector, BaseInferencer)
        self.assertSequenceEqual(inferencer.det_cat_ids, (0, ))

        # 2. init with config name
        inferencer = Pose2DInferencer(
            model='td-hm_res50_8xb32-210e_onehand10k-256x256',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=0 if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.detector, BaseInferencer)
        self.assertSequenceEqual(inferencer.det_cat_ids, (0, ))

        # 3. init with alias
        inferencer = Pose2DInferencer(
            model='animal',
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=(15, 16, 17, 18, 19, 20, 21, 22,
                         23) if det_model else None)
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertIsInstance(inferencer.detector, BaseInferencer)
        self.assertSequenceEqual(inferencer.det_cat_ids,
                                 (15, 16, 17, 18, 19, 20, 21, 22, 23))

        # 4. init with bottom-up model
        inferencer = Pose2DInferencer(
            model='configs/body_2d_keypoint/dekr/coco/'
            'dekr_hrnet-w32_8xb10-140e_coco-512x512.py',
            weights='https://download.openmmlab.com/mmpose/v1/'
            'body_2d_keypoint/dekr/coco/'
            'dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth',
        )
        self.assertIsInstance(inferencer.model, torch.nn.Module)
        self.assertFalse(hasattr(inferencer, 'detector'))

    def test_call(self):

        try:
            from mmdet.apis.det_inferencer import DetInferencer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return unittest.skip('mmdet is not installed')

        # top-down model
        det_model, det_weights = self._get_det_model_weights()
        inferencer = Pose2DInferencer(
            'human', det_model=det_model, det_weights=det_weights)

        img_path = 'tests/data/coco/000000197388.jpg'
        img = mmcv.imread(img_path)

        # `inputs` is path to an image
        inputs = img_path
        results1 = next(inferencer(inputs, return_vis=True))
        self.assertIn('visualization', results1)
        self.assertSequenceEqual(results1['visualization'][0].shape, img.shape)
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
        inputs = 'tests/data/posetrack18/videos/000001_mpiinew_test/' \
                 '000001_mpiinew_test.mp4'
        with TemporaryDirectory() as tmp_dir:
            results = defaultdict(list)
            for res in inferencer(inputs, out_dir=tmp_dir):
                for key in res:
                    results[key].extend(res[key])
            self.assertIn('000001_mpiinew_test.mp4',
                          os.listdir(f'{tmp_dir}/visualizations'))
            self.assertIn('000001_mpiinew_test.json',
                          os.listdir(f'{tmp_dir}/predictions'))
        self.assertTrue(inferencer._video_input)
        self.assertIn(len(results['predictions']), (4, 5))
