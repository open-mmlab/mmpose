# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import unittest
from collections import defaultdict
from tempfile import TemporaryDirectory
from unittest import TestCase

import mmcv

from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules


class TestMMPoseInferencer(TestCase):

    def tearDown(self) -> None:
        register_all_modules(init_default_scope=True)
        return super().tearDown()

    def test_pose2d_call(self):
        try:
            from mmdet.apis.det_inferencer import DetInferencer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return unittest.skip('mmdet is not installed')

        # top-down model
        if platform.system().lower() == 'windows':
            # the default human pose estimator utilizes rtmdet-m detector
            # through alias, which seems not compatible with windows
            det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
            det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/' \
                          'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/' \
                          'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        else:
            det_model, det_weights = None, None
        inferencer = MMPoseInferencer(
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

    def test_pose3d_call(self):
        try:
            from mmdet.apis.det_inferencer import DetInferencer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return unittest.skip('mmdet is not installed')

        # top-down model
        if platform.system().lower() == 'windows':
            # the default human pose estimator utilizes rtmdet-m detector
            # through alias, which seems not compatible with windows
            det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
            det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/' \
                          'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/' \
                          'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        else:
            det_model, det_weights = None, None
        inferencer = MMPoseInferencer(
            pose3d='human3d', det_model=det_model, det_weights=det_weights)

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
