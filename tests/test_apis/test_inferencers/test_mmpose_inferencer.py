# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory
from unittest import TestCase

import mmcv

from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.structures import PoseDataSample


class TestMMPoseInferencer(TestCase):

    def test_call(self):

        # top-down model
        inferencer = MMPoseInferencer('human')

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
                                 results3['predictions'][2][0]['keypoints'])

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
