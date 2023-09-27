# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory
from unittest import TestCase

import mmcv
import torch

from mmpose.apis.inferencers import Hand3DInferencer
from mmpose.structures import PoseDataSample
from mmpose.utils import register_all_modules


class TestHand3DInferencer(TestCase):

    def tearDown(self) -> None:
        register_all_modules(init_default_scope=True)
        return super().tearDown()

    def test_init(self):

        inferencer = Hand3DInferencer(model='hand3d')
        self.assertIsInstance(inferencer.model, torch.nn.Module)

    def test_call(self):

        inferencer = Hand3DInferencer(model='hand3d')

        img_path = 'tests/data/interhand2.6m/image29590.jpg'
        img = mmcv.imread(img_path)

        # `inputs` is path to an image
        inputs = img_path
        results1 = next(inferencer(inputs, return_vis=True))
        self.assertIn('visualization', results1)
        self.assertIn('predictions', results1)
        self.assertIn('keypoints', results1['predictions'][0][0])
        self.assertEqual(len(results1['predictions'][0][0]['keypoints']), 42)

        # `inputs` is an image array
        inputs = img
        results2 = next(inferencer(inputs))
        self.assertEqual(
            len(results1['predictions'][0]), len(results2['predictions'][0]))
        self.assertSequenceEqual(results1['predictions'][0][0]['keypoints'],
                                 results2['predictions'][0][0]['keypoints'])
        results2 = next(inferencer(inputs, return_datasamples=True))
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
                                 results3['predictions'][1][0]['keypoints'])
