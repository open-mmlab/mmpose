# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from mmengine.structures import InstanceData

from mmpose.evaluation import MPJPE
from mmpose.structures import PoseDataSample


class TestMPJPE(TestCase):

    def setUp(self):
        """Setup variables used in every test method."""
        self.batch_size = 8
        num_keypoints = 15
        self.data_batch = []
        self.data_samples = []

        for i in range(self.batch_size):
            gt_instances = InstanceData()
            keypoints = np.random.random((1, num_keypoints, 3))
            gt_instances.lifting_target = np.random.random((num_keypoints, 3))
            gt_instances.lifting_target_visible = np.ones(
                (num_keypoints, 1)).astype(bool)

            pred_instances = InstanceData()
            pred_instances.keypoints = keypoints + np.random.normal(
                0, 0.01, keypoints.shape)

            data = {'inputs': None}
            data_sample = PoseDataSample(
                gt_instances=gt_instances, pred_instances=pred_instances)
            data_sample.set_metainfo(
                dict(target_img_path='tests/data/h36m/S7/'
                     'S7_Greeting.55011271/S7_Greeting.55011271_000396.jpg'))

            self.data_batch.append(data)
            self.data_samples.append(data_sample.to_dict())

    def test_init(self):
        """Test metric init method."""
        # Test invalid mode
        with self.assertRaisesRegex(
                KeyError, "`mode` should be 'mpjpe', 'p-mpjpe', or 'n-mpjpe', "
                "but got 'invalid'."):
            MPJPE(mode='invalid')

    def test_evaluate(self):
        """Test MPJPE evaluation metric."""
        mpjpe_metric = MPJPE(mode='mpjpe')
        mpjpe_metric.process(self.data_batch, self.data_samples)
        mpjpe = mpjpe_metric.evaluate(self.batch_size)
        self.assertIsInstance(mpjpe, dict)
        self.assertIn('MPJPE', mpjpe)
        self.assertTrue(mpjpe['MPJPE'] >= 0)

        p_mpjpe_metric = MPJPE(mode='p-mpjpe')
        p_mpjpe_metric.process(self.data_batch, self.data_samples)
        p_mpjpe = p_mpjpe_metric.evaluate(self.batch_size)
        self.assertIsInstance(p_mpjpe, dict)
        self.assertIn('P-MPJPE', p_mpjpe)
        self.assertTrue(p_mpjpe['P-MPJPE'] >= 0)

        n_mpjpe_metric = MPJPE(mode='n-mpjpe')
        n_mpjpe_metric.process(self.data_batch, self.data_samples)
        n_mpjpe = n_mpjpe_metric.evaluate(self.batch_size)
        self.assertIsInstance(n_mpjpe, dict)
        self.assertIn('N-MPJPE', n_mpjpe)
        self.assertTrue(n_mpjpe['N-MPJPE'] >= 0)
