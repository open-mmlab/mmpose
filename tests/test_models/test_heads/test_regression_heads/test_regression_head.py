# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmpose.models.heads import RegressionHead
from mmpose.testing import get_packed_inputs


class TestRegressionHead(TestCase):

    def _get_feats(
        self,
        batch_size: int = 2,
        feat_shapes: List[Tuple[int, int, int]] = [(32, 1, 1)],
    ):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]

        return feats

    def test_init(self):

        head = RegressionHead(in_channels=1024, num_joints=17)
        self.assertEqual(head.fc.weight.shape, (17 * 2, 1024))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = RegressionHead(
            in_channels=1024,
            num_joints=17,
            decoder=dict(type='RegressionLabel', input_size=(192, 256)),
        )
        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg = dict(type='RegressionLabel', input_size=(192, 256))

        head = RegressionHead(
            in_channels=32,
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 1, 1)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_heatmap=False)['data_samples']
        preds = head.predict(feats, batch_data_samples)

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

    def test_tta(self):
        decoder_cfg = dict(type='RegressionLabel', input_size=(192, 256))

        # inputs transform: select
        head = RegressionHead(
            in_channels=32,
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 1, 1)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_heatmap=False)['data_samples']
        preds = head.predict([feats, feats],
                             batch_data_samples,
                             test_cfg=dict(flip_test=True, shift_coords=True))

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

    def test_loss(self):
        head = RegressionHead(
            in_channels=32,
            num_joints=17,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 1, 1)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_heatmap=False)['data_samples']
        losses = head.loss(feats, batch_data_samples)

        self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
        self.assertEqual(losses['loss_kpt'].shape, torch.Size())
        self.assertIsInstance(losses['acc_pose'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
