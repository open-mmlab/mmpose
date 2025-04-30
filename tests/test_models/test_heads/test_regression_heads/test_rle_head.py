# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmpose.models.heads import RLEHead
from mmpose.testing import get_packed_inputs


class TestRLEHead(TestCase):

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

        # w/ sigma
        head = RLEHead(in_channels=1024, num_joints=17)
        self.assertEqual(head.fc.weight.shape, (17 * 4, 1024))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = RLEHead(
            in_channels=1024,
            num_joints=17,
            decoder=dict(type='RegressionLabel', input_size=(192, 256)),
        )
        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg = dict(type='RegressionLabel', input_size=(192, 256))

        head = RLEHead(
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

        head = RLEHead(
            in_channels=32,
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 1, 1)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_heatmap=False)['data_samples']
        preds = head.predict([feats, feats],
                             batch_data_samples,
                             test_cfg=dict(flip_test=True))

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

    def test_loss(self):
        head = RLEHead(
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

    def test_state_dict_compatible(self):

        head = RLEHead(in_channels=2048, num_joints=17)

        state_dict = {
            'fc.weight': torch.zeros((17 * 4, 2048)),
            'fc.bias': torch.zeros((17 * 4)),
            'loss.flow_model.loc': torch.zeros(torch.Size([2])),
            'loss.flow_model.cov': torch.zeros(torch.Size([2, 2])),
            'loss.flow_model.mask': torch.zeros(torch.Size([6, 2])),
            'loss.flow_model.s.0.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.0.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.0.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.0.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.0.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.0.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.s.1.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.1.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.1.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.1.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.1.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.1.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.s.2.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.2.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.2.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.2.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.2.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.2.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.s.3.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.3.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.3.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.3.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.3.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.3.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.s.4.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.4.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.4.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.4.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.4.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.4.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.s.5.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.s.5.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.5.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.s.5.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.s.5.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.s.5.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.0.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.0.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.0.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.0.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.0.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.0.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.1.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.1.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.1.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.1.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.1.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.1.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.2.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.2.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.2.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.2.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.2.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.2.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.3.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.3.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.3.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.3.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.3.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.3.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.4.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.4.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.4.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.4.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.4.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.4.4.bias': torch.zeros(torch.Size([2])),
            'loss.flow_model.t.5.0.weight': torch.zeros(torch.Size([64, 2])),
            'loss.flow_model.t.5.0.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.5.2.weight': torch.zeros(torch.Size([64, 64])),
            'loss.flow_model.t.5.2.bias': torch.zeros(torch.Size([64])),
            'loss.flow_model.t.5.4.weight': torch.zeros(torch.Size([2, 64])),
            'loss.flow_model.t.5.4.bias': torch.zeros(torch.Size([2]))
        }
        head.load_state_dict(state_dict)


if __name__ == '__main__':
    unittest.main()
