# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData, PixelData
from torch import nn

from mmpose.models.heads import VisPredictHead
from mmpose.testing import get_packed_inputs


class TestVisPredictHead(TestCase):

    def _get_feats(
        self,
        batch_size: int = 2,
        feat_shapes: List[Tuple[int, int, int]] = [(32, 8, 6)],
    ):
        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]
        return feats

    def test_init(self):
        codec = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)

        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                deconv_out_channels=None,
                loss=dict(type='KeypointMSELoss', use_target_weight=True),
                decoder=codec))

        self.assertTrue(isinstance(head.vis_head, nn.Sequential))
        self.assertEqual(head.vis_head[2].weight.shape, (17, 32))
        self.assertIsNotNone(head.pose_head)

    def test_forward(self):

        codec = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)

        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                deconv_out_channels=None,
                loss=dict(type='KeypointMSELoss', use_target_weight=True),
                decoder=codec))

        feats = [torch.rand(1, 32, 128, 128)]
        output_pose, output_vis = head.forward(feats)

        self.assertIsInstance(output_pose, torch.Tensor)
        self.assertEqual(output_pose.shape, (1, 17, 128, 128))

        self.assertIsInstance(output_vis, torch.Tensor)
        self.assertEqual(output_vis.shape, (1, 17))

    def test_predict(self):

        codec = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)

        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                deconv_out_channels=None,
                loss=dict(type='KeypointMSELoss', use_target_weight=True),
                decoder=codec))

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 128, 128)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']

        preds, _ = head.predict(feats, batch_data_samples)

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)
        self.assertEqual(
            preds[0].keypoint_scores.shape,
            batch_data_samples[0].gt_instance_labels.keypoint_weights.shape)

        # output heatmap
        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                decoder=codec))
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        _, pred_heatmaps = head.predict(
            feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

        self.assertIsInstance(pred_heatmaps[0], PixelData)
        self.assertEqual(pred_heatmaps[0].heatmaps.shape, (17, 64, 48))

    def test_tta(self):
        # flip test: vis and heatmap
        decoder_cfg = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)

        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                decoder=decoder_cfg))

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        preds, _ = head.predict([feats, feats],
                                batch_data_samples,
                                test_cfg=dict(
                                    flip_test=True,
                                    flip_mode='heatmap',
                                    shift_heatmap=True,
                                ))

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)
        self.assertEqual(
            preds[0].keypoint_scores.shape,
            batch_data_samples[0].gt_instance_labels.keypoint_weights.shape)

    def test_loss(self):
        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
            ))

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        losses = head.loss(feats, batch_data_samples)
        self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
        self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], torch.Tensor)

        self.assertIsInstance(losses['loss_vis'], torch.Tensor)
        self.assertEqual(losses['loss_vis'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_vis'], torch.Tensor)

        head = VisPredictHead(
            pose_cfg=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
            ))

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        losses = head.loss(feats, batch_data_samples)
        self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
        self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], torch.Tensor)

        self.assertIsInstance(losses['loss_vis'], torch.Tensor)
        self.assertEqual(losses['loss_vis'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_vis'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
