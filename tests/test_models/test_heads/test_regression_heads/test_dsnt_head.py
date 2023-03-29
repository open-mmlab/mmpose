# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData, PixelData

from mmpose.models.heads import DSNTHead
from mmpose.testing import get_packed_inputs


class TestDSNTHead(TestCase):

    def _get_feats(
        self,
        batch_size: int = 2,
        feat_shapes: List[Tuple[int, int, int]] = [(32, 6, 8)],
    ):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]

        return feats

    def test_init(self):
        # square heatmap
        head = DSNTHead(
            in_channels=32, in_featuremap_size=(8, 8), num_joints=17)
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 64))
        self.assertEqual(head.linspace_y.shape, (1, 1, 64, 1))
        self.assertIsNone(head.decoder)

        # rectangle heatmap
        head = DSNTHead(
            in_channels=32, in_featuremap_size=(6, 8), num_joints=17)
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 6 * 8))
        self.assertEqual(head.linspace_y.shape, (1, 1, 8 * 8, 1))
        self.assertIsNone(head.decoder)

        # 2 deconv + 1x1 conv
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(32, ),
            conv_kernel_sizes=(1, ),
        )
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 6 * 4))
        self.assertEqual(head.linspace_y.shape, (1, 1, 8 * 4, 1))
        self.assertIsNone(head.decoder)

        # 2 deconv + w/o 1x1 conv
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(32, ),
            conv_kernel_sizes=(1, ),
            final_layer=None,
        )
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 6 * 4))
        self.assertEqual(head.linspace_y.shape, (1, 1, 8 * 4, 1))
        self.assertIsNone(head.decoder)

        # w/o deconv and 1x1 conv
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            deconv_out_channels=tuple(),
            deconv_kernel_sizes=tuple(),
            final_layer=None,
        )
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 6))
        self.assertEqual(head.linspace_y.shape, (1, 1, 8, 1))
        self.assertIsNone(head.decoder)

        # w/o deconv and 1x1 conv
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            deconv_out_channels=None,
            deconv_kernel_sizes=None,
            final_layer=None,
        )
        self.assertEqual(head.linspace_x.shape, (1, 1, 1, 6))
        self.assertEqual(head.linspace_y.shape, (1, 1, 8, 1))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = DSNTHead(
            in_channels=1024,
            in_featuremap_size=(6, 8),
            num_joints=17,
            decoder=dict(
                type='IntegralRegressionLabel',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2))

        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg = dict(
            type='IntegralRegressionLabel',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)

        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_reg_label=False)['data_samples']
        preds = head.predict(feats, batch_data_samples)

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

        # output heatmap
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_reg_label=False)['data_samples']
        _, pred_heatmaps = head.predict(
            feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

        self.assertTrue(len(pred_heatmaps), 2)
        self.assertIsInstance(pred_heatmaps[0], PixelData)
        self.assertEqual(pred_heatmaps[0].heatmaps.shape, (17, 8 * 8, 6 * 8))

    def test_tta(self):
        decoder_cfg = dict(
            type='IntegralRegressionLabel',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)

        # inputs transform: select
        head = DSNTHead(
            in_channels=32,
            in_featuremap_size=(6, 8),
            num_joints=17,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_reg_label=False)['data_samples']
        preds = head.predict([feats, feats],
                             batch_data_samples,
                             test_cfg=dict(flip_test=True))

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

    def test_loss(self):
        for dist_loss in ['l1', 'l2']:
            head = DSNTHead(
                in_channels=32,
                in_featuremap_size=(6, 8),
                num_joints=17,
                loss=dict(
                    type='MultipleLossWrapper',
                    losses=[
                        dict(type='SmoothL1Loss', use_target_weight=True),
                        dict(type='JSDiscretLoss', use_target_weight=True)
                    ]))

            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2, with_reg_label=True)['data_samples']
            losses = head.loss(feats, batch_data_samples)

            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size())
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
