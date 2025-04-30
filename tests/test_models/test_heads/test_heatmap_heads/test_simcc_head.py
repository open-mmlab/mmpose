# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmpose.models.heads import SimCCHead
from mmpose.testing import get_packed_inputs


class TestSimCCHead(TestCase):

    def _get_feats(self,
                   batch_size: int = 2,
                   feat_shapes: List[Tuple[int, int, int]] = [(32, 6, 8)]):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]
        return feats

    def test_init(self):

        # w/ gaussian decoder
        head = SimCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='gaussian',
                sigma=6.,
                simcc_split_ratio=2.0))
        self.assertIsNotNone(head.decoder)

        # w/ label-smoothing decoder
        head = SimCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=3.0,
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='standard',
                sigma=6.,
                simcc_split_ratio=3.0,
                label_smooth_weight=0.1))
        self.assertIsNotNone(head.decoder)

        # w/ one-hot decoder
        head = SimCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=3.0,
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='standard',
                sigma=6.,
                simcc_split_ratio=3.0))
        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg1 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=2.,
            simcc_split_ratio=2.0)

        decoder_cfg2 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='standard',
            sigma=2.,
            simcc_split_ratio=2.0)

        decoder_cfg3 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='standard',
            sigma=2.,
            simcc_split_ratio=2.0,
            label_smooth_weight=0.1)

        for decoder_cfg in [decoder_cfg1, decoder_cfg2, decoder_cfg3]:

            head = SimCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # input transform: output heatmap
            head = SimCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, pred_heatmaps = head.predict(
                feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

            self.assertEqual(preds[0].keypoint_x_labels.shape,
                             (1, 17, 192 * 2))
            self.assertEqual(preds[0].keypoint_y_labels.shape,
                             (1, 17, 256 * 2))
            self.assertTrue(len(pred_heatmaps), 2)
            self.assertEqual(pred_heatmaps[0].heatmaps.shape, (17, 512, 384))

    def test_tta(self):
        # flip test
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=2.,
            simcc_split_ratio=2.0)

        head = SimCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            decoder=decoder_cfg)
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, simcc_split_ratio=2.0,
            with_simcc_label=True)['data_samples']
        preds = head.predict([feats, feats],
                             batch_data_samples,
                             test_cfg=dict(flip_test=True))

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

    def test_loss(self):
        decoder_cfg1 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=2.,
            simcc_split_ratio=2.0)

        decoder_cfg2 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='standard',
            sigma=2.,
            simcc_split_ratio=2.0)

        decoder_cfg3 = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='standard',
            sigma=2.,
            simcc_split_ratio=2.0,
            label_smooth_weight=0.1)

        # decoder
        for decoder_cfg in [decoder_cfg1, decoder_cfg2, decoder_cfg3]:
            head = SimCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                decoder=decoder_cfg)

            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2, simcc_split_ratio=2.0,
                with_simcc_label=True)['data_samples']
            losses = head.loss(feats, batch_data_samples)
            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)

    def test_errors(self):
        # Invalid arguments
        with self.assertRaisesRegex(ValueError, 'Got mismatched lengths'):
            _ = SimCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(48, 64),
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(4, 4))

        with self.assertRaisesRegex(ValueError, 'Got mismatched lengths'):
            _ = SimCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(48, 64),
                conv_out_channels=(256, ),
                conv_kernel_sizes=(1, 1))

        with self.assertRaisesRegex(ValueError, 'Unsupported kernel size'):
            _ = SimCCHead(
                in_channels=16,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(48, 64),
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(1, ))


if __name__ == '__main__':
    unittest.main()
