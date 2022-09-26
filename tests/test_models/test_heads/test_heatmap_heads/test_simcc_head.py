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

    def _get_data_samples(self,
                          batch_size: int = 2,
                          simcc_split_ratio: float = 2.0,
                          with_simcc_label=False):

        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                simcc_split_ratio=simcc_split_ratio,
                with_simcc_label=with_simcc_label)
        ]

        return batch_data_samples

    def test_init(self):

        # w/ gaussian decoder
        head = SimCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(8, 6),
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
            in_featuremap_size=(8, 6),
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
            in_featuremap_size=(8, 6),
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
            # input transform: select
            head = SimCCHead(
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(8, 6),
                input_transform='select',
                input_index=-1,
                decoder=decoder_cfg)
            feats = self._get_feats(
                batch_size=2, feat_shapes=[(16, 16, 12), (32, 8, 6)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, simcc_split_ratio=2.0, with_simcc_label=True)
            preds = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # input transform: resize and concat
            head = SimCCHead(
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(16, 12),
                input_transform='resize_concat',
                input_index=[0, 1],
                deconv_out_channels=(256, 256),
                deconv_kernel_sizes=(4, 4),
                conv_out_channels=(256, ),
                conv_kernel_sizes=(1, ),
                decoder=decoder_cfg)
            feats = self._get_feats(
                batch_size=2, feat_shapes=[(16, 16, 12), (32, 8, 6)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, simcc_split_ratio=2.0, with_simcc_label=True)
            preds = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # input transform: output heatmap
            head = SimCCHead(
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(8, 6),
                input_transform='select',
                input_index=-1,
                decoder=decoder_cfg)
            feats = self._get_feats(
                batch_size=2, feat_shapes=[(16, 16, 12), (32, 8, 6)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, simcc_split_ratio=2.0, with_simcc_label=True)
            preds = head.predict(
                feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

            self.assertEqual(preds[0].keypoint_x_labels.shape,
                             (1, 17, 192 * 2))
            self.assertEqual(preds[0].keypoint_y_labels.shape,
                             (1, 17, 256 * 2))

    def test_tta(self):
        # flip test
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=2.,
            simcc_split_ratio=2.0)

        head = SimCCHead(
            in_channels=[16, 32],
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(8, 6),
            input_transform='select',
            input_index=-1,
            decoder=decoder_cfg)
        feats = self._get_feats(
            batch_size=2, feat_shapes=[(16, 16, 12), (32, 8, 6)])
        batch_data_samples = self._get_data_samples(
            batch_size=2, simcc_split_ratio=2.0, with_simcc_label=True)
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
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(8, 6),
                input_transform='select',
                input_index=-1,
                decoder=decoder_cfg)

            feats = self._get_feats(
                batch_size=2, feat_shapes=[(16, 16, 12), (32, 8, 6)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, simcc_split_ratio=2.0, with_simcc_label=True)
            losses = head.loss(feats, batch_data_samples)
            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)

    def test_errors(self):
        # Invalid arguments
        with self.assertRaisesRegex(ValueError, 'Got unmatched values'):
            _ = SimCCHead(
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(48, 64),
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(4, 4))

        with self.assertRaisesRegex(ValueError, 'Got unmatched values'):
            _ = SimCCHead(
                in_channels=[16, 32],
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

        with self.assertRaisesRegex(ValueError,
                                    'selecting multiple input features'):
            _ = SimCCHead(
                in_channels=[16, 32],
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(48, 64),
                input_transform='select',
                input_index=[0, 1],
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(4, ))


if __name__ == '__main__':
    unittest.main()
