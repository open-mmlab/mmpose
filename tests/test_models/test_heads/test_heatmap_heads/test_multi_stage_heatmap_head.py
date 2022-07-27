# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import torch
from torch import Tensor, nn

from mmpose.core.data_structures.pose_data_sample import PoseDataSample
from mmpose.models.heads import (MultiStageHeatmapHead,
                                 MultiStageMultiUnitHeatmapHead)
from mmpose.testing import get_packed_inputs


class TestMultiStageHeatmapHead(TestCase):

    def _get_feats(self,
                   batch_size: int = 2,
                   feat_shapes: List[Tuple[int, int, int]] = [(17, 32, 24)]):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]
        return feats

    def _get_data_samples(self, batch_size: int = 2):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size=batch_size,
                num_instances=1,
                num_keypoints=17,
                image_shape=(3, 128, 128),
                input_size=(192, 256),
                heatmap_size=(24, 32),
                with_heatmap=True,
                with_reg_label=False)
        ]
        return batch_data_samples

    def test_init(self):
        # w/o deconv
        head = MultiStageHeatmapHead(
            num_stages=1,
            in_channels=256,
            out_channels=17,
            deconv_out_channels=None)
        self.assertTrue(isinstance(head.multi_deconv_layers, nn.ModuleList))
        self.assertTrue(isinstance(head.multi_deconv_layers[0], nn.Identity))

        # w/ deconv
        head = MultiStageHeatmapHead(
            num_stages=1,
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4))
        self.assertTrue(isinstance(head.multi_deconv_layers, nn.ModuleList))
        self.assertTrue(isinstance(head.multi_deconv_layers[0], nn.Sequential))

        # w/o final layer
        head = MultiStageHeatmapHead(
            num_stages=6,
            in_channels=17,
            out_channels=17,
            has_final_layer=False)
        self.assertTrue(isinstance(head.multi_final_layers, nn.ModuleList))
        self.assertTrue(isinstance(head.multi_final_layers[0], nn.Identity))

        # w/ decoder
        head = MultiStageHeatmapHead(
            num_stages=1,
            in_channels=32,
            out_channels=17,
            decoder=dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.))
        self.assertIsNotNone(head.decoder)

        # the same loss for different stages
        head = MultiStageHeatmapHead(
            num_stages=6,
            in_channels=17,
            out_channels=17,
            has_final_layer=False,
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
        )
        self.assertTrue(isinstance(head.loss_module, nn.Module))

        # different loss for different stages
        num_stages = 6
        head = MultiStageHeatmapHead(
            num_stages=num_stages,
            in_channels=17,
            out_channels=17,
            has_final_layer=False,
            loss=[dict(type='KeypointMSELoss', use_target_weight=True)] * 6,
        )
        self.assertTrue(isinstance(head.loss_module, nn.Sequential))
        self.assertTrue(len(head.loss_module), num_stages)

    def test_predict(self):
        decoder_cfg = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(24, 32),
            sigma=2.)

        # num_stages = 6, has_final_layer = False
        head = MultiStageHeatmapHead(
            num_stages=6,
            in_channels=17,
            out_channels=17,
            has_final_layer=False,
            decoder=decoder_cfg)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_stages`'):
            feats = self._get_feats(batch_size=2, feat_shapes=[(17, 32, 24)])
            batch_data_samples = self._get_data_samples(batch_size=2)
            _ = head.predict(feats, batch_data_samples)

        feats = self._get_feats(batch_size=2, feat_shapes=[(17, 32, 24)] * 6)
        batch_data_samples = self._get_data_samples(batch_size=2)
        preds = head.predict(feats, batch_data_samples)

        self.assertEqual(len(preds), 2)
        self.assertIsInstance(preds[0], PoseDataSample)
        self.assertIn('pred_instances', preds[0])
        self.assertEqual(preds[0].pred_instances.keypoints.shape,
                         preds[0].gt_instances.keypoints.shape)
        # output_heatmaps = False
        self.assertNotIn('pred_fields', preds[0])

        # num_stages = 1, has_final_layer = True
        head = MultiStageHeatmapHead(
            num_stages=1,
            in_channels=32,
            out_channels=17,
            has_final_layer=True,
            decoder=decoder_cfg)
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 32, 24)])
        batch_data_samples = self._get_data_samples(batch_size=2)
        preds = head.predict(
            feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

        self.assertEqual(len(preds), 2)
        self.assertIsInstance(preds[0], PoseDataSample)
        self.assertIn('pred_instances', preds[0])
        self.assertEqual(preds[0].pred_instances.keypoints.shape,
                         preds[0].gt_instances.keypoints.shape)
        self.assertIn('pred_fields', preds[0])
        self.assertEqual(preds[0].pred_fields.heatmaps.shape, (17, 32, 24))

    def test_loss(self):
        # num_stages = 1
        head = MultiStageHeatmapHead(
            num_stages=1,
            in_channels=32,
            out_channels=17,
            has_final_layer=True)

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 32, 24)])
        batch_data_samples = self._get_data_samples(batch_size=2)
        losses = head.loss(feats, batch_data_samples)

        self.assertIsInstance(losses['loss_heatmap'], torch.Tensor)
        self.assertEqual(losses['loss_heatmap'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], float)

        num_stages = 6
        head = MultiStageHeatmapHead(
            num_stages=num_stages,
            in_channels=17,
            out_channels=17,
            has_final_layer=False,
            loss=[dict(type='KeypointMSELoss', use_target_weight=True)] *
            num_stages)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_stages`'):
            feats = self._get_feats(batch_size=2, feat_shapes=[(17, 32, 24)])
            batch_data_samples = self._get_data_samples(batch_size=2)
            _ = head.loss(feats, batch_data_samples)

        feats = self._get_feats(
            batch_size=2, feat_shapes=[(17, 32, 24)] * num_stages)
        batch_data_samples = self._get_data_samples(batch_size=2)
        losses = head.loss(feats, batch_data_samples)

        self.assertIsInstance(losses['loss_heatmap'], torch.Tensor)
        self.assertEqual(losses['loss_heatmap'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], float)

    def test_errors(self):
        # Invalid arguments
        with self.assertRaisesRegex(ValueError, 'Got unmatched values'):
            _ = MultiStageHeatmapHead(
                num_stages=1,
                in_channels=17,
                out_channels=17,
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(4, 4))

        with self.assertRaisesRegex(ValueError, 'Unsupported kernel size'):
            _ = MultiStageHeatmapHead(
                num_stages=1,
                in_channels=17,
                out_channels=17,
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(1, ))

        with self.assertRaisesRegex(ValueError, 'did not match `num_stages`'):
            _ = MultiStageHeatmapHead(
                num_stages=6,
                in_channels=17,
                out_channels=17,
                loss=[dict(type='KeypointMSELoss', use_target_weight=True)] *
                4)


class TestMultiStageMultiUnitHeatmapHead(TestCase):

    def _get_feats(
        self,
        num_stages: int = 1,
        num_units: int = 4,
        batch_size: int = 2,
        feat_shapes: List[Tuple[int, int, int]] = [(17, 64, 48)]
    ) -> List[List[Tensor]]:
        feats_stages = []
        for i in range(num_stages):
            feats_units = []
            for j in range(num_units):
                feats_units.append(
                    torch.rand(
                        (batch_size, ) + feat_shapes[j], dtype=torch.float32))
            feats_stages.append(feats_units)

        return feats_stages

    def _get_data_samples(self,
                          batch_size: int = 2,
                          heatmap_size=(48, 64),
                          num_levels=1):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size=batch_size,
                num_instances=1,
                num_keypoints=17,
                image_shape=(3, 128, 128),
                input_size=(192, 256),
                heatmap_size=heatmap_size,
                with_heatmap=True,
                with_reg_label=False,
                num_levels=num_levels)
        ]
        return batch_data_samples

    def test_init(self):
        # w/ decoder
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=1,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=256,
            out_channels=17,
            use_prm=False,
            norm_cfg=dict(type='BN'),
            decoder=dict(
                type='MegviiHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                kernel_size=11))
        self.assertIsNotNone(head.decoder)

        # the same loss for different stages
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=1,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=256,
            out_channels=17,
            use_prm=False,
            norm_cfg=dict(type='BN'),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
        )
        self.assertTrue(isinstance(head.loss_module, nn.Module))

        # different loss for different stages and different units
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=2,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=256,
            out_channels=17,
            use_prm=False,
            norm_cfg=dict(type='BN'),
            loss=[dict(type='KeypointMSELoss', use_target_weight=True)] * 8,
        )
        self.assertTrue(isinstance(head.loss_module, nn.Sequential))
        self.assertTrue(len(head.loss_module), 8)

    def test_loss(self):
        # num_stages = 1, num_units = 4, unit_channels = 16
        # the same loss for all different stages and units
        unit_channels = 16
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=1,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=unit_channels,
            out_channels=17)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_stages`'):
            feats = self._get_feats(
                num_stages=2,
                num_units=4,
                batch_size=2,
                feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                             (unit_channels, 32, 24), (unit_channels, 64, 48)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=8)
            _ = head.loss(feats, batch_data_samples)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_units`'):
            feats = self._get_feats(
                num_stages=1,
                num_units=2,
                batch_size=2,
                feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=2)
            _ = head.loss(feats, batch_data_samples)

        with self.assertRaisesRegex(
                AssertionError,
                'number of feature map channels did not match'):
            feats = self._get_feats(
                num_stages=1,
                num_units=4,
                batch_size=2,
                feat_shapes=[(unit_channels * 2, 8, 6),
                             (unit_channels * 2, 16, 12),
                             (unit_channels * 2, 32, 24),
                             (unit_channels * 2, 64, 48)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=4)
            _ = head.loss(feats, batch_data_samples)

        feats = self._get_feats(
            num_stages=1,
            num_units=4,
            batch_size=2,
            feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                         (unit_channels, 32, 24), (unit_channels, 64, 48)])
        batch_data_samples = self._get_data_samples(
            batch_size=2, heatmap_size=(48, 64), num_levels=4)
        losses = head.loss(feats, batch_data_samples)

        self.assertIsInstance(losses['loss_heatmap'], torch.Tensor)
        self.assertEqual(losses['loss_heatmap'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], float)

        # num_stages = 4, num_units = 4, unit_channels = 16
        # different losses for different stages and units
        unit_channels = 16
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=4,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=unit_channels,
            out_channels=17,
            loss=([
                dict(
                    type='KeypointMSELoss',
                    use_target_weight=True,
                    loss_weight=0.25)
            ] * 3 + [
                dict(
                    type='KeypointOHKMMSELoss',
                    use_target_weight=True,
                    loss_weight=0.1)
            ]) * 4)

        feats = self._get_feats(
            num_stages=4,
            num_units=4,
            batch_size=2,
            feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                         (unit_channels, 32, 24), (unit_channels, 64, 48)])
        batch_data_samples = self._get_data_samples(
            batch_size=2, heatmap_size=(48, 64), num_levels=16)
        losses = head.loss(feats, batch_data_samples)

        self.assertIsInstance(losses['loss_heatmap'], torch.Tensor)
        self.assertEqual(losses['loss_heatmap'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], float)

    def test_predict(self):
        decoder_cfg = dict(
            type='MegviiHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            kernel_size=11)

        # num_stages = 1, num_units = 4, unit_channels = 16
        unit_channels = 16
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=1,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=unit_channels,
            out_channels=17,
            decoder=decoder_cfg)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_stages`'):
            feats = self._get_feats(
                num_stages=2,
                num_units=4,
                batch_size=2,
                feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                             (unit_channels, 32, 24), (unit_channels, 64, 48)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=8)
            _ = head.predict(feats, batch_data_samples)

        with self.assertRaisesRegex(
                AssertionError,
                'length of feature maps did not match the `num_units`'):
            feats = self._get_feats(
                num_stages=1,
                num_units=2,
                batch_size=2,
                feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=2)
            _ = head.loss(feats, batch_data_samples)

        with self.assertRaisesRegex(
                AssertionError,
                'number of feature map channels did not match'):
            feats = self._get_feats(
                num_stages=1,
                num_units=4,
                batch_size=2,
                feat_shapes=[(unit_channels * 2, 8, 6),
                             (unit_channels * 2, 16, 12),
                             (unit_channels * 2, 32, 24),
                             (unit_channels * 2, 64, 48)])
            batch_data_samples = self._get_data_samples(
                batch_size=2, heatmap_size=(48, 64), num_levels=4)
            _ = head.loss(feats, batch_data_samples)

        feats = self._get_feats(
            num_stages=1,
            num_units=4,
            batch_size=2,
            feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                         (unit_channels, 32, 24), (unit_channels, 64, 48)])
        batch_data_samples = self._get_data_samples(
            batch_size=2, heatmap_size=(48, 64), num_levels=4)
        preds = head.predict(feats, batch_data_samples)

        self.assertEqual(len(preds), 2)
        self.assertIsInstance(preds[0], PoseDataSample)
        self.assertIn('pred_instances', preds[0])
        self.assertEqual(preds[0].pred_instances.keypoints.shape,
                         preds[0].gt_instances.keypoints.shape)
        # output_heatmaps = False
        self.assertNotIn('pred_fields', preds[0])

        # num_stages = 4, num_units = 4, unit_channels = 16
        unit_channels = 16
        head = MultiStageMultiUnitHeatmapHead(
            num_stages=4,
            num_units=4,
            out_shape=(64, 48),
            unit_channels=unit_channels,
            out_channels=17,
            decoder=decoder_cfg)
        feats = self._get_feats(
            num_stages=4,
            num_units=4,
            batch_size=2,
            feat_shapes=[(unit_channels, 8, 6), (unit_channels, 16, 12),
                         (unit_channels, 32, 24), (unit_channels, 64, 48)])
        batch_data_samples = self._get_data_samples(
            batch_size=2, heatmap_size=(48, 64), num_levels=16)
        preds = head.predict(
            feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

        self.assertEqual(len(preds), 2)
        self.assertIsInstance(preds[0], PoseDataSample)
        self.assertIn('pred_instances', preds[0])
        self.assertEqual(preds[0].pred_instances.keypoints.shape,
                         preds[0].gt_instances.keypoints.shape)
        self.assertIn('pred_fields', preds[0])
        # output_heatmaps = True
        self.assertEqual(preds[0].pred_fields.heatmaps.shape, (17, 64, 48))

    def test_errors(self):
        # Invalid arguments
        with self.assertRaisesRegex(ValueError, 'The length of loss_module'):
            _ = MultiStageMultiUnitHeatmapHead(
                num_stages=2,
                num_units=4,
                out_shape=(64, 48),
                unit_channels=256,
                out_channels=17,
                loss=[dict(type='KeypointMSELoss', use_target_weight=True)] *
                3)
