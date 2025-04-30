# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData, PixelData
from torch import nn

from mmpose.models.heads import ViPNASHead
from mmpose.testing import get_packed_inputs


class TestViPNASHead(TestCase):

    def _get_feats(self,
                   batch_size: int = 2,
                   feat_shapes: List[Tuple[int, int, int]] = [(32, 6, 8)]):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]
        return feats

    def test_init(self):
        # w/o deconv
        head = ViPNASHead(
            in_channels=32, out_channels=17, deconv_out_channels=None)
        self.assertTrue(isinstance(head.deconv_layers, nn.Identity))

        # w/ deconv and w/o conv
        head = ViPNASHead(
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4),
            deconv_num_groups=(1, 1))
        self.assertTrue(isinstance(head.deconv_layers, nn.Sequential))
        self.assertTrue(isinstance(head.conv_layers, nn.Identity))

        # w/ both deconv and conv
        head = ViPNASHead(
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4),
            deconv_num_groups=(1, 1),
            conv_out_channels=(32, ),
            conv_kernel_sizes=(1, ))
        self.assertTrue(isinstance(head.deconv_layers, nn.Sequential))
        self.assertTrue(isinstance(head.conv_layers, nn.Sequential))

        # w/o final layer
        head = ViPNASHead(in_channels=32, out_channels=17, final_layer=None)
        self.assertTrue(isinstance(head.final_layer, nn.Identity))

        # w/ decoder
        head = ViPNASHead(
            in_channels=32,
            out_channels=17,
            decoder=dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.))
        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)

        head = ViPNASHead(
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(256, 256),
            deconv_kernel_sizes=(4, 4),
            deconv_num_groups=(1, 1),
            conv_out_channels=(256, ),
            conv_kernel_sizes=(1, ),
            decoder=decoder_cfg)
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        preds = head.predict(feats, batch_data_samples)

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape,
                         batch_data_samples[0].gt_instances.keypoints.shape)

        # input transform: output heatmap
        head = ViPNASHead(in_channels=32, out_channels=17, decoder=decoder_cfg)
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        _, pred_heatmaps = head.predict(
            feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

        self.assertTrue(len(pred_heatmaps), 2)
        self.assertIsInstance(pred_heatmaps[0], PixelData)
        self.assertEqual(pred_heatmaps[0].heatmaps.shape, (17, 64, 48))

    def test_tta(self):
        # flip test: heatmap
        decoder_cfg = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.)

        # input transform: select
        head = ViPNASHead(in_channels=32, out_channels=17, decoder=decoder_cfg)
        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        preds = head.predict([feats, feats],
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

    def test_loss(self):
        head = ViPNASHead(in_channels=32, out_channels=17)

        feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
        batch_data_samples = get_packed_inputs(batch_size=2)['data_samples']
        losses = head.loss(feats, batch_data_samples)
        self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
        self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
        self.assertIsInstance(losses['acc_pose'], torch.Tensor)

    def test_errors(self):
        # Invalid arguments
        with self.assertRaisesRegex(ValueError, 'Got mismatched lengths'):
            _ = ViPNASHead(
                in_channels=32,
                out_channels=17,
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(4, 4))
        with self.assertRaisesRegex(ValueError, 'Got mismatched lengths'):
            _ = ViPNASHead(
                in_channels=32,
                out_channels=17,
                deconv_out_channels=(256, 256),
                deconv_kernel_sizes=(4, 4),
                deconv_num_groups=(1, ))

        with self.assertRaisesRegex(ValueError, 'Got mismatched lengths'):
            _ = ViPNASHead(
                in_channels=32,
                out_channels=17,
                conv_out_channels=(256, ),
                conv_kernel_sizes=(1, 1))

        with self.assertRaisesRegex(ValueError, 'Unsupported kernel size'):
            _ = ViPNASHead(
                in_channels=16,
                out_channels=17,
                deconv_out_channels=(256, ),
                deconv_kernel_sizes=(1, ),
                deconv_num_groups=(1, ))

    def test_state_dict_compatible(self):
        # Typical setting for MobileNetV3
        head = ViPNASHead(
            in_channels=160,
            out_channels=17,
            deconv_out_channels=(160, 160, 160),
            deconv_num_groups=(160, 160, 160))

        state_dict = {
            'deconv_layers.0.weight': torch.zeros([160, 1, 4, 4]),
            'deconv_layers.1.weight': torch.zeros([160]),
            'deconv_layers.1.bias': torch.zeros([160]),
            'deconv_layers.1.running_mean': torch.zeros([160]),
            'deconv_layers.1.running_var': torch.zeros([160]),
            'deconv_layers.1.num_batches_tracked': torch.zeros([]),
            'deconv_layers.3.weight': torch.zeros([160, 1, 4, 4]),
            'deconv_layers.4.weight': torch.zeros([160]),
            'deconv_layers.4.bias': torch.zeros([160]),
            'deconv_layers.4.running_mean': torch.zeros([160]),
            'deconv_layers.4.running_var': torch.zeros([160]),
            'deconv_layers.4.num_batches_tracked': torch.zeros([]),
            'deconv_layers.6.weight': torch.zeros([160, 1, 4, 4]),
            'deconv_layers.7.weight': torch.zeros([160]),
            'deconv_layers.7.bias': torch.zeros([160]),
            'deconv_layers.7.running_mean': torch.zeros([160]),
            'deconv_layers.7.running_var': torch.zeros([160]),
            'deconv_layers.7.num_batches_tracked': torch.zeros([]),
            'final_layer.weight': torch.zeros([17, 160, 1, 1]),
            'final_layer.bias': torch.zeros([17])
        }
        head.load_state_dict(state_dict)

        # Typical setting for Resnet
        head = ViPNASHead(in_channels=608, out_channels=17)

        state_dict = {
            'deconv_layers.0.weight': torch.zeros([608, 9, 4, 4]),
            'deconv_layers.1.weight': torch.zeros([144]),
            'deconv_layers.1.bias': torch.zeros([144]),
            'deconv_layers.1.running_mean': torch.zeros([144]),
            'deconv_layers.1.running_var': torch.zeros([144]),
            'deconv_layers.1.num_batches_tracked': torch.zeros([]),
            'deconv_layers.3.weight': torch.zeros([144, 9, 4, 4]),
            'deconv_layers.4.weight': torch.zeros([144]),
            'deconv_layers.4.bias': torch.zeros([144]),
            'deconv_layers.4.running_mean': torch.zeros([144]),
            'deconv_layers.4.running_var': torch.zeros([144]),
            'deconv_layers.4.num_batches_tracked': torch.zeros([]),
            'deconv_layers.6.weight': torch.zeros([144, 9, 4, 4]),
            'deconv_layers.7.weight': torch.zeros([144]),
            'deconv_layers.7.bias': torch.zeros([144]),
            'deconv_layers.7.running_mean': torch.zeros([144]),
            'deconv_layers.7.running_var': torch.zeros([144]),
            'deconv_layers.7.num_batches_tracked': torch.zeros([]),
            'final_layer.weight': torch.zeros([17, 144, 1, 1]),
            'final_layer.bias': torch.zeros([17])
        }
        head.load_state_dict(state_dict)
