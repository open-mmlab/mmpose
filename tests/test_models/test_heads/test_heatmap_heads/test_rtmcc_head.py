# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmpose.models.heads import RTMCCHead
from mmpose.models.utils import RTMCCBlock
from mmpose.testing import get_packed_inputs


class TestRTMCCHead(TestCase):

    def _get_feats(self,
                   batch_size: int = 2,
                   feat_shapes: List[Tuple[int, int, int]] = [(32, 6, 8)]):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]
        return feats

    def test_init(self):

        if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
            return unittest.skip('RTMCCHead requires PyTorch >= 1.7')

        # original version
        head = RTMCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='gaussian',
                sigma=(4.9, 5.66),
                simcc_split_ratio=2.0,
                normalize=False))
        self.assertIsNotNone(head.decoder)
        self.assertTrue(isinstance(head.final_layer, nn.Conv2d))
        self.assertTrue(isinstance(head.mlp, nn.Sequential))
        self.assertTrue(isinstance(head.gau, RTMCCBlock))
        self.assertTrue(isinstance(head.cls_x, nn.Linear))
        self.assertTrue(isinstance(head.cls_y, nn.Linear))

        # w/ 1x1 conv
        head = RTMCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=1,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='gaussian',
                sigma=(4.9, 5.66),
                simcc_split_ratio=2.0,
                normalize=False))
        self.assertIsNotNone(head.decoder)
        self.assertTrue(isinstance(head.final_layer, nn.Conv2d))
        self.assertTrue(isinstance(head.mlp, nn.Sequential))
        self.assertTrue(isinstance(head.gau, RTMCCBlock))
        self.assertTrue(isinstance(head.cls_x, nn.Linear))
        self.assertTrue(isinstance(head.cls_y, nn.Linear))

        # hidden_dims
        head = RTMCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=512,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='gaussian',
                sigma=(4.9, 5.66),
                simcc_split_ratio=2.0,
                normalize=False))
        self.assertIsNotNone(head.decoder)
        self.assertTrue(isinstance(head.final_layer, nn.Conv2d))
        self.assertTrue(isinstance(head.mlp, nn.Sequential))
        self.assertTrue(isinstance(head.gau, RTMCCBlock))
        self.assertTrue(isinstance(head.cls_x, nn.Linear))
        self.assertTrue(isinstance(head.cls_y, nn.Linear))

        # s = 256
        head = RTMCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=256,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
            decoder=dict(
                type='SimCCLabel',
                input_size=(192, 256),
                smoothing_type='gaussian',
                sigma=(4.9, 5.66),
                simcc_split_ratio=2.0,
                normalize=False))
        self.assertIsNotNone(head.decoder)
        self.assertTrue(isinstance(head.final_layer, nn.Conv2d))
        self.assertTrue(isinstance(head.mlp, nn.Sequential))
        self.assertTrue(isinstance(head.gau, RTMCCBlock))
        self.assertTrue(isinstance(head.cls_x, nn.Linear))
        self.assertTrue(isinstance(head.cls_y, nn.Linear))

    def test_predict(self):

        if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
            return unittest.skip('RTMCCHead requires PyTorch >= 1.7')

        decoder_cfg_list = []
        # original version
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False)
        decoder_cfg_list.append(decoder_cfg)

        # single sigma
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=6.,
            simcc_split_ratio=2.0,
            normalize=False)
        decoder_cfg_list.append(decoder_cfg)

        # normalize
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=6.,
            simcc_split_ratio=2.0,
            normalize=True)
        decoder_cfg_list.append(decoder_cfg)

        # dark
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=6.,
            simcc_split_ratio=2.0,
            use_dark=True)
        decoder_cfg_list.append(decoder_cfg)

        for decoder_cfg in decoder_cfg_list:
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # 1x1 conv
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=1,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            # hidden dims
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=512,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # s
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=64,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # expansion factor
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=3,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # drop path
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.1,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # act fn
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='ReLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # use_rel_bias
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=True,
                    pos_enc=False),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # pos_enc
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=True),
                decoder=decoder_cfg)
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, _ = head.predict(feats, batch_data_samples)

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)

            # output_heatmaps
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                decoder=decoder_cfg,
            )
            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2,
                simcc_split_ratio=decoder_cfg['simcc_split_ratio'],
                with_simcc_label=True)['data_samples']
            preds, pred_heatmaps = head.predict(
                feats, batch_data_samples, test_cfg=dict(output_heatmaps=True))

            self.assertTrue(len(preds), 2)
            self.assertIsInstance(preds[0], InstanceData)
            self.assertEqual(preds[0].keypoint_x_labels.shape, (1, 17, 384))
            self.assertEqual(preds[0].keypoint_y_labels.shape, (1, 17, 512))
            self.assertEqual(
                preds[0].keypoints.shape,
                batch_data_samples[0].gt_instances.keypoints.shape)
            self.assertEqual(pred_heatmaps[0].heatmaps.shape, (17, 512, 384))

    def test_tta(self):
        if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
            return unittest.skip('RTMCCHead requires PyTorch >= 1.7')

        # flip test
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False)

        head = RTMCCHead(
            in_channels=32,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
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
        if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
            return unittest.skip('RTMCCHead requires PyTorch >= 1.7')

        decoder_cfg_list = []
        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False)
        decoder_cfg_list.append(decoder_cfg)

        decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=True)
        decoder_cfg_list.append(decoder_cfg)

        # decoder
        for decoder_cfg in decoder_cfg_list:
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                loss=dict(
                    type='KLDiscretLoss',
                    use_target_weight=True,
                    beta=1.,
                    label_softmax=False,
                ),
                decoder=decoder_cfg)

            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2, simcc_split_ratio=2.0,
                with_simcc_label=True)['data_samples']
            losses = head.loss(feats, batch_data_samples)
            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)

            # beta = 10
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                loss=dict(
                    type='KLDiscretLoss',
                    use_target_weight=True,
                    beta=10.,
                    label_softmax=False,
                ),
                decoder=decoder_cfg)

            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2, simcc_split_ratio=2.0,
                with_simcc_label=True)['data_samples']
            losses = head.loss(feats, batch_data_samples)
            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)

            # label softmax
            head = RTMCCHead(
                in_channels=32,
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                loss=dict(
                    type='KLDiscretLoss',
                    use_target_weight=True,
                    beta=10.,
                    label_softmax=True,
                ),
                decoder=decoder_cfg)

            feats = self._get_feats(batch_size=2, feat_shapes=[(32, 8, 6)])
            batch_data_samples = get_packed_inputs(
                batch_size=2, simcc_split_ratio=2.0,
                with_simcc_label=True)['data_samples']
            losses = head.loss(feats, batch_data_samples)
            self.assertIsInstance(losses['loss_kpt'], torch.Tensor)
            self.assertEqual(losses['loss_kpt'].shape, torch.Size(()))
            self.assertIsInstance(losses['acc_pose'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
