# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import numpy as np
import torch

from mmpose.models.heads import CIDHead
from mmpose.testing import get_coco_sample, get_packed_inputs
from mmpose.utils.tensor_utils import to_tensor


class TestCIDHead(TestCase):

    def _get_feats(
        self,
        batch_size: int = 1,
        feat_shapes: List[Tuple[int, int, int]] = [(32, 128, 128)],
    ):

        feats = [
            torch.rand((batch_size, ) + shape, dtype=torch.float32)
            for shape in feat_shapes
        ]

        if len(feats) > 1:
            feats = [[x] for x in feats]

        return feats

    def _get_data_samples(self):
        data_samples = get_packed_inputs(
            1,
            input_size=(512, 512),
            heatmap_size=(128, 128),
            img_shape=(512, 512))['data_samples']
        return data_samples

    def test_forward(self):

        head = CIDHead(in_channels=32, num_keypoints=17, gfd_channels=32)

        feats = [torch.rand(1, 32, 128, 128)]
        with torch.no_grad():
            output = head.forward(feats)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape[1:], (17, 128, 128))

    def test_predict(self):

        codec = dict(
            type='DecoupledHeatmap',
            input_size=(512, 512),
            heatmap_size=(128, 128))

        head = CIDHead(
            in_channels=32, num_keypoints=17, gfd_channels=32, decoder=codec)

        feats = self._get_feats()
        data_samples = self._get_data_samples()
        with torch.no_grad():
            preds = head.predict(feats, data_samples)
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].keypoints.shape[1:], (17, 2))
            self.assertEqual(preds[0].keypoint_scores.shape[1:], (17, ))

        # tta
        with torch.no_grad():
            feats_flip = self._get_feats(feat_shapes=[(32, 128,
                                                       128), (32, 128, 128)])
            preds = head.predict(feats_flip, data_samples,
                                 dict(flip_test=True))
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].keypoints.shape[1:], (17, 2))
            self.assertEqual(preds[0].keypoint_scores.shape[1:], (17, ))

        # output heatmaps
        with torch.no_grad():
            _, pred_fields = head.predict(feats, data_samples,
                                          dict(output_heatmaps=True))
            self.assertEqual(len(pred_fields), 1)
            self.assertEqual(pred_fields[0].heatmaps.shape[1:], (128, 128))
            self.assertEqual(pred_fields[0].heatmaps.shape[0] % 17, 0)

    def test_loss(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)
        data['bbox'] = np.tile(data['bbox'], 2).reshape(-1, 4, 2)
        data['bbox'][:, 1:3, 0] = data['bbox'][:, 0:2, 0]

        codec_cfg = dict(
            type='DecoupledHeatmap',
            input_size=(512, 512),
            heatmap_size=(128, 128))

        head = CIDHead(
            in_channels=32,
            num_keypoints=17,
            gfd_channels=32,
            decoder=codec_cfg,
            coupled_heatmap_loss=dict(
                type='FocalHeatmapLoss', loss_weight=1.0),
            decoupled_heatmap_loss=dict(
                type='FocalHeatmapLoss', loss_weight=4.0),
            contrastive_loss=dict(type='InfoNCELoss', loss_weight=1.0))

        encoded = head.decoder.encode(data['keypoints'],
                                      data['keypoints_visible'], data['bbox'])
        feats = self._get_feats()
        data_samples = self._get_data_samples()
        for data_sample in data_samples:
            data_sample.gt_fields.set_data({
                'heatmaps':
                to_tensor(encoded['heatmaps']),
                'instance_heatmaps':
                to_tensor(encoded['instance_heatmaps'])
            })
            data_sample.gt_instance_labels.set_data(
                {'instance_coords': to_tensor(encoded['instance_coords'])})
            data_sample.gt_instance_labels.set_data(
                {'keypoint_weights': to_tensor(encoded['keypoint_weights'])})

        losses = head.loss(feats, data_samples)
        self.assertIn('loss/heatmap_coupled', losses)
        self.assertEqual(losses['loss/heatmap_coupled'].ndim, 0)
        self.assertIn('loss/heatmap_decoupled', losses)
        self.assertEqual(losses['loss/heatmap_decoupled'].ndim, 0)
        self.assertIn('loss/contrastive', losses)
        self.assertEqual(losses['loss/contrastive'].ndim, 0)
