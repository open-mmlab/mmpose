# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.utils import is_tuple_of

from mmpose.models.heads import DEKRHead
from mmpose.testing import get_coco_sample, get_packed_inputs
from mmpose.utils.tensor_utils import to_tensor


class TestDEKRHead(TestCase):

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

        head = DEKRHead(in_channels=32, num_keypoints=17)

        feats = [torch.rand(1, 32, 128, 128)]
        output = head.forward(feats)  # should be (heatmaps, displacements)
        self.assertTrue(is_tuple_of(output, torch.Tensor))
        self.assertEqual(output[0].shape, (1, 18, 128, 128))
        self.assertEqual(output[1].shape, (1, 34, 128, 128))

    def test_predict(self):

        codec_cfg = dict(
            type='SPR',
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, 2),
            generate_keypoint_heatmaps=True,
        )

        head = DEKRHead(in_channels=32, num_keypoints=17, decoder=codec_cfg)

        feats = self._get_feats()
        data_samples = self._get_data_samples()
        with torch.no_grad():
            preds = head.predict(feats, data_samples)
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0].keypoints.shape[1:], (17, 2))
            self.assertEqual(preds[0].keypoint_scores.shape[1:], (17, ))

        # predict with rescore net
        head = DEKRHead(
            in_channels=32,
            num_keypoints=17,
            decoder=codec_cfg,
            rescore_cfg=dict(in_channels=74, norm_indexes=(5, 6)))

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
            self.assertEqual(pred_fields[0].heatmaps.shape, (18, 128, 128))
            self.assertEqual(pred_fields[0].displacements.shape,
                             (34, 128, 128))

    def test_loss(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)

        codec_cfg = dict(
            type='SPR',
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, 2),
            generate_keypoint_heatmaps=True,
        )

        head = DEKRHead(
            in_channels=32,
            num_keypoints=17,
            decoder=codec_cfg,
            heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
            displacement_loss=dict(
                type='SoftWeightSmoothL1Loss',
                use_target_weight=True,
                supervise_empty=False,
                beta=1 / 9,
            ))

        encoded = head.decoder.encode(data['keypoints'],
                                      data['keypoints_visible'])
        feats = self._get_feats()
        data_samples = self._get_data_samples()
        for data_sample in data_samples:
            data_sample.gt_fields.set_data(
                {k: to_tensor(v)
                 for k, v in encoded.items()})

        losses = head.loss(feats, data_samples)
        self.assertIn('loss/heatmap', losses)
        self.assertEqual(losses['loss/heatmap'].ndim, 0)
        self.assertIn('loss/displacement', losses)
        self.assertEqual(losses['loss/displacement'].ndim, 0)
