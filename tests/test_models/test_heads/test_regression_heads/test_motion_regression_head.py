# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmpose.models.heads import MotionRegressionHead
from mmpose.testing import get_packed_inputs


class TestMotionRegressionHead(TestCase):

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

        head = MotionRegressionHead(in_channels=1024)
        self.assertEqual(head.fc.weight.shape, (3, 512))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = MotionRegressionHead(
            in_channels=1024,
            decoder=dict(type='VideoPoseLifting', num_keypoints=17),
        )
        self.assertIsNotNone(head.decoder)

    def test_predict(self):
        decoder_cfg = dict(type='VideoPoseLifting', num_keypoints=17)

        head = MotionRegressionHead(
            in_channels=1024,
            decoder=decoder_cfg,
        )

        feats = self._get_feats(batch_size=4, feat_shapes=[(2, 17, 1024)])
        batch_data_samples = get_packed_inputs(
            batch_size=2, with_heatmap=False)['data_samples']
        preds = head.predict(feats, batch_data_samples)

        self.assertTrue(len(preds), 2)
        self.assertIsInstance(preds[0], InstanceData)
        self.assertEqual(preds[0].keypoints.shape, (1, 2, 17, 3))
