# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from unittest import TestCase

import torch

from mmpose.models.necks import FeatureMapProcessor


class TestFeatureMapProcessor(TestCase):

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

        neck = FeatureMapProcessor(select_index=0)
        self.assertSequenceEqual(neck.select_index, (0, ))

        with self.assertRaises(AssertionError):
            neck = FeatureMapProcessor(scale_factor=0.0)

    def test_call(self):

        inputs = self._get_feats(
            batch_size=2, feat_shapes=[(2, 16, 16), (4, 8, 8), (8, 4, 4)])

        neck = FeatureMapProcessor(select_index=0)
        output = neck(inputs)
        self.assertEqual(len(output), 1)
        self.assertSequenceEqual(output[0].shape, (2, 2, 16, 16))

        neck = FeatureMapProcessor(select_index=(2, 1))
        output = neck(inputs)
        self.assertEqual(len(output), 2)
        self.assertSequenceEqual(output[1].shape, (2, 4, 8, 8))
        self.assertSequenceEqual(output[0].shape, (2, 8, 4, 4))

        neck = FeatureMapProcessor(select_index=(1, 2), concat=True)
        output = neck(inputs)
        self.assertEqual(len(output), 1)
        self.assertSequenceEqual(output[0].shape, (2, 12, 8, 8))

        neck = FeatureMapProcessor(
            select_index=(2, 1), concat=True, scale_factor=2)
        output = neck(inputs)
        self.assertEqual(len(output), 1)
        self.assertSequenceEqual(output[0].shape, (2, 12, 8, 8))

        neck = FeatureMapProcessor(concat=True, apply_relu=True)
        output = neck(inputs)
        self.assertEqual(len(output), 1)
        self.assertSequenceEqual(output[0].shape, (2, 14, 16, 16))
        self.assertGreaterEqual(output[0].max(), 0)
