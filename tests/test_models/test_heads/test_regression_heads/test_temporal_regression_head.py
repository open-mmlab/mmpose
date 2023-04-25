# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch

from mmpose.models.heads import TemporalRegressionHead


class TestTemporalRegressionHead(TestCase):

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

        head = TemporalRegressionHead(in_channels=1024, num_joints=17)
        self.assertEqual(head.conv.weight.shape, (17 * 3, 1024, 1))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = TemporalRegressionHead(
            in_channels=1024,
            num_joints=17,
            decoder=dict(type='VideoPoseLifting', num_keypoints=17),
        )
        self.assertIsNotNone(head.decoder)


if __name__ == '__main__':
    unittest.main()
