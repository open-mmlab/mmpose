# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import List, Tuple
from unittest import TestCase

import torch

from mmpose.models.heads import TrajectoryRegressionHead


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

        head = TrajectoryRegressionHead(in_channels=1024, num_joints=1)
        self.assertEqual(head.conv.weight.shape, (17 * 3, 1024, 1))
        self.assertIsNone(head.decoder)

        # w/ decoder
        head = TrajectoryRegressionHead(
            in_channels=1024,
            num_joints=1,
            decoder=dict(type='ImagePoseLifting', num_keypoints=1),
        )
        self.assertIsNotNone(head.decoder)


if __name__ == '__main__':
    unittest.main()
