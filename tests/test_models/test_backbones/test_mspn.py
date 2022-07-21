# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import MSPN


class TestMSPN(TestCase):

    def test_mspn_backbone(self):
        with self.assertRaises(AssertionError):
            # MSPN's num_stages should larger than 0
            MSPN(num_stages=0)
        with self.assertRaises(AssertionError):
            # MSPN's num_units should larger than 1
            MSPN(num_units=1)
        with self.assertRaises(AssertionError):
            # len(num_blocks) should equal num_units
            MSPN(num_units=2, num_blocks=[2, 2, 2])

        # Test MSPN's outputs
        model = MSPN(num_stages=2, num_units=2, num_blocks=[2, 2])
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 511, 511)
        feat = model(imgs)
        self.assertEqual(len(feat), 2)
        self.assertEqual(len(feat[0]), 2)
        self.assertEqual(len(feat[1]), 2)
        self.assertEqual(feat[0][0].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(feat[0][1].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(feat[1][0].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(feat[1][1].shape, torch.Size([1, 256, 128, 128]))
