# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import CPM
from mmpose.models.backbones.cpm import CpmBlock


class TestCPM(TestCase):

    def test_cpm_block(self):
        with self.assertRaises(AssertionError):
            # len(channels) == len(kernels)
            CpmBlock(
                3, channels=[3, 3, 3], kernels=[
                    1,
                ])

        # Test CPM Block
        model = CpmBlock(3, channels=[3, 3, 3], kernels=[1, 1, 1])
        model.train()

        imgs = torch.randn(1, 3, 10, 10)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 3, 10, 10]))

    def test_cpm_backbone(self):
        with self.assertRaises(AssertionError):
            # CPM's num_stacks should larger than 0
            CPM(in_channels=3, out_channels=17, num_stages=-1)

        with self.assertRaises(AssertionError):
            # CPM's in_channels should be 3
            CPM(in_channels=2, out_channels=17)

        # Test CPM
        model = CPM(in_channels=3, out_channels=17, num_stages=1)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 256, 192)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([1, 17, 32, 24]))

        imgs = torch.randn(1, 3, 384, 288)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([1, 17, 48, 36]))

        imgs = torch.randn(1, 3, 368, 368)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([1, 17, 46, 46]))

        # Test CPM multi-stages
        model = CPM(in_channels=3, out_channels=17, num_stages=2)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 368, 368)
        feat = model(imgs)
        self.assertEqual(len(feat), 2)
        self.assertEqual(feat[0].shape, torch.Size([1, 17, 46, 46]))
        self.assertEqual(feat[1].shape, torch.Size([1, 17, 46, 46]))
