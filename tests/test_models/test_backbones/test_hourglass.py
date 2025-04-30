# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import HourglassAENet, HourglassNet


class TestHourglass(TestCase):

    def test_hourglass_backbone(self):
        with self.assertRaises(AssertionError):
            # HourglassNet's num_stacks should larger than 0
            HourglassNet(num_stacks=0)

        with self.assertRaises(AssertionError):
            # len(stage_channels) should equal len(stage_blocks)
            HourglassNet(
                stage_channels=[256, 256, 384, 384, 384],
                stage_blocks=[2, 2, 2, 2, 2, 4])

        with self.assertRaises(AssertionError):
            # len(stage_channels) should larger than downsample_times
            HourglassNet(
                downsample_times=5,
                stage_channels=[256, 256, 384, 384, 384],
                stage_blocks=[2, 2, 2, 2, 2])

        # Test HourglassNet-52
        model = HourglassNet(num_stacks=1)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 256, 256)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([1, 256, 64, 64]))

        # Test HourglassNet-104
        model = HourglassNet(num_stacks=2)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 256, 256)
        feat = model(imgs)
        self.assertEqual(len(feat), 2)
        self.assertEqual(feat[0].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(feat[1].shape, torch.Size([1, 256, 64, 64]))

    def test_hourglass_ae_backbone(self):
        with self.assertRaises(AssertionError):
            # HourglassAENet's num_stacks should larger than 0
            HourglassAENet(num_stacks=0)

        with self.assertRaises(AssertionError):
            # len(stage_channels) should larger than downsample_times
            HourglassAENet(
                downsample_times=5, stage_channels=[256, 256, 384, 384, 384])

        # num_stack=1
        model = HourglassAENet(num_stacks=1)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 256, 256)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([1, 34, 64, 64]))

        # num_stack=2
        model = HourglassAENet(num_stacks=2)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 256, 256)
        feat = model(imgs)
        self.assertEqual(len(feat), 2)
        self.assertEqual(feat[0].shape, torch.Size([1, 34, 64, 64]))
        self.assertEqual(feat[1].shape, torch.Size([1, 34, 64, 64]))
