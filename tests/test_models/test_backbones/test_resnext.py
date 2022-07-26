# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import ResNeXt
from mmpose.models.backbones.resnext import Bottleneck as BottleneckX


class TestResnext(TestCase):

    def test_bottleneck(self):
        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            BottleneckX(
                64, 64, groups=32, width_per_group=4, style='tensorflow')

        # Test ResNeXt Bottleneck structure
        block = BottleneckX(
            64, 256, groups=32, width_per_group=4, stride=2, style='pytorch')
        self.assertEqual(block.conv2.stride, (2, 2))
        self.assertEqual(block.conv2.groups, 32)
        self.assertEqual(block.conv2.out_channels, 128)

        # Test ResNeXt Bottleneck forward
        block = BottleneckX(
            64, 64, base_channels=16, groups=32, width_per_group=4)
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_resnext(self):
        with self.assertRaises(KeyError):
            # ResNeXt depth should be in [50, 101, 152]
            ResNeXt(depth=18)

        # Test ResNeXt with group 32, width_per_group 4
        model = ResNeXt(
            depth=50, groups=32, width_per_group=4, out_indices=(0, 1, 2, 3))
        for m in model.modules():
            if isinstance(m, BottleneckX):
                self.assertEqual(m.conv2.groups, 32)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([1, 256, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([1, 512, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([1, 1024, 14, 14]))
        self.assertEqual(feat[3].shape, torch.Size([1, 2048, 7, 7]))

        # Test ResNeXt with layers 3 out forward
        model = ResNeXt(
            depth=50, groups=32, width_per_group=4, out_indices=(3, ))
        for m in model.modules():
            if isinstance(m, BottleneckX):
                self.assertEqual(m.conv2.groups, 32)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[-1].shape, torch.Size([1, 2048, 7, 7]))
