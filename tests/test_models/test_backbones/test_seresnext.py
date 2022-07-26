# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import SEResNeXt
from mmpose.models.backbones.seresnext import SEBottleneck as SEBottleneckX


class TestSEResnext(TestCase):

    def test_bottleneck(self):
        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            SEBottleneckX(
                64, 64, groups=32, width_per_group=4, style='tensorflow')

        # Test SEResNeXt Bottleneck structure
        block = SEBottleneckX(
            64, 256, groups=32, width_per_group=4, stride=2, style='pytorch')
        self.assertEqual(block.width_per_group, 4)
        self.assertEqual(block.conv2.stride, (2, 2))
        self.assertEqual(block.conv2.groups, 32)
        self.assertEqual(block.conv2.out_channels, 128)
        self.assertEqual(block.conv2.out_channels, block.mid_channels)

        # Test SEResNeXt Bottleneck structure (groups=1)
        block = SEBottleneckX(
            64, 256, groups=1, width_per_group=4, stride=2, style='pytorch')
        self.assertEqual(block.conv2.stride, (2, 2))
        self.assertEqual(block.conv2.groups, 1)
        self.assertEqual(block.conv2.out_channels, 64)
        self.assertEqual(block.mid_channels, 64)
        self.assertEqual(block.conv2.out_channels, block.mid_channels)

        # Test SEResNeXt Bottleneck forward
        block = SEBottleneckX(
            64, 64, base_channels=16, groups=32, width_per_group=4)
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_seresnext(self):
        with self.assertRaises(KeyError):
            # SEResNeXt depth should be in [50, 101, 152]
            SEResNeXt(depth=18)

        # Test SEResNeXt with group 32, width_per_group 4
        model = SEResNeXt(
            depth=50, groups=32, width_per_group=4, out_indices=(0, 1, 2, 3))
        for m in model.modules():
            if isinstance(m, SEBottleneckX):
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

        # Test SEResNeXt with layers 3 out forward
        model = SEResNeXt(
            depth=50, groups=32, width_per_group=4, out_indices=(3, ))
        for m in model.modules():
            if isinstance(m, SEBottleneckX):
                self.assertEqual(m.conv2.groups, 32)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 2048, 7, 7]))
