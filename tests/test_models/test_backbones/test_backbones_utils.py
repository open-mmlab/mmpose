# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones.utils import (InvertedResidual, SELayer,
                                           channel_shuffle, make_divisible)


class TestBackboneUtils(TestCase):

    @staticmethod
    def is_norm(modules):
        """Check if is one of the norms."""
        if isinstance(modules, (GroupNorm, _BatchNorm)):
            return True
        return False

    def test_make_divisible(self):
        # test min_value is None
        result = make_divisible(34, 8, None)
        self.assertEqual(result, 32)

        # test when new_value > min_ratio * value
        result = make_divisible(10, 8, min_ratio=0.9)
        self.assertEqual(result, 16)

        # test min_value = 0.8
        result = make_divisible(33, 8, min_ratio=0.8)
        self.assertEqual(result, 32)

    def test_channel_shuffle(self):
        x = torch.randn(1, 24, 56, 56)
        with self.assertRaisesRegex(
                AssertionError, 'num_channels should be divisible by groups'):
            channel_shuffle(x, 7)

        groups = 3
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        out = channel_shuffle(x, groups)
        # test the output value when groups = 3
        for b in range(batch_size):
            for c in range(num_channels):
                c_out = c % channels_per_group * groups + \
                        c // channels_per_group
                for i in range(height):
                    for j in range(width):
                        self.assertEqual(x[b, c, i, j], out[b, c_out, i, j])

    def test_inverted_residual(self):

        with self.assertRaises(AssertionError):
            # stride must be in [1, 2]
            InvertedResidual(16, 16, 32, stride=3)

        with self.assertRaises(AssertionError):
            # se_cfg must be None or dict
            InvertedResidual(16, 16, 32, se_cfg=list())

        with self.assertRaises(AssertionError):
            # in_channeld and out_channels must be the same if
            # with_expand_conv is False
            InvertedResidual(16, 16, 32, with_expand_conv=False)

        # Test InvertedResidual forward, stride=1
        block = InvertedResidual(16, 16, 32, stride=1)
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        self.assertIsNone(getattr(block, 'se', None))
        self.assertTrue(block.with_res_shortcut)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))

        # Test InvertedResidual forward, stride=2
        block = InvertedResidual(16, 16, 32, stride=2)
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        self.assertFalse(block.with_res_shortcut)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 28, 28)))

        # Test InvertedResidual forward with se layer
        se_cfg = dict(channels=32)
        block = InvertedResidual(16, 16, 32, stride=1, se_cfg=se_cfg)
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        self.assertIsInstance(block.se, SELayer)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))

        # Test InvertedResidual forward, with_expand_conv=False
        block = InvertedResidual(32, 16, 32, with_expand_conv=False)
        x = torch.randn(1, 32, 56, 56)
        x_out = block(x)
        self.assertIsNone(getattr(block, 'expand_conv', None))
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))

        # Test InvertedResidual forward with GroupNorm
        block = InvertedResidual(
            16, 16, 32, norm_cfg=dict(type='GN', num_groups=2))
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        for m in block.modules():
            if self.is_norm(m):
                self.assertIsInstance(m, GroupNorm)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))

        # Test InvertedResidual forward with HSigmoid
        block = InvertedResidual(16, 16, 32, act_cfg=dict(type='HSigmoid'))
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))

        # Test InvertedResidual forward with checkpoint
        block = InvertedResidual(16, 16, 32, with_cp=True)
        x = torch.randn(1, 16, 56, 56)
        x_out = block(x)
        self.assertTrue(block.with_cp)
        self.assertEqual(x_out.shape, torch.Size((1, 16, 56, 56)))
