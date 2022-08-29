# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpose.models.backbones import ViPNAS_ResNet
from mmpose.models.backbones.vipnas_resnet import (ViPNAS_Bottleneck,
                                                   ViPNAS_ResLayer,
                                                   get_expansion)


class TestVipnasResnet(TestCase):

    @staticmethod
    def is_block(modules):
        """Check if is ViPNAS_ResNet building block."""
        if isinstance(modules, (ViPNAS_Bottleneck)):
            return True
        return False

    @staticmethod
    def all_zeros(modules):
        """Check if the weight(and bias) is all zero."""
        weight_zero = torch.equal(modules.weight.data,
                                  torch.zeros_like(modules.weight.data))
        if hasattr(modules, 'bias'):
            bias_zero = torch.equal(modules.bias.data,
                                    torch.zeros_like(modules.bias.data))
        else:
            bias_zero = True

        return weight_zero and bias_zero

    @staticmethod
    def check_norm_state(modules, train_state):
        """Check if norm layer is in correct train state."""
        for mod in modules:
            if isinstance(mod, _BatchNorm):
                if mod.training != train_state:
                    return False
        return True

    def test_get_expansion(self):
        self.assertEqual(get_expansion(ViPNAS_Bottleneck, 2), 2)
        self.assertEqual(get_expansion(ViPNAS_Bottleneck), 1)

        class MyResBlock(nn.Module):

            expansion = 8

        self.assertEqual(get_expansion(MyResBlock), 8)

        # expansion must be an integer or None
        with self.assertRaises(TypeError):
            get_expansion(ViPNAS_Bottleneck, '0')

        # expansion is not specified and cannot be inferred
        with self.assertRaises(TypeError):

            class SomeModule(nn.Module):
                pass

            get_expansion(SomeModule)

    def test_vipnas_bottleneck(self):
        # style must be in ['pytorch', 'caffe']
        with self.assertRaises(AssertionError):
            ViPNAS_Bottleneck(64, 64, style='tensorflow')

        # expansion must be divisible by out_channels
        with self.assertRaises(AssertionError):
            ViPNAS_Bottleneck(64, 64, expansion=3)

        # Test ViPNAS_Bottleneck style
        block = ViPNAS_Bottleneck(64, 64, stride=2, style='pytorch')
        self.assertEqual(block.conv1.stride, (1, 1))
        self.assertEqual(block.conv2.stride, (2, 2))
        block = ViPNAS_Bottleneck(64, 64, stride=2, style='caffe')
        self.assertEqual(block.conv1.stride, (2, 2))
        self.assertEqual(block.conv2.stride, (1, 1))

        # ViPNAS_Bottleneck with stride 1
        block = ViPNAS_Bottleneck(64, 64, style='pytorch')
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 16)
        self.assertEqual(block.out_channels, 64)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 16)
        self.assertEqual(block.conv1.kernel_size, (1, 1))
        self.assertEqual(block.conv2.in_channels, 16)
        self.assertEqual(block.conv2.out_channels, 16)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        self.assertEqual(block.conv3.in_channels, 16)
        self.assertEqual(block.conv3.out_channels, 64)
        self.assertEqual(block.conv3.kernel_size, (1, 1))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, (1, 64, 56, 56))

        # ViPNAS_Bottleneck with stride 1 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128))
        block = ViPNAS_Bottleneck(
            64, 128, style='pytorch', downsample=downsample)
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 32)
        self.assertEqual(block.out_channels, 128)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 32)
        self.assertEqual(block.conv1.kernel_size, (1, 1))
        self.assertEqual(block.conv2.in_channels, 32)
        self.assertEqual(block.conv2.out_channels, 32)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        self.assertEqual(block.conv3.in_channels, 32)
        self.assertEqual(block.conv3.out_channels, 128)
        self.assertEqual(block.conv3.kernel_size, (1, 1))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, (1, 128, 56, 56))

        # ViPNAS_Bottleneck with stride 2 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
        block = ViPNAS_Bottleneck(
            64, 128, stride=2, style='pytorch', downsample=downsample)
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, (1, 128, 28, 28))

        # ViPNAS_Bottleneck with expansion 2
        block = ViPNAS_Bottleneck(64, 64, style='pytorch', expansion=2)
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 32)
        self.assertEqual(block.out_channels, 64)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 32)
        self.assertEqual(block.conv1.kernel_size, (1, 1))
        self.assertEqual(block.conv2.in_channels, 32)
        self.assertEqual(block.conv2.out_channels, 32)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        self.assertEqual(block.conv3.in_channels, 32)
        self.assertEqual(block.conv3.out_channels, 64)
        self.assertEqual(block.conv3.kernel_size, (1, 1))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, (1, 64, 56, 56))

        # Test ViPNAS_Bottleneck with checkpointing
        block = ViPNAS_Bottleneck(64, 64, with_cp=True)
        block.train()
        self.assertTrue(block.with_cp)
        x = torch.randn(1, 64, 56, 56, requires_grad=True)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_vipnas_bottleneck_reslayer(self):
        # 3 Bottleneck w/o downsample
        layer = ViPNAS_ResLayer(ViPNAS_Bottleneck, 3, 32, 32)
        self.assertEqual(len(layer), 3)
        for i in range(3):
            self.assertEqual(layer[i].in_channels, 32)
            self.assertEqual(layer[i].out_channels, 32)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 32, 56, 56))

        # 3 ViPNAS_Bottleneck w/ stride 1 and downsample
        layer = ViPNAS_ResLayer(ViPNAS_Bottleneck, 3, 32, 64)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 1)
        self.assertEqual(layer[0].conv1.out_channels, 64)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (1, 1))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 64)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 56, 56))

        # 3 ViPNAS_Bottleneck w/ stride 2 and downsample
        layer = ViPNAS_ResLayer(ViPNAS_Bottleneck, 3, 32, 64, stride=2)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(layer[0].conv1.out_channels, 64)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (2, 2))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 64)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

        # 3 ViPNAS_Bottleneck w/ stride 2 and downsample with avg pool
        layer = ViPNAS_ResLayer(
            ViPNAS_Bottleneck, 3, 32, 64, stride=2, avg_down=True)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(layer[0].conv1.out_channels, 64)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 3)
        self.assertIsInstance(layer[0].downsample[0], nn.AvgPool2d)
        self.assertEqual(layer[0].downsample[0].stride, 2)
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 64)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

        # 3 ViPNAS_Bottleneck with custom expansion
        layer = ViPNAS_ResLayer(ViPNAS_Bottleneck, 3, 32, 32, expansion=2)
        self.assertEqual(len(layer), 3)
        for i in range(3):
            self.assertEqual(layer[i].in_channels, 32)
            self.assertEqual(layer[i].out_channels, 32)
            self.assertEqual(layer[i].stride, 1)
            self.assertEqual(layer[i].conv1.out_channels, 16)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 32, 56, 56))

    def test_resnet(self):
        """Test ViPNAS_ResNet backbone."""
        with self.assertRaises(KeyError):
            # ViPNAS_ResNet depth should be in [50]
            ViPNAS_ResNet(20)

        with self.assertRaises(AssertionError):
            # In ViPNAS_ResNet: 1 <= num_stages <= 4
            ViPNAS_ResNet(50, num_stages=0)

        with self.assertRaises(AssertionError):
            # In ViPNAS_ResNet: 1 <= num_stages <= 4
            ViPNAS_ResNet(50, num_stages=5)

        with self.assertRaises(AssertionError):
            # len(strides) == len(dilations) == num_stages
            ViPNAS_ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

        with self.assertRaises(TypeError):
            # init_weights must have no parameter
            model = ViPNAS_ResNet(50)
            model.init_weights(pretrained=0)

        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            ViPNAS_ResNet(50, style='tensorflow')

        # Test ViPNAS_ResNet50 norm_eval=True
        model = ViPNAS_ResNet(50, norm_eval=True)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test ViPNAS_ResNet50 with first stage frozen
        frozen_stages = 1
        model = ViPNAS_ResNet(50, frozen_stages=frozen_stages)
        model.init_weights()
        model.train()
        self.assertFalse(model.norm1.training)
        for layer in [model.conv1, model.norm1]:
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f'layer{i}')
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    self.assertFalse(mod.training)
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

        # Test ViPNAS_ResNet50 with BatchNorm forward
        model = ViPNAS_ResNet(50, out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 80, 56, 56))
        self.assertEqual(feat[1].shape, (1, 160, 28, 28))
        self.assertEqual(feat[2].shape, (1, 304, 14, 14))
        self.assertEqual(feat[3].shape, (1, 608, 7, 7))

        # Test ViPNAS_ResNet50 with layers 1, 2, 3 out forward
        model = ViPNAS_ResNet(50, out_indices=(0, 1, 2))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 3)
        self.assertEqual(feat[0].shape, (1, 80, 56, 56))
        self.assertEqual(feat[1].shape, (1, 160, 28, 28))
        self.assertEqual(feat[2].shape, (1, 304, 14, 14))

        # Test ViPNAS_ResNet50 with layers 3 (top feature maps) out forward
        model = ViPNAS_ResNet(50, out_indices=(3, ))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, (1, 608, 7, 7))

        # Test ViPNAS_ResNet50 with checkpoint forward
        model = ViPNAS_ResNet(50, out_indices=(0, 1, 2, 3), with_cp=True)
        for m in model.modules():
            if self.is_block(m):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 80, 56, 56))
        self.assertEqual(feat[1].shape, (1, 160, 28, 28))
        self.assertEqual(feat[2].shape, (1, 304, 14, 14))
        self.assertEqual(feat[3].shape, (1, 608, 7, 7))
