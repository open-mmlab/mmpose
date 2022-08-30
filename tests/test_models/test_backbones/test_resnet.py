# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpose.models.backbones import ResNet, ResNetV1d
from mmpose.models.backbones.resnet import (BasicBlock, Bottleneck, ResLayer,
                                            get_expansion)


class TestResnet(TestCase):

    @staticmethod
    def is_block(modules):
        """Check if is ResNet building block."""
        if isinstance(modules, (BasicBlock, Bottleneck)):
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
        self.assertEqual(get_expansion(Bottleneck, 2), 2)
        self.assertEqual(get_expansion(BasicBlock), 1)
        self.assertEqual(get_expansion(Bottleneck), 4)

        class MyResBlock(nn.Module):

            expansion = 8

        self.assertEqual(get_expansion(MyResBlock), 8)

        # expansion must be an integer or None
        with self.assertRaises(TypeError):
            get_expansion(Bottleneck, '0')

        # expansion is not specified and cannot be inferred
        with self.assertRaises(TypeError):

            class SomeModule(nn.Module):
                pass

            get_expansion(SomeModule)

    def test_basic_block(self):
        # expansion must be 1
        with self.assertRaises(AssertionError):
            BasicBlock(64, 64, expansion=2)

        # BasicBlock with stride 1, out_channels == in_channels
        block = BasicBlock(64, 64)
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 64)
        self.assertEqual(block.out_channels, 64)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 64)
        self.assertEqual(block.conv1.kernel_size, (3, 3))
        self.assertEqual(block.conv1.stride, (1, 1))
        self.assertEqual(block.conv2.in_channels, 64)
        self.assertEqual(block.conv2.out_channels, 64)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

        # BasicBlock with stride 1 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128))
        block = BasicBlock(64, 128, downsample=downsample)
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 128)
        self.assertEqual(block.out_channels, 128)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 128)
        self.assertEqual(block.conv1.kernel_size, (3, 3))
        self.assertEqual(block.conv1.stride, (1, 1))
        self.assertEqual(block.conv2.in_channels, 128)
        self.assertEqual(block.conv2.out_channels, 128)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 128, 56, 56]))

        # BasicBlock with stride 2 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128))
        block = BasicBlock(64, 128, stride=2, downsample=downsample)
        self.assertEqual(block.in_channels, 64)
        self.assertEqual(block.mid_channels, 128)
        self.assertEqual(block.out_channels, 128)
        self.assertEqual(block.conv1.in_channels, 64)
        self.assertEqual(block.conv1.out_channels, 128)
        self.assertEqual(block.conv1.kernel_size, (3, 3))
        self.assertEqual(block.conv1.stride, (2, 2))
        self.assertEqual(block.conv2.in_channels, 128)
        self.assertEqual(block.conv2.out_channels, 128)
        self.assertEqual(block.conv2.kernel_size, (3, 3))
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 128, 28, 28]))

        # forward with checkpointing
        block = BasicBlock(64, 64, with_cp=True)
        self.assertTrue(block.with_cp)
        x = torch.randn(1, 64, 56, 56, requires_grad=True)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_bottleneck(self):
        # style must be in ['pytorch', 'caffe']
        with self.assertRaises(AssertionError):
            Bottleneck(64, 64, style='tensorflow')

        # expansion must be divisible by out_channels
        with self.assertRaises(AssertionError):
            Bottleneck(64, 64, expansion=3)

        # Test Bottleneck style
        block = Bottleneck(64, 64, stride=2, style='pytorch')
        self.assertEqual(block.conv1.stride, (1, 1))
        self.assertEqual(block.conv2.stride, (2, 2))
        block = Bottleneck(64, 64, stride=2, style='caffe')
        self.assertEqual(block.conv1.stride, (2, 2))
        self.assertEqual(block.conv2.stride, (1, 1))

        # Bottleneck with stride 1
        block = Bottleneck(64, 64, style='pytorch')
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

        # Bottleneck with stride 1 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128))
        block = Bottleneck(64, 128, style='pytorch', downsample=downsample)
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

        # Bottleneck with stride 2 and downsample
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
        block = Bottleneck(
            64, 128, stride=2, style='pytorch', downsample=downsample)
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, (1, 128, 28, 28))

        # Bottleneck with expansion 2
        block = Bottleneck(64, 64, style='pytorch', expansion=2)
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

        # Test Bottleneck with checkpointing
        block = Bottleneck(64, 64, with_cp=True)
        block.train()
        self.assertTrue(block.with_cp)
        x = torch.randn(1, 64, 56, 56, requires_grad=True)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_basicblock_reslayer(self):
        # 3 BasicBlock w/o downsample
        layer = ResLayer(BasicBlock, 3, 32, 32)
        self.assertEqual(len(layer), 3)
        for i in range(3):
            self.assertEqual(layer[i].in_channels, 32)
            self.assertEqual(layer[i].out_channels, 32)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 32, 56, 56))

        # 3 BasicBlock w/ stride 1 and downsample
        layer = ResLayer(BasicBlock, 3, 32, 64)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (1, 1))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 56, 56))

        # 3 BasicBlock w/ stride 2 and downsample
        layer = ResLayer(BasicBlock, 3, 32, 64, stride=2)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (2, 2))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

        # 3 BasicBlock w/ stride 2 and downsample with avg pool
        layer = ResLayer(BasicBlock, 3, 32, 64, stride=2, avg_down=True)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 3)
        self.assertIsInstance(layer[0].downsample[0], nn.AvgPool2d)
        self.assertEqual(layer[0].downsample[0].stride, 2)
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

    def test_bottleneck_reslayer(self):
        # 3 Bottleneck w/o downsample
        layer = ResLayer(Bottleneck, 3, 32, 32)
        self.assertEqual(len(layer), 3)
        for i in range(3):
            self.assertEqual(layer[i].in_channels, 32)
            self.assertEqual(layer[i].out_channels, 32)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 32, 56, 56))

        # 3 Bottleneck w/ stride 1 and downsample
        layer = ResLayer(Bottleneck, 3, 32, 64)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 1)
        self.assertEqual(layer[0].conv1.out_channels, 16)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (1, 1))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 16)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 56, 56))

        # 3 Bottleneck w/ stride 2 and downsample
        layer = ResLayer(Bottleneck, 3, 32, 64, stride=2)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(layer[0].conv1.out_channels, 16)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 2)
        self.assertIsInstance(layer[0].downsample[0], nn.Conv2d)
        self.assertEqual(layer[0].downsample[0].stride, (2, 2))
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 16)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

        # 3 Bottleneck w/ stride 2 and downsample with avg pool
        layer = ResLayer(Bottleneck, 3, 32, 64, stride=2, avg_down=True)
        self.assertEqual(len(layer), 3)
        self.assertEqual(layer[0].in_channels, 32)
        self.assertEqual(layer[0].out_channels, 64)
        self.assertEqual(layer[0].stride, 2)
        self.assertEqual(layer[0].conv1.out_channels, 16)
        self.assertEqual(
            layer[0].downsample is not None and len(layer[0].downsample), 3)
        self.assertIsInstance(layer[0].downsample[0], nn.AvgPool2d)
        self.assertEqual(layer[0].downsample[0].stride, 2)
        for i in range(1, 3):
            self.assertEqual(layer[i].in_channels, 64)
            self.assertEqual(layer[i].out_channels, 64)
            self.assertEqual(layer[i].conv1.out_channels, 16)
            self.assertEqual(layer[i].stride, 1)
            self.assertIsNone(layer[i].downsample)
        x = torch.randn(1, 32, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, (1, 64, 28, 28))

        # 3 Bottleneck with custom expansion
        layer = ResLayer(Bottleneck, 3, 32, 32, expansion=2)
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
        """Test resnet backbone."""
        with self.assertRaises(KeyError):
            # ResNet depth should be in [18, 34, 50, 101, 152]
            ResNet(20)

        with self.assertRaises(AssertionError):
            # In ResNet: 1 <= num_stages <= 4
            ResNet(50, num_stages=0)

        with self.assertRaises(AssertionError):
            # In ResNet: 1 <= num_stages <= 4
            ResNet(50, num_stages=5)

        with self.assertRaises(AssertionError):
            # len(strides) == len(dilations) == num_stages
            ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

        with self.assertRaises(AssertionError):
            # Style must be in ['pytorch', 'caffe']
            ResNet(50, style='tensorflow')

        # Test ResNet50 norm_eval=True
        model = ResNet(50, norm_eval=True)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test ResNet50 with torchvision pretrained weight
        init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')
        model = ResNet(depth=50, norm_eval=True, init_cfg=init_cfg)
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test ResNet50 with first stage frozen
        frozen_stages = 1
        model = ResNet(50, frozen_stages=frozen_stages)
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

        # Test ResNet18 forward
        model = ResNet(18, out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 64, 56, 56))
        self.assertEqual(feat[1].shape, (1, 128, 28, 28))
        self.assertEqual(feat[2].shape, (1, 256, 14, 14))
        self.assertEqual(feat[3].shape, (1, 512, 7, 7))

        # Test ResNet50 with BatchNorm forward
        model = ResNet(50, out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 256, 56, 56))
        self.assertEqual(feat[1].shape, (1, 512, 28, 28))
        self.assertEqual(feat[2].shape, (1, 1024, 14, 14))
        self.assertEqual(feat[3].shape, (1, 2048, 7, 7))

        # Test ResNet50 with layers 1, 2, 3 out forward
        model = ResNet(50, out_indices=(0, 1, 2))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 3)
        self.assertEqual(feat[0].shape, (1, 256, 56, 56))
        self.assertEqual(feat[1].shape, (1, 512, 28, 28))
        self.assertEqual(feat[2].shape, (1, 1024, 14, 14))

        # Test ResNet50 with layers 3 (top feature maps) out forward
        model = ResNet(50, out_indices=(3, ))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[-1].shape, (1, 2048, 7, 7))

        # Test ResNet50 with checkpoint forward
        model = ResNet(50, out_indices=(0, 1, 2, 3), with_cp=True)
        for m in model.modules():
            if self.is_block(m):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 256, 56, 56))
        self.assertEqual(feat[1].shape, (1, 512, 28, 28))
        self.assertEqual(feat[2].shape, (1, 1024, 14, 14))
        self.assertEqual(feat[3].shape, (1, 2048, 7, 7))

        # zero initialization of residual blocks
        model = ResNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=True)
        model.init_weights()
        for m in model.modules():
            if isinstance(m, Bottleneck):
                self.assertTrue(self.all_zeros(m.norm3))
            elif isinstance(m, BasicBlock):
                self.assertTrue(self.all_zeros(m.norm2))

        # non-zero initialization of residual blocks
        model = ResNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=False)
        model.init_weights()
        for m in model.modules():
            if isinstance(m, Bottleneck):
                self.assertFalse(self.all_zeros(m.norm3))
            elif isinstance(m, BasicBlock):
                self.assertFalse(self.all_zeros(m.norm2))

    def test_resnet_v1d(self):
        model = ResNetV1d(depth=50, out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        self.assertEqual(len(model.stem), 3)
        for i in range(3):
            self.assertIsInstance(model.stem[i], ConvModule)

        imgs = torch.randn(1, 3, 224, 224)
        feat = model.stem(imgs)
        self.assertEqual(feat.shape, (1, 64, 112, 112))
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 256, 56, 56))
        self.assertEqual(feat[1].shape, (1, 512, 28, 28))
        self.assertEqual(feat[2].shape, (1, 1024, 14, 14))
        self.assertEqual(feat[3].shape, (1, 2048, 7, 7))

        # Test ResNet50V1d with first stage frozen
        frozen_stages = 1
        model = ResNetV1d(depth=50, frozen_stages=frozen_stages)
        self.assertEqual(len(model.stem), 3)
        for i in range(3):
            self.assertIsInstance(model.stem[i], ConvModule)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.stem, False))
        for param in model.stem.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f'layer{i}')
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    self.assertFalse(mod.training)
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

    def test_resnet_half_channel(self):
        model = ResNet(50, base_channels=32, out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, (1, 128, 56, 56))
        self.assertEqual(feat[1].shape, (1, 256, 28, 28))
        self.assertEqual(feat[2].shape, (1, 512, 14, 14))
        self.assertEqual(feat[3].shape, (1, 1024, 7, 7))
