# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones import SCNet
from mmpose.models.backbones.scnet import SCBottleneck, SCConv


class TestSCnet(TestCase):

    @staticmethod
    def is_block(modules):
        """Check if is SCNet building block."""
        if isinstance(modules, (SCBottleneck, )):
            return True
        return False

    @staticmethod
    def is_norm(modules):
        """Check if is one of the norms."""
        if isinstance(modules, (_BatchNorm, )):
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

    def test_scnet_scconv(self):
        # Test scconv forward
        layer = SCConv(64, 64, 1, 4)
        x = torch.randn(1, 64, 56, 56)
        x_out = layer(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_scnet_bottleneck(self):
        # Test Bottleneck forward
        block = SCBottleneck(64, 64)
        x = torch.randn(1, 64, 56, 56)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([1, 64, 56, 56]))

    def test_scnet_backbone(self):
        """Test scnet backbone."""
        with self.assertRaises(KeyError):
            # SCNet depth should be in [50, 101]
            SCNet(20)

        with self.assertRaises(TypeError):
            # pretrained must be a string path
            model = SCNet(50)
            model.init_weights(pretrained=0)

        # Test SCNet norm_eval=True
        model = SCNet(50, norm_eval=True)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test SCNet50 with first stage frozen
        frozen_stages = 1
        model = SCNet(50, frozen_stages=frozen_stages)
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

        # Test SCNet with BatchNorm forward
        model = SCNet(50, out_indices=(0, 1, 2, 3))
        for m in model.modules():
            if self.is_norm(m):
                self.assertIsInstance(m, _BatchNorm)
        model.init_weights()
        model.train()

        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([2, 256, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([2, 512, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([2, 1024, 14, 14]))
        self.assertEqual(feat[3].shape, torch.Size([2, 2048, 7, 7]))

        # Test SCNet with layers 1, 2, 3 out forward
        model = SCNet(50, out_indices=(0, 1, 2))
        model.init_weights()
        model.train()

        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 3)
        self.assertEqual(feat[0].shape, torch.Size([2, 256, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([2, 512, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([2, 1024, 14, 14]))

        # Test SEResNet50 with layers 3 (top feature maps) out forward
        model = SCNet(50, out_indices=(3, ))
        model.init_weights()
        model.train()

        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([2, 2048, 7, 7]))

        # Test SEResNet50 with checkpoint forward
        model = SCNet(50, out_indices=(0, 1, 2, 3), with_cp=True)
        for m in model.modules():
            if self.is_block(m):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([2, 256, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([2, 512, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([2, 1024, 14, 14]))
        self.assertEqual(feat[3].shape, torch.Size([2, 2048, 7, 7]))

        # Test SCNet zero initialization of residual
        model = SCNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=True)
        model.init_weights()
        for m in model.modules():
            if isinstance(m, SCBottleneck):
                self.assertTrue(self.all_zeros(m.norm3))
        model.train()

        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([2, 256, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([2, 512, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([2, 1024, 14, 14]))
        self.assertEqual(feat[3].shape, torch.Size([2, 2048, 7, 7]))
