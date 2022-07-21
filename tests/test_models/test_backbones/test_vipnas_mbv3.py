# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones import ViPNAS_MobileNetV3
from mmpose.models.backbones.utils import InvertedResidual


class TestVipnasMbv3(TestCase):

    @staticmethod
    def is_norm(modules):
        """Check if is one of the norms."""
        if isinstance(modules, (GroupNorm, _BatchNorm)):
            return True
        return False

    @staticmethod
    def check_norm_state(modules, train_state):
        """Check if norm layer is in correct train state."""
        for mod in modules:
            if isinstance(mod, _BatchNorm):
                if mod.training != train_state:
                    return False
        return True

    def test_mobilenetv3_backbone(self):
        with self.assertRaises(TypeError):
            # init_weights must have no parameter
            model = ViPNAS_MobileNetV3()
            model.init_weights(pretrained=0)

        with self.assertRaises(AttributeError):
            # frozen_stages must no more than 21
            model = ViPNAS_MobileNetV3(frozen_stages=22)
            model.train()

        # Test MobileNetv3
        model = ViPNAS_MobileNetV3()
        model.init_weights()
        model.train()

        # Test MobileNetv3 with first stage frozen
        frozen_stages = 1
        model = ViPNAS_MobileNetV3(frozen_stages=frozen_stages)
        model.init_weights()
        model.train()
        for param in model.conv1.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f'layer{i}')
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    self.assertFalse(mod.training)
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

        # Test MobileNetv3 with norm eval
        model = ViPNAS_MobileNetV3(norm_eval=True)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test MobileNetv3 forward
        model = ViPNAS_MobileNetV3()
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 160, 7, 7]))

        # Test MobileNetv3 forward with GroupNorm
        model = ViPNAS_MobileNetV3(
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
        for m in model.modules():
            if self.is_norm(m):
                self.assertIsInstance(m, GroupNorm)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 160, 7, 7]))

        # Test MobileNetv3 with checkpoint forward
        model = ViPNAS_MobileNetV3(with_cp=True)
        for m in model.modules():
            if isinstance(m, InvertedResidual):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 160, 7, 7]))
