# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones import MobileNetV3
from mmpose.models.backbones.utils import InvertedResidual


class TestMobilenetV3(TestCase):

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
        with self.assertRaises(AssertionError):
            # arch must in [small, big]
            MobileNetV3(arch='others')

        with self.assertRaises(ValueError):
            # frozen_stages must less than 12 when arch is small
            MobileNetV3(arch='small', frozen_stages=12)

        with self.assertRaises(ValueError):
            # frozen_stages must less than 16 when arch is big
            MobileNetV3(arch='big', frozen_stages=16)

        with self.assertRaises(ValueError):
            # max out_indices must less than 11 when arch is small
            MobileNetV3(arch='small', out_indices=(11, ))

        with self.assertRaises(ValueError):
            # max out_indices must less than 15 when arch is big
            MobileNetV3(arch='big', out_indices=(15, ))

        # Test MobileNetv3
        model = MobileNetV3()
        model.init_weights()
        model.train()

        # Test MobileNetv3 with first stage frozen
        frozen_stages = 1
        model = MobileNetV3(frozen_stages=frozen_stages)
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
        model = MobileNetV3(norm_eval=True, out_indices=range(0, 11))
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test MobileNetv3 forward with small arch
        model = MobileNetV3(out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 11)
        self.assertEqual(feat[0].shape, torch.Size([1, 16, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([1, 24, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([1, 24, 28, 28]))
        self.assertEqual(feat[3].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[4].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[5].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[6].shape, torch.Size([1, 48, 14, 14]))
        self.assertEqual(feat[7].shape, torch.Size([1, 48, 14, 14]))
        self.assertEqual(feat[8].shape, torch.Size([1, 96, 7, 7]))
        self.assertEqual(feat[9].shape, torch.Size([1, 96, 7, 7]))
        self.assertEqual(feat[10].shape, torch.Size([1, 96, 7, 7]))

        # Test MobileNetv3 forward with small arch and GroupNorm
        model = MobileNetV3(
            out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
        for m in model.modules():
            if self.is_norm(m):
                self.assertIsInstance(m, GroupNorm)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 11)
        self.assertEqual(feat[0].shape, torch.Size([1, 16, 56, 56]))
        self.assertEqual(feat[1].shape, torch.Size([1, 24, 28, 28]))
        self.assertEqual(feat[2].shape, torch.Size([1, 24, 28, 28]))
        self.assertEqual(feat[3].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[4].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[5].shape, torch.Size([1, 40, 14, 14]))
        self.assertEqual(feat[6].shape, torch.Size([1, 48, 14, 14]))
        self.assertEqual(feat[7].shape, torch.Size([1, 48, 14, 14]))
        self.assertEqual(feat[8].shape, torch.Size([1, 96, 7, 7]))
        self.assertEqual(feat[9].shape, torch.Size([1, 96, 7, 7]))
        self.assertEqual(feat[10].shape, torch.Size([1, 96, 7, 7]))

        # Test MobileNetv3 forward with big arch
        model = MobileNetV3(
            arch='big',
            out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 15)
        self.assertEqual(feat[0].shape, torch.Size([1, 16, 112, 112]))
        self.assertEqual(feat[1].shape, torch.Size([1, 24, 56, 56]))
        self.assertEqual(feat[2].shape, torch.Size([1, 24, 56, 56]))
        self.assertEqual(feat[3].shape, torch.Size([1, 40, 28, 28]))
        self.assertEqual(feat[4].shape, torch.Size([1, 40, 28, 28]))
        self.assertEqual(feat[5].shape, torch.Size([1, 40, 28, 28]))
        self.assertEqual(feat[6].shape, torch.Size([1, 80, 14, 14]))
        self.assertEqual(feat[7].shape, torch.Size([1, 80, 14, 14]))
        self.assertEqual(feat[8].shape, torch.Size([1, 80, 14, 14]))
        self.assertEqual(feat[9].shape, torch.Size([1, 80, 14, 14]))
        self.assertEqual(feat[10].shape, torch.Size([1, 112, 14, 14]))
        self.assertEqual(feat[11].shape, torch.Size([1, 112, 14, 14]))
        self.assertEqual(feat[12].shape, torch.Size([1, 160, 14, 14]))
        self.assertEqual(feat[13].shape, torch.Size([1, 160, 7, 7]))
        self.assertEqual(feat[14].shape, torch.Size([1, 160, 7, 7]))

        # Test MobileNetv3 forward with big arch
        model = MobileNetV3(arch='big', out_indices=(0, ))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 16, 112, 112]))

        # Test MobileNetv3 with checkpoint forward
        model = MobileNetV3(with_cp=True)
        for m in model.modules():
            if isinstance(m, InvertedResidual):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, torch.Size([1, 96, 7, 7]))
