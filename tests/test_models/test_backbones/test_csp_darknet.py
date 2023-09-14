# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones.csp_darknet import CSPDarknet


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class TestCSPDarknetBackbone(unittest.TestCase):

    def test_invalid_frozen_stages(self):
        with self.assertRaises(ValueError):
            CSPDarknet(frozen_stages=6)

    def test_invalid_out_indices(self):
        with self.assertRaises(AssertionError):
            CSPDarknet(out_indices=[6])

    def test_frozen_stages(self):
        frozen_stages = 1
        model = CSPDarknet(frozen_stages=frozen_stages)
        model.train()

        for mod in model.stem.modules():
            for param in mod.parameters():
                self.assertFalse(param.requires_grad)
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f'stage{i}')
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    self.assertFalse(mod.training)
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

    def test_norm_eval(self):
        model = CSPDarknet(norm_eval=True)
        model.train()

        self.assertFalse(check_norm_state(model.modules(), True))

    def test_csp_darknet_p5_forward(self):
        model = CSPDarknet(
            arch='P5', widen_factor=0.25, out_indices=range(0, 5))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        self.assertEqual(len(feat), 5)
        self.assertEqual(feat[0].shape, torch.Size((1, 16, 32, 32)))
        self.assertEqual(feat[1].shape, torch.Size((1, 32, 16, 16)))
        self.assertEqual(feat[2].shape, torch.Size((1, 64, 8, 8)))
        self.assertEqual(feat[3].shape, torch.Size((1, 128, 4, 4)))
        self.assertEqual(feat[4].shape, torch.Size((1, 256, 2, 2)))

    def test_csp_darknet_p6_forward(self):
        model = CSPDarknet(
            arch='P6',
            widen_factor=0.25,
            out_indices=range(0, 6),
            spp_kernal_sizes=(3, 5, 7))
        model.train()

        imgs = torch.randn(1, 3, 128, 128)
        feat = model(imgs)
        self.assertEqual(feat[0].shape, torch.Size((1, 16, 64, 64)))
        self.assertEqual(feat[1].shape, torch.Size((1, 32, 32, 32)))
        self.assertEqual(feat[2].shape, torch.Size((1, 64, 16, 16)))
        self.assertEqual(feat[3].shape, torch.Size((1, 128, 8, 8)))
        self.assertEqual(feat[4].shape, torch.Size((1, 192, 4, 4)))
        self.assertEqual(feat[5].shape, torch.Size((1, 256, 2, 2)))

    def test_csp_darknet_custom_arch_forward(self):
        arch_ovewrite = [[32, 56, 3, True, False], [56, 224, 2, True, False],
                         [224, 512, 1, True, False]]
        model = CSPDarknet(
            arch_ovewrite=arch_ovewrite,
            widen_factor=0.25,
            out_indices=(0, 1, 2, 3))
        model.train()

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size((1, 8, 16, 16)))
        self.assertEqual(feat[1].shape, torch.Size((1, 14, 8, 8)))
        self.assertEqual(feat[2].shape, torch.Size((1, 56, 4, 4)))
        self.assertEqual(feat[3].shape, torch.Size((1, 128, 2, 2)))

    def test_csp_darknet_custom_arch_norm(self):
        model = CSPDarknet(widen_factor=0.125, out_indices=range(0, 5))
        for m in model.modules():
            if is_norm(m):
                self.assertIsInstance(m, _BatchNorm)
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        self.assertEqual(len(feat), 5)
        self.assertEqual(feat[0].shape, torch.Size((1, 8, 32, 32)))
        self.assertEqual(feat[1].shape, torch.Size((1, 16, 16, 16)))
        self.assertEqual(feat[2].shape, torch.Size((1, 32, 8, 8)))
        self.assertEqual(feat[3].shape, torch.Size((1, 64, 4, 4)))
        self.assertEqual(feat[4].shape, torch.Size((1, 128, 2, 2)))


if __name__ == '__main__':
    unittest.main()
