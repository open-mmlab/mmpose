# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpose.models.backbones import VGG


class TestVGG(TestCase):

    @staticmethod
    def check_norm_state(modules, train_state):
        """Check if norm layer is in correct train state."""
        for mod in modules:
            if isinstance(mod, _BatchNorm):
                if mod.training != train_state:
                    return False
        return True

    def test_vgg(self):
        """Test VGG backbone."""
        with self.assertRaises(KeyError):
            # VGG depth should be in [11, 13, 16, 19]
            VGG(18)

        with self.assertRaises(AssertionError):
            # In VGG: 1 <= num_stages <= 5
            VGG(11, num_stages=0)

        with self.assertRaises(AssertionError):
            # In VGG: 1 <= num_stages <= 5
            VGG(11, num_stages=6)

        with self.assertRaises(AssertionError):
            # len(dilations) == num_stages
            VGG(11, dilations=(1, 1), num_stages=3)

        # Test VGG11 norm_eval=True
        model = VGG(11, norm_eval=True)
        model.init_weights()
        model.train()
        self.assertTrue(self.check_norm_state(model.modules(), False))

        # Test VGG11 forward without classifiers
        model = VGG(11, out_indices=(0, 1, 2, 3, 4))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 5)
        self.assertEqual(feat[0].shape, (1, 64, 112, 112))
        self.assertEqual(feat[1].shape, (1, 128, 56, 56))
        self.assertEqual(feat[2].shape, (1, 256, 28, 28))
        self.assertEqual(feat[3].shape, (1, 512, 14, 14))
        self.assertEqual(feat[4].shape, (1, 512, 7, 7))

        # Test VGG11 forward with classifiers
        model = VGG(11, num_classes=10, out_indices=(0, 1, 2, 3, 4, 5))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 6)
        self.assertEqual(feat[0].shape, (1, 64, 112, 112))
        self.assertEqual(feat[1].shape, (1, 128, 56, 56))
        self.assertEqual(feat[2].shape, (1, 256, 28, 28))
        self.assertEqual(feat[3].shape, (1, 512, 14, 14))
        self.assertEqual(feat[4].shape, (1, 512, 7, 7))
        self.assertEqual(feat[5].shape, (1, 10))

        # Test VGG11BN forward
        model = VGG(11, norm_cfg=dict(type='BN'), out_indices=(0, 1, 2, 3, 4))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 5)
        self.assertEqual(feat[0].shape, (1, 64, 112, 112))
        self.assertEqual(feat[1].shape, (1, 128, 56, 56))
        self.assertEqual(feat[2].shape, (1, 256, 28, 28))
        self.assertEqual(feat[3].shape, (1, 512, 14, 14))
        self.assertEqual(feat[4].shape, (1, 512, 7, 7))

        # Test VGG11BN forward with classifiers
        model = VGG(
            11,
            num_classes=10,
            norm_cfg=dict(type='BN'),
            out_indices=(0, 1, 2, 3, 4, 5))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 6)
        self.assertEqual(feat[0].shape, (1, 64, 112, 112))
        self.assertEqual(feat[1].shape, (1, 128, 56, 56))
        self.assertEqual(feat[2].shape, (1, 256, 28, 28))
        self.assertEqual(feat[3].shape, (1, 512, 14, 14))
        self.assertEqual(feat[4].shape, (1, 512, 7, 7))
        self.assertEqual(feat[5].shape, (1, 10))

        # Test VGG13 with layers 1, 2, 3 out forward
        model = VGG(13, out_indices=(0, 1, 2))
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 3)
        self.assertEqual(feat[0].shape, (1, 64, 112, 112))
        self.assertEqual(feat[1].shape, (1, 128, 56, 56))
        self.assertEqual(feat[2].shape, (1, 256, 28, 28))

        # Test VGG16 with top feature maps out forward
        model = VGG(16)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[-1].shape, (1, 512, 7, 7))

        # Test VGG19 with classification score out forward
        model = VGG(19, num_classes=10)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[-1].shape, (1, 10))
