# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import AlexNet


class TestAlexNet(TestCase):

    def test_alexnet_backbone(self):
        """Test alexnet backbone."""
        model = AlexNet(-1)
        model.train()

        imgs = torch.randn(1, 3, 256, 192)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, (1, 256, 7, 5))

        model = AlexNet(1)
        model.train()

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        self.assertIsInstance(feat, tuple)
        self.assertEqual(feat[-1].shape, (1, 1))
