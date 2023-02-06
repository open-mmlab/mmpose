# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.heatmap_loss import AdaptiveWingLoss


class TestAdaptiveWingLoss(TestCase):

    def test_loss(self):

        # test loss w/o target_weight
        loss = AdaptiveWingLoss(use_target_weight=False)

        fake_pred = torch.zeros((1, 3, 2, 2))
        fake_label = torch.zeros((1, 3, 2, 2))
        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.)))

        fake_pred = torch.ones((1, 3, 2, 2))
        fake_label = torch.zeros((1, 3, 2, 2))
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label), torch.tensor(8.4959), atol=1e-4))

        # test loss w/ target_weight
        loss = AdaptiveWingLoss(use_target_weight=True)

        fake_pred = torch.zeros((1, 3, 2, 2))
        fake_label = torch.zeros((1, 3, 2, 2))
        fake_weight = torch.tensor([1, 0, 1]).reshape(1, 3).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight), torch.tensor(0.)))
