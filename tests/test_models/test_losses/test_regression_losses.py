# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.regression_loss import SoftWeightSmoothL1Loss


class TestSoftWeightSmoothL1Loss(TestCase):

    def test_loss(self):

        # test loss w/o target_weight
        loss = SoftWeightSmoothL1Loss(use_target_weight=False, beta=0.5)

        fake_pred = torch.zeros((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.)))

        fake_pred = torch.ones((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(.75)))

        # test loss w/ target_weight
        loss = SoftWeightSmoothL1Loss(
            use_target_weight=True, supervise_empty=True)

        fake_pred = torch.ones((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        fake_weight = torch.arange(6).reshape(1, 3, 2).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight), torch.tensor(1.25)))

        # test loss that does not take empty channels into account
        loss = SoftWeightSmoothL1Loss(
            use_target_weight=True, supervise_empty=False)
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight), torch.tensor(1.5)))

        with self.assertRaises(ValueError):
            _ = loss.smooth_l1_loss(fake_pred, fake_label, reduction='fake')

        output = loss.smooth_l1_loss(fake_pred, fake_label, reduction='sum')
        self.assertTrue(torch.allclose(output, torch.tensor(3.0)))
