# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F

from mmpose.models.losses.regression_loss import (SmoothL1Loss,
                                                  SoftWeightSmoothL1Loss)


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


class TestSmoothL1Loss(TestCase):

    def test_mean_reduction(self):
        """Test the loss with mean reduction."""
        loss_fn = SmoothL1Loss(reduction='mean')
        output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        loss = loss_fn(output, target)
        expected_loss = F.smooth_l1_loss(output, target, reduction='mean')
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

    def test_sum_reduction(self):
        """Test the loss with sum reduction."""
        loss_fn = SmoothL1Loss(reduction='sum')
        output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        loss = loss_fn(output, target)
        expected_loss = F.smooth_l1_loss(output, target, reduction='sum')
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

    def test_with_target_weight(self):
        """Test the loss using target weights."""
        loss_fn = SmoothL1Loss(use_target_weight=True)
        output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        target_weight = torch.tensor([[1.0, 0.5], [1.0, 0.5]])
        loss = loss_fn(output, target, target_weight)
        # Manually compute expected loss with target weight
        weighted_output = output * target_weight
        weighted_target = target * target_weight
        expected_loss = F.smooth_l1_loss(
            weighted_output, weighted_target, reduction='mean')
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

    def test_no_target_weight(self):
        """Test the loss without using target weights."""
        loss_fn = SmoothL1Loss()
        output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        loss = loss_fn(output, target)
        expected_loss = F.smooth_l1_loss(output, target, reduction='mean')
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)
