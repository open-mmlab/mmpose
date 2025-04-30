# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.classification_loss import InfoNCELoss, VariFocalLoss


class TestInfoNCELoss(TestCase):

    def test_loss(self):

        # test loss w/o target_weight
        loss = InfoNCELoss(temperature=0.05)

        fake_pred = torch.arange(5 * 2).reshape(5, 2).float()
        self.assertTrue(
            torch.allclose(loss(fake_pred), torch.tensor(5.4026), atol=1e-4))

        # check if the value of temperature is positive
        with self.assertRaises(AssertionError):
            loss = InfoNCELoss(temperature=0.)


class TestVariFocalLoss(TestCase):

    def test_forward_no_target_weight_mean_reduction(self):
        # Test the forward method with no target weight and mean reduction
        output = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss_func = VariFocalLoss(use_target_weight=False, reduction='mean')
        loss = loss_func(output, target)

        # Calculate expected loss manually or using an alternative method
        expected_loss = 0.31683
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

    def test_forward_with_target_weight_sum_reduction(self):
        # Test the forward method with target weight and sum reduction
        output = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        target_weight = torch.tensor([1.0, 0.5], dtype=torch.float32)

        loss_func = VariFocalLoss(use_target_weight=True, reduction='sum')
        loss = loss_func(output, target, target_weight)

        # Calculate expected loss manually or using an alternative method
        expected_loss = 0.956299
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

    def test_inf_nan_handling(self):
        # Test handling of inf and nan values
        output = torch.tensor([[float('inf'), float('-inf')],
                               [float('nan'), 0.4]],
                              dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss_func = VariFocalLoss(use_target_weight=False, reduction='mean')
        loss = loss_func(output, target)

        # Check if loss is valid (not nan or inf)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
