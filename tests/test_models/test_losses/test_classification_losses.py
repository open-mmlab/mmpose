# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.classification_loss import InfoNCELoss


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
