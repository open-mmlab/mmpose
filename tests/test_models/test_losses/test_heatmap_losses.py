# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.heatmap_loss import (AdaptiveWingLoss,
                                               FocalHeatmapLoss,
                                               KeypointMSELoss, MLECCLoss)


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


class TestFocalHeatmapLoss(TestCase):

    def test_loss(self):

        loss = FocalHeatmapLoss(use_target_weight=False)

        fake_pred = torch.zeros((1, 3, 5, 5))
        fake_label = torch.zeros((1, 3, 5, 5))

        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.)))

        fake_pred = torch.ones((1, 3, 5, 5)) * 0.4
        fake_label = torch.ones((1, 3, 5, 5)) * 0.6
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label), torch.tensor(0.1569), atol=1e-4))

        # test loss w/ target_weight
        loss = FocalHeatmapLoss(use_target_weight=True)

        fake_weight = torch.arange(3 * 5 * 5).reshape(1, 3, 5, 5).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight),
                torch.tensor(5.8062),
                atol=1e-4))


class TestKeypointMSELoss(TestCase):

    def test_loss(self):

        # test loss w/o target_weight and without mask
        loss = KeypointMSELoss(
            use_target_weight=False, skip_empty_channel=False)

        fake_pred = torch.zeros((1, 4, 4, 4))
        fake_label = torch.zeros((1, 4, 4, 4))

        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.)))

        fake_pred = torch.ones((1, 4, 4, 4)) * 0.5
        fake_label = torch.ones((1, 4, 4, 4)) * 0.5
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label), torch.tensor(0.), atol=1e-4))

        # test loss w/ target_weight and without mask
        loss = KeypointMSELoss(
            use_target_weight=True, skip_empty_channel=False)

        fake_weight = torch.ones((1, 4)).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight),
                torch.tensor(0.),
                atol=1e-4))

        # test loss w/ target_weight and with mask
        loss = KeypointMSELoss(
            use_target_weight=True, skip_empty_channel=False)

        fake_mask = torch.ones((1, 1, 4, 4)).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight, fake_mask),
                torch.tensor(0.),
                atol=1e-4))

        # test loss w/ target_weight and skip empty channels
        loss = KeypointMSELoss(use_target_weight=True, skip_empty_channel=True)

        fake_mask = torch.ones((1, 1, 4, 4)).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight, fake_mask),
                torch.tensor(0.),
                atol=1e-4))


class TestMLECCLoss(TestCase):

    def setUp(self):
        self.outputs = (torch.rand(10, 2, 5), torch.rand(10, 2, 5))
        self.targets = (torch.rand(10, 2, 5), torch.rand(10, 2, 5))

    def test_mean_reduction_log_mode(self):
        loss_func = MLECCLoss(reduction='mean', mode='log')
        loss = loss_func(self.outputs, self.targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_sum_reduction_linear_mode(self):
        loss_func = MLECCLoss(reduction='sum', mode='linear')
        loss = loss_func(self.outputs, self.targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_none_reduction_square_mode(self):
        loss_func = MLECCLoss(reduction='none', mode='square')
        loss = loss_func(self.outputs, self.targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_target_weight(self):
        target_weight = torch.rand(10)  # Random weights
        loss_func = MLECCLoss(use_target_weight=True)
        loss = loss_func(self.outputs, self.targets, target_weight)
        self.assertIsInstance(loss, torch.Tensor)

    def test_invalid_reduction(self):
        with self.assertRaises(AssertionError):
            MLECCLoss(reduction='invalid_reduction')

    def test_invalid_mode(self):
        with self.assertRaises(AssertionError):
            MLECCLoss(mode='invalid_mode')
