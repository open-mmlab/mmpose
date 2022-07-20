# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpose.models import build_loss


def test_adaptive_wing_loss():
    # test Adaptive WingLoss without target weight
    loss_cfg = dict(type='AdaptiveWingLoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    # test WingLoss with target weight
    loss_cfg = dict(type='AdaptiveWingLoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.ones((1, 3, 64, 64))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones([1, 3, 1])), torch.tensor(0.))


def test_mse_loss():
    # test MSE loss without target weight
    loss_cfg = dict(type='JointsMSELoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    fake_pred = torch.zeros((1, 2, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 2, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))

    with pytest.raises(ValueError):
        loss_cfg = dict(type='JointsOHKMMSELoss')
        loss = build_loss(loss_cfg)
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    with pytest.raises(AssertionError):
        loss_cfg = dict(type='JointsOHKMMSELoss', topk=-1)
        loss = build_loss(loss_cfg)
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    loss_cfg = dict(type='JointsOHKMMSELoss', topk=2)
    loss = build_loss(loss_cfg)
    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    loss_cfg = dict(type='JointsOHKMMSELoss', topk=2)
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))

    loss_cfg = dict(type='CombinedTargetMSELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)
    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    target_weight = torch.ones((1, 1, 1))
    assert torch.allclose(
        loss(fake_pred, fake_label, target_weight), torch.tensor(0.5))

    loss_cfg = dict(type='CombinedTargetMSELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)
    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    target_weight = torch.zeros((1, 1, 1))
    assert torch.allclose(
        loss(fake_pred, fake_label, target_weight), torch.tensor(0.))


def test_smoothl1_loss():
    # test MSE loss without target weight
    loss_cfg = dict(type='SmoothL1Loss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3))
    fake_label = torch.zeros((1, 3))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))
