# Copyright (c) OpenMMLab. All rights reserved.
import torch


def test_bce_loss():
    from mmpose.models import build_loss

    # test BCE loss without target weight(None)
    loss_cfg = dict(type='BCELoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 2))
    fake_label = torch.zeros((1, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 2)) * 0.5
    fake_label = torch.zeros((1, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label), -torch.log(torch.tensor(0.5)))

    # test BCE loss with target weight
    loss_cfg = dict(type='BCELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.ones((1, 2)) * 0.5
    fake_label = torch.zeros((1, 2))
    fake_weight = torch.ones((1, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight),
        -torch.log(torch.tensor(0.5)))

    fake_weight[:, 0] = 0
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight),
        -0.5 * torch.log(torch.tensor(0.5)))

    fake_weight = torch.ones(1)
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight),
        -torch.log(torch.tensor(0.5)))
