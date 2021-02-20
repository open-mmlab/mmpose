import torch


def test_smooth_l1_loss():
    from mmpose.models import build_loss

    # test SmoothL1Loss without target weight
    loss_cfg = dict(type='SmoothL1Loss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    # test SmoothL1Loss with target weight
    loss_cfg = dict(type='SmoothL1Loss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(1.))


def test_wing_loss():
    from mmpose.models import build_loss

    # test WingLoss without target weight
    loss_cfg = dict(type='WingLoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(loss(fake_pred, fake_label, None), torch.tensor(1.))

    # test WingLoss with target weight
    loss_cfg = dict(type='WingLoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(1.))
