# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


def test_multi_loss_factory():
    from mmpose.models import build_loss

    # test heatmap loss
    loss_cfg = dict(type='HeatmapLoss')
    loss = build_loss(loss_cfg)

    with pytest.raises(AssertionError):
        fake_pred = torch.zeros((2, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        fake_mask = torch.zeros((1, 64, 64))
        loss(fake_pred, fake_label, fake_mask)

    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    fake_mask = torch.zeros((1, 64, 64))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_mask), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    fake_mask = torch.zeros((1, 64, 64))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_mask), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    fake_mask = torch.ones((1, 64, 64))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_mask), torch.tensor(1.))

    # test AE loss
    fake_tags = torch.zeros((1, 18, 1))
    fake_joints = torch.zeros((1, 3, 2, 2), dtype=torch.int)

    loss_cfg = dict(type='AELoss', loss_type='exp')
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(fake_tags, fake_joints)[0], torch.tensor(0.))
    assert torch.allclose(loss(fake_tags, fake_joints)[1], torch.tensor(0.))

    fake_tags[0, 0, 0] = 1.
    fake_tags[0, 10, 0] = 0.
    fake_joints[0, 0, 0, :] = torch.IntTensor((0, 1))
    fake_joints[0, 0, 1, :] = torch.IntTensor((10, 1))
    loss_cfg = dict(type='AELoss', loss_type='exp')
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(fake_tags, fake_joints)[0], torch.tensor(0.))
    assert torch.allclose(loss(fake_tags, fake_joints)[1], torch.tensor(0.25))

    fake_tags[0, 0, 0] = 0
    fake_tags[0, 7, 0] = 1.
    fake_tags[0, 17, 0] = 1.
    fake_joints[0, 1, 0, :] = torch.IntTensor((7, 1))
    fake_joints[0, 1, 1, :] = torch.IntTensor((17, 1))

    loss_cfg = dict(type='AELoss', loss_type='exp')
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(fake_tags, fake_joints)[1], torch.tensor(0.))

    loss_cfg = dict(type='AELoss', loss_type='max')
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(fake_tags, fake_joints)[0], torch.tensor(0.))

    with pytest.raises(ValueError):
        loss_cfg = dict(type='AELoss', loss_type='min')
        loss = build_loss(loss_cfg)
        loss(fake_tags, fake_joints)

    # test MultiLossFactory
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='MultiLossFactory',
            num_joints=2,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=True,
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])
        loss = build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='MultiLossFactory',
            num_joints=2,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=0.001,
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])
        loss = build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='MultiLossFactory',
            num_joints=2,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=0.001,
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])
        loss = build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='MultiLossFactory',
            num_joints=2,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=True,
            heatmaps_loss_factor=[1.0])
        loss = build_loss(loss_cfg)
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='MultiLossFactory',
            num_joints=2,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=1.0)
        loss = build_loss(loss_cfg)
    loss_cfg = dict(
        type='MultiLossFactory',
        num_joints=17,
        num_stages=1,
        ae_loss_type='exp',
        with_ae_loss=[False],
        push_loss_factor=[0.001],
        pull_loss_factor=[0.001],
        with_heatmaps_loss=[False],
        heatmaps_loss_factor=[1.0])
    loss = build_loss(loss_cfg)
    fake_outputs = [torch.zeros((1, 34, 64, 64))]
    fake_heatmaps = [torch.zeros((1, 17, 64, 64))]
    fake_masks = [torch.ones((1, 64, 64))]
    fake_joints = [torch.zeros((1, 30, 17, 2))]
    heatmaps_losses, push_losses, pull_losses = \
        loss(fake_outputs, fake_heatmaps, fake_masks, fake_joints)
    assert heatmaps_losses == [None]
    assert pull_losses == [None]
    assert push_losses == [None]
    loss_cfg = dict(
        type='MultiLossFactory',
        num_joints=17,
        num_stages=1,
        ae_loss_type='exp',
        with_ae_loss=[True],
        push_loss_factor=[0.001],
        pull_loss_factor=[0.001],
        with_heatmaps_loss=[True],
        heatmaps_loss_factor=[1.0])
    loss = build_loss(loss_cfg)
    heatmaps_losses, push_losses, pull_losses = \
        loss(fake_outputs, fake_heatmaps, fake_masks, fake_joints)
    assert len(heatmaps_losses) == 1
