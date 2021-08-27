# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from numpy.testing import assert_almost_equal

from mmpose.models import build_loss
from mmpose.models.utils.geometry import batch_rodrigues


def test_mesh_loss():
    """test mesh loss."""
    loss_cfg = dict(
        type='MeshLoss',
        joints_2d_loss_weight=1,
        joints_3d_loss_weight=1,
        vertex_loss_weight=1,
        smpl_pose_loss_weight=1,
        smpl_beta_loss_weight=1,
        img_res=256,
        focal_length=5000)

    loss = build_loss(loss_cfg)

    smpl_pose = torch.zeros([1, 72], dtype=torch.float32)
    smpl_rotmat = batch_rodrigues(smpl_pose.view(-1, 3)).view(-1, 24, 3, 3)
    smpl_beta = torch.zeros([1, 10], dtype=torch.float32)
    camera = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    vertices = torch.rand([1, 6890, 3], dtype=torch.float32)
    joints_3d = torch.ones([1, 24, 3], dtype=torch.float32)
    joints_2d = loss.project_points(joints_3d, camera) + (256 - 1) / 2

    fake_pred = {}
    fake_pred['pose'] = smpl_rotmat
    fake_pred['beta'] = smpl_beta
    fake_pred['camera'] = camera
    fake_pred['vertices'] = vertices
    fake_pred['joints_3d'] = joints_3d

    fake_gt = {}
    fake_gt['pose'] = smpl_pose
    fake_gt['beta'] = smpl_beta
    fake_gt['vertices'] = vertices
    fake_gt['has_smpl'] = torch.ones(1, dtype=torch.float32)
    fake_gt['joints_3d'] = joints_3d
    fake_gt['joints_3d_visible'] = torch.ones([1, 24, 1], dtype=torch.float32)
    fake_gt['joints_2d'] = joints_2d
    fake_gt['joints_2d_visible'] = torch.ones([1, 24, 1], dtype=torch.float32)

    losses = loss(fake_pred, fake_gt)
    assert torch.allclose(losses['vertex_loss'], torch.tensor(0.))
    assert torch.allclose(losses['smpl_pose_loss'], torch.tensor(0.))
    assert torch.allclose(losses['smpl_beta_loss'], torch.tensor(0.))
    assert torch.allclose(losses['joints_3d_loss'], torch.tensor(0.))
    assert torch.allclose(losses['joints_2d_loss'], torch.tensor(0.))

    fake_pred = {}
    fake_pred['pose'] = smpl_rotmat + 1
    fake_pred['beta'] = smpl_beta + 1
    fake_pred['camera'] = camera
    fake_pred['vertices'] = vertices + 1
    fake_pred['joints_3d'] = joints_3d.clone()

    joints_3d_t = joints_3d.clone()
    joints_3d_t[:, 0] = joints_3d_t[:, 0] + 1
    fake_gt = {}
    fake_gt['pose'] = smpl_pose
    fake_gt['beta'] = smpl_beta
    fake_gt['vertices'] = vertices
    fake_gt['has_smpl'] = torch.ones(1, dtype=torch.float32)
    fake_gt['joints_3d'] = joints_3d_t
    fake_gt['joints_3d_visible'] = torch.ones([1, 24, 1], dtype=torch.float32)
    fake_gt['joints_2d'] = joints_2d + (256 - 1) / 2
    fake_gt['joints_2d_visible'] = torch.ones([1, 24, 1], dtype=torch.float32)

    losses = loss(fake_pred, fake_gt)
    assert torch.allclose(losses['vertex_loss'], torch.tensor(1.))
    assert torch.allclose(losses['smpl_pose_loss'], torch.tensor(1.))
    assert torch.allclose(losses['smpl_beta_loss'], torch.tensor(1.))
    assert torch.allclose(losses['joints_3d_loss'], torch.tensor(0.5 / 24))
    assert torch.allclose(losses['joints_2d_loss'], torch.tensor(0.5))


def test_gan_loss():
    """test gan loss."""
    with pytest.raises(NotImplementedError):
        loss_cfg = dict(
            type='GANLoss',
            gan_type='test',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1)
        _ = build_loss(loss_cfg)

    input_1 = torch.ones(1, 1)
    input_2 = torch.ones(1, 3, 6, 6) * 2

    # vanilla
    loss_cfg = dict(
        type='GANLoss',
        gan_type='vanilla',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=2.0)
    gan_loss = build_loss(loss_cfg)
    loss = gan_loss(input_1, True, is_disc=False)
    assert_almost_equal(loss.item(), 0.6265233)
    loss = gan_loss(input_1, False, is_disc=False)
    assert_almost_equal(loss.item(), 2.6265232)
    loss = gan_loss(input_1, True, is_disc=True)
    assert_almost_equal(loss.item(), 0.3132616)
    loss = gan_loss(input_1, False, is_disc=True)
    assert_almost_equal(loss.item(), 1.3132616)

    # lsgan
    loss_cfg = dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=2.0)
    gan_loss = build_loss(loss_cfg)
    loss = gan_loss(input_2, True, is_disc=False)
    assert_almost_equal(loss.item(), 2.0)
    loss = gan_loss(input_2, False, is_disc=False)
    assert_almost_equal(loss.item(), 8.0)
    loss = gan_loss(input_2, True, is_disc=True)
    assert_almost_equal(loss.item(), 1.0)
    loss = gan_loss(input_2, False, is_disc=True)
    assert_almost_equal(loss.item(), 4.0)

    # wgan
    loss_cfg = dict(
        type='GANLoss',
        gan_type='wgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=2.0)
    gan_loss = build_loss(loss_cfg)
    loss = gan_loss(input_2, True, is_disc=False)
    assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    assert_almost_equal(loss.item(), 4)
    loss = gan_loss(input_2, True, is_disc=True)
    assert_almost_equal(loss.item(), -2.0)
    loss = gan_loss(input_2, False, is_disc=True)
    assert_almost_equal(loss.item(), 2.0)

    # hinge
    loss_cfg = dict(
        type='GANLoss',
        gan_type='hinge',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=2.0)
    gan_loss = build_loss(loss_cfg)
    loss = gan_loss(input_2, True, is_disc=False)
    assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, True, is_disc=True)
    assert_almost_equal(loss.item(), 0.0)
    loss = gan_loss(input_2, False, is_disc=True)
    assert_almost_equal(loss.item(), 3.0)
