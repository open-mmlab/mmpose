# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import numpy as np
import torch

from mmpose.core.optimizer import build_optimizers
from mmpose.models.detectors.mesh import ParametricMesh
from tests.utils.mesh_utils import generate_smpl_weight_file


def test_parametric_mesh_forward():
    """Test parametric mesh forward."""

    tmpdir = tempfile.TemporaryDirectory()
    # generate weight file for SMPL model.
    generate_smpl_weight_file(tmpdir.name)

    # Test ParametricMesh without discriminator
    model_cfg = dict(
        pretrained=None,
        backbone=dict(type='ResNet', depth=50),
        mesh_head=dict(
            type='HMRMeshHead',
            in_channels=2048,
            smpl_mean_params='tests/data/smpl/smpl_mean_params.npz'),
        disc=None,
        smpl=dict(
            type='SMPL',
            smpl_path=tmpdir.name,
            joints_regressor=osp.join(tmpdir.name,
                                      'test_joint_regressor.npy')),
        train_cfg=dict(disc_step=1),
        test_cfg=dict(
            flip_test=False,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11),
        loss_mesh=dict(
            type='MeshLoss',
            joints_2d_loss_weight=1,
            joints_3d_loss_weight=1,
            vertex_loss_weight=1,
            smpl_pose_loss_weight=1,
            smpl_beta_loss_weight=1,
            focal_length=5000,
            img_res=256),
        loss_gan=None)

    detector = ParametricMesh(**model_cfg)
    detector.init_weights()

    optimizers_config = dict(generator=dict(type='Adam', lr=0.0001))
    optims = build_optimizers(detector, optimizers_config)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)
    # Test forward train
    output = detector.train_step(mm_inputs, optims)
    assert isinstance(output, dict)

    # Test forward test
    with torch.no_grad():
        output = detector.val_step(data_batch=mm_inputs)
        assert isinstance(output, dict)

        imgs = mm_inputs.pop('img')
        img_metas = mm_inputs.pop('img_metas')
        output = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        assert isinstance(output, dict)

    # Test ParametricMesh with discriminator
    model_cfg['disc'] = dict()
    model_cfg['loss_gan'] = dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1)

    optimizers_config['discriminator'] = dict(type='Adam', lr=0.0001)

    detector = ParametricMesh(**model_cfg)
    detector.init_weights()
    optims = build_optimizers(detector, optimizers_config)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)
    # Test forward train
    output = detector.train_step(mm_inputs, optims)
    assert isinstance(output, dict)

    # Test forward test
    with torch.no_grad():
        output = detector.val_step(data_batch=mm_inputs)
        assert isinstance(output, dict)

        imgs = mm_inputs.pop('img')
        img_metas = mm_inputs.pop('img_metas')
        output = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        assert isinstance(output, dict)

        _ = detector.forward_dummy(imgs)

    tmpdir.cleanup()


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    joints_2d = np.zeros([N, 24, 2])
    joints_2d_visible = np.ones([N, 24, 1])
    joints_3d = np.zeros([N, 24, 3])
    joints_3d_visible = np.ones([N, 24, 1])
    pose = np.zeros([N, 72])
    beta = np.zeros([N, 10])
    has_smpl = np.ones([N])
    mosh_theta = np.zeros([N, 3 + 72 + 10])

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'img': torch.FloatTensor(imgs).requires_grad_(True),
        'joints_2d': torch.FloatTensor(joints_2d),
        'joints_2d_visible': torch.FloatTensor(joints_2d_visible),
        'joints_3d': torch.FloatTensor(joints_3d),
        'joints_3d_visible': torch.FloatTensor(joints_3d_visible),
        'pose': torch.FloatTensor(pose),
        'beta': torch.FloatTensor(beta),
        'has_smpl': torch.FloatTensor(has_smpl),
        'img_metas': img_metas,
        'mosh_theta': torch.FloatTensor(mosh_theta)
    }

    return mm_inputs
