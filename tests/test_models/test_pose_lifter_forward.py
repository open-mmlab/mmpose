# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmpose.models import build_posenet


def _create_inputs(joint_num_in,
                   joint_channel_in,
                   joint_num_out,
                   joint_channel_out,
                   seq_len,
                   batch_size,
                   semi=False):
    rng = np.random.RandomState(0)
    pose_in = rng.rand(batch_size, joint_num_in * joint_channel_in, seq_len)
    target = np.zeros((batch_size, joint_num_out, joint_channel_out),
                      dtype=np.float32)
    target_weight = np.ones((batch_size, joint_num_out, joint_channel_out),
                            dtype=np.float32)

    meta_info = {
        'root_position': np.zeros((1, joint_channel_out), np.float32),
        'root_position_index': 0,
        'target_mean': np.zeros((joint_num_out, joint_channel_out),
                                np.float32),
        'target_std': np.ones((joint_num_out, joint_channel_out), np.float32)
    }
    metas = [meta_info.copy() for _ in range(batch_size)]
    inputs = {
        'input': torch.FloatTensor(pose_in).requires_grad_(True),
        'target': torch.FloatTensor(target),
        'target_weight': torch.FloatTensor(target_weight),
        'metas': metas,
    }

    if semi:
        traj_target = np.zeros((batch_size, 1, joint_channel_out), np.float32)
        unlabeled_pose_in = rng.rand(batch_size,
                                     joint_num_in * joint_channel_in, seq_len)
        unlabeled_target_2d = np.zeros(
            (batch_size, joint_num_out, joint_channel_in), dtype=np.float32)
        intrinsics = np.ones((batch_size, 4))

        inputs['traj_target'] = torch.FloatTensor(traj_target)
        inputs['unlabeled_input'] = torch.FloatTensor(
            unlabeled_pose_in).requires_grad_(True)
        inputs['unlabeled_target_2d'] = torch.FloatTensor(unlabeled_target_2d)
        inputs['intrinsics'] = torch.FloatTensor(intrinsics)

    return inputs


def test_pose_lifter_forward():
    # Test forward train for supervised learning with pose model only
    model_cfg = dict(
        type='PoseLifter',
        pretrained=None,
        backbone=dict(type='TCN', in_channels=2 * 17),
        keypoint_head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=16,
            max_norm=1.0,
            loss_keypoint=dict(type='MPJPELoss'),
            test_cfg=dict(restore_global_position=True)),
        train_cfg=dict(),
        test_cfg=dict())

    cfg = mmcv.Config({'model': model_cfg})
    detector = build_posenet(cfg.model)

    with pytest.raises(TypeError):
        detector.init_weights(pretrained=dict())
    detector.pretrained = model_cfg['pretrained']
    detector.init_weights()

    inputs = _create_inputs(
        joint_num_in=17,
        joint_channel_in=2,
        joint_num_out=16,
        joint_channel_out=3,
        seq_len=27,
        batch_size=8)

    losses = detector.forward(
        inputs['input'],
        inputs['target'],
        inputs['target_weight'],
        inputs['metas'],
        return_loss=True)

    assert isinstance(losses, dict)

    # Test forward test for supervised learning with pose model only
    with torch.no_grad():
        _ = detector.forward(
            inputs['input'],
            inputs['target'],
            inputs['target_weight'],
            inputs['metas'],
            return_loss=False)
        _ = detector.forward_dummy(inputs['input'])

    # Test forward train for semi-supervised learning
    model_cfg = dict(
        type='PoseLifter',
        pretrained=None,
        backbone=dict(type='TCN', in_channels=2 * 17),
        keypoint_head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss_keypoint=dict(type='MPJPELoss'),
            test_cfg=dict(restore_global_position=True)),
        traj_backbone=dict(type='TCN', in_channels=2 * 17),
        traj_head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=1,
            loss_keypoint=dict(type='MPJPELoss'),
            is_trajectory=True),
        loss_semi=dict(
            type='SemiSupervisionLoss',
            joint_parents=[
                0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15
            ]),
        train_cfg=dict(),
        test_cfg=dict())

    cfg = mmcv.Config({'model': model_cfg})
    detector = build_posenet(cfg.model)

    detector.init_weights()

    inputs = _create_inputs(
        joint_num_in=17,
        joint_channel_in=2,
        joint_num_out=17,
        joint_channel_out=3,
        seq_len=27,
        batch_size=8,
        semi=True)

    losses = detector.forward(**inputs, return_loss=True)

    assert isinstance(losses, dict)
    assert 'proj_loss' in losses

    # Test forward test for semi-supervised learning
    with torch.no_grad():
        _ = detector.forward(**inputs, return_loss=False)
        _ = detector.forward_dummy(inputs['input'])

    # Test forward train for supervised learning with pose model and trajectory
    # model sharing one backbone
    model_cfg = dict(
        type='PoseLifter',
        pretrained=None,
        backbone=dict(type='TCN', in_channels=2 * 17),
        keypoint_head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss_keypoint=dict(type='MPJPELoss'),
            test_cfg=dict(restore_global_position=True)),
        traj_head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=1,
            loss_keypoint=dict(type='MPJPELoss'),
            is_trajectory=True),
        train_cfg=dict(),
        test_cfg=dict())

    cfg = mmcv.Config({'model': model_cfg})
    detector = build_posenet(cfg.model)

    detector.init_weights()

    inputs = _create_inputs(
        joint_num_in=17,
        joint_channel_in=2,
        joint_num_out=17,
        joint_channel_out=3,
        seq_len=27,
        batch_size=8,
        semi=True)

    losses = detector.forward(**inputs, return_loss=True)

    assert isinstance(losses, dict)
    assert 'traj_loss' in losses

    # Test forward test for semi-supervised learning with pose model and
    # trajectory model sharing one backbone
    with torch.no_grad():
        _ = detector.forward(**inputs, return_loss=False)
        _ = detector.forward_dummy(inputs['input'])
