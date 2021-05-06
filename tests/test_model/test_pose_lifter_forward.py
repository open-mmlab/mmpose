import mmcv
import numpy as np
import torch

from mmpose.models import build_posenet


def _create_inputs(joint_num_in, joint_channel_in, joint_num_out,
                   joint_channel_out, seq_len, batch_size):

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

    return inputs


def test_pose_lifter_forward():
    # Test forward train
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

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(
            inputs['input'],
            inputs['target'],
            inputs['target_weight'],
            inputs['metas'],
            return_loss=False)
        _ = detector.forward_dummy(inputs['input'])
