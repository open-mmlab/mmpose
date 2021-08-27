# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models import TemporalRegressionHead


def test_temporal_regression_head():
    """Test temporal head."""

    # w/o global position restoration
    head = TemporalRegressionHead(
        in_channels=1024,
        num_joints=17,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        test_cfg=dict(restore_global_position=False))

    head.init_weights()

    with pytest.raises(AssertionError):
        # ndim of the input tensor should be 3
        input_shape = (1, 1024, 1, 1)
        inputs = _demo_inputs(input_shape)
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # size of the last dim should be 1
        input_shape = (1, 1024, 3)
        inputs = _demo_inputs(input_shape)
        _ = head(inputs)

    input_shape = (1, 1024, 1)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 17, 3])

    loss = head.get_loss(out, out, None)
    assert torch.allclose(loss['reg_loss'], torch.tensor(0.))

    _ = head.inference_model(inputs)
    _ = head.inference_model(inputs, [(0, 1), (2, 3)])
    metas = [{}]

    acc = head.get_accuracy(out, out, None, metas=metas)
    assert acc['mpjpe'] == 0.
    np.testing.assert_almost_equal(acc['p_mpjpe'], 0., decimal=6)

    # w/ global position restoration
    head = TemporalRegressionHead(
        in_channels=1024,
        num_joints=16,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        test_cfg=dict(restore_global_position=True))
    head.init_weights()

    input_shape = (1, 1024, 1)
    inputs = _demo_inputs(input_shape)
    metas = [{
        'root_position': np.zeros((1, 3)),
        'root_position_index': 0,
        'root_weight': 1.
    }]
    out = head(inputs)
    assert out.shape == torch.Size([1, 16, 3])

    inference_out = head.inference_model(inputs)
    acc = head.get_accuracy(out, out, torch.ones_like(out), metas)
    assert acc['mpjpe'] == 0.
    np.testing.assert_almost_equal(acc['p_mpjpe'], 0.)

    _ = head.decode(metas, inference_out)

    # trajectory model (only predict root position)
    head = TemporalRegressionHead(
        in_channels=1024,
        num_joints=1,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        is_trajectory=True,
        test_cfg=dict(restore_global_position=False))

    head.init_weights()

    input_shape = (1, 1024, 1)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 1, 3])

    loss = head.get_loss(out, out.squeeze(1), torch.ones_like(out))
    assert torch.allclose(loss['traj_loss'], torch.tensor(0.))


def _demo_inputs(input_shape=(1, 1024, 1)):
    """Create a superset of inputs needed to run head.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 1024, 1).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
