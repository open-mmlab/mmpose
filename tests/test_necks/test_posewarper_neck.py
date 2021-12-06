# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models.necks import PoseWarperNeck


def test_posewarper_neck():
    """Test PoseWarperNeck."""
    with pytest.raises(AssertionError):
        # test value of trans_conv_kernel
        _ = PoseWarperNeck(
            out_channels=3,
            in_channels=512,
            inner_channels=128,
            trans_conv_kernel=2)

    with pytest.raises(TypeError):
        # test type of res_blocks_cfg
        _ = PoseWarperNeck(
            out_channels=3,
            in_channels=512,
            inner_channels=128,
            res_blocks_cfg=2)

    with pytest.raises(AssertionError):
        # test value of dilations
        neck = PoseWarperNeck(
            out_channels=3, in_channels=512, inner_channels=128, dilations=[])

    in_channels = 48
    out_channels = 17
    inner_channels = 128

    neck = PoseWarperNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        inner_channels=inner_channels)

    with pytest.raises(TypeError):
        # the forward require two arguments: inputs and frame_weight
        _ = neck(1)

    with pytest.raises(AssertionError):
        # the inputs to PoseWarperNeck must be list or tuple
        _ = neck(1, [0.1])

    # test the case when num_frames * batch_size if larger than
    # the default value of 'im2col_step' but can not be divided
    # by it in mmcv.ops.deform_conv
    b_0 = 8  # batch_size
    b_1 = 16
    h_0 = 4  # image height
    h_1 = 2

    num_frame_0 = 2
    num_frame_1 = 5

    # test input format
    # B, C, H, W
    x0_shape = (b_0, in_channels, h_0, h_0)
    x1_shape = (b_1, in_channels, h_1, h_1)

    # test concat_tensors case
    # at the same time, features output from backbone like ResNet is Tensors
    x0_shape = (b_0 * num_frame_0, in_channels, h_0, h_0)
    x0 = _demo_inputs(x0_shape, length=1)
    frame_weight_0 = np.random.uniform(0, 1, num_frame_0)

    # test forward
    y = neck(x0, frame_weight_0)
    assert y.shape == torch.Size([b_0, out_channels, h_0, h_0])

    # test concat_tensors case
    # this time, features output from backbone like HRNet
    # is list of Tensors rather than Tensors
    x0_shape = (b_0 * num_frame_0, in_channels, h_0, h_0)
    x0 = _demo_inputs(x0_shape, length=2)
    x0 = [x0]
    frame_weight_0 = np.random.uniform(0, 1, num_frame_0)

    # test forward
    y = neck(x0, frame_weight_0)
    assert y.shape == torch.Size([b_0, out_channels, h_0, h_0])

    # test not concat_tensors case
    # at the same time, features output from backbone like ResNet is Tensors
    x1_shape = (b_1, in_channels, h_1, h_1)
    x1 = _demo_inputs(x1_shape, length=num_frame_1)
    frame_weight_1 = np.random.uniform(0, 1, num_frame_1)

    # test forward
    y = neck(x1, frame_weight_1)
    assert y.shape == torch.Size([b_1, out_channels, h_1, h_1])

    # test not concat_tensors case
    # this time, features output from backbone like HRNet
    # is list of Tensors rather than Tensors
    x1_shape = (b_1, in_channels, h_1, h_1)
    x1 = _demo_inputs(x1_shape, length=2)
    x1 = [x1 for _ in range(num_frame_1)]
    frame_weight_1 = np.random.uniform(0, 1, num_frame_1)

    # test forward
    y = neck(x1, frame_weight_1)
    assert y.shape == torch.Size([b_1, out_channels, h_1, h_1])

    # test special case that when in concat_tensors case,
    # batch_size * num_frames is larger than the default value
    # 'im2col_step' in mmcv.ops.deform_conv, but can not be divided by it
    # see https://github.com/open-mmlab/mmcv/issues/1440
    x1_shape = (b_1 * num_frame_1, in_channels, h_1, h_1)
    x1 = _demo_inputs(x1_shape, length=2)
    x1 = [x1]
    frame_weight_0 = np.random.uniform(0, 1, num_frame_1)

    y = neck(x1, frame_weight_1)
    assert y.shape == torch.Size([b_1, out_channels, h_1, h_1])

    # test the inappropriate value of `im2col_step`
    neck = PoseWarperNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        inner_channels=inner_channels,
        im2col_step=32)
    with pytest.raises(AssertionError):
        _ = neck(x1, frame_weight_1)


def _demo_inputs(input_shape=(80, 48, 4, 4), length=1):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
        length (int): the length of output list
        nested (bool): whether the output Tensor is double-nested list.
    """
    imgs = [
        torch.FloatTensor(np.random.random(input_shape)) for _ in range(length)
    ]
    return imgs
