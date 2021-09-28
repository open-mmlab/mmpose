# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models import AEHigherResolutionHead, AESimpleHead


def test_ae_simple_head():
    """test bottom up AE simple head."""

    with pytest.raises(TypeError):
        # extra
        _ = AESimpleHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True],
            extra=[],
            loss_keypoint=dict(
                type='MultiLossFactory',
                num_joints=17,
                num_stages=1,
                ae_loss_type='exp',
                with_ae_loss=[True],
                push_loss_factor=[0.001],
                pull_loss_factor=[0.001],
                with_heatmaps_loss=[True],
                heatmaps_loss_factor=[1.0]))
    # test final_conv_kernel
    with pytest.raises(AssertionError):
        _ = AESimpleHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True],
            extra={'final_conv_kernel': -1},
            loss_keypoint=dict(
                type='MultiLossFactory',
                num_joints=17,
                num_stages=1,
                ae_loss_type='exp',
                with_ae_loss=[True],
                push_loss_factor=[0.001],
                pull_loss_factor=[0.001],
                with_heatmaps_loss=[True],
                heatmaps_loss_factor=[1.0]))
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 1},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    assert head.final_layer.padding == (0, 0)
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    assert head.final_layer.padding == (0, 0)
    # test with_ae_loss
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        with_ae_loss=[False],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    # test tag_per_joint
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[False],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 18, 32, 32])
    head = AESimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 18, 32, 32])


def test_ae_higherresolution_head():
    """test bottom up AE higherresolution head."""

    # test final_conv_kernel
    with pytest.raises(AssertionError):
        _ = AEHigherResolutionHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True, False],
            extra={'final_conv_kernel': 0},
            loss_keypoint=dict(
                type='MultiLossFactory',
                num_joints=17,
                num_stages=2,
                ae_loss_type='exp',
                with_ae_loss=[True, False],
                push_loss_factor=[0.001, 0.001],
                pull_loss_factor=[0.001, 0.001],
                with_heatmaps_loss=[True, True],
                heatmaps_loss_factor=[1.0, 1.0]))
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.final_layers[0].padding == (1, 1)
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 1},
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.final_layers[0].padding == (0, 0)
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.final_layers[0].padding == (0, 0)
    # test deconv layers
    with pytest.raises(ValueError):
        _ = AEHigherResolutionHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True, False],
            num_deconv_kernels=[1],
            cat_output=[True],
            loss_keypoint=dict(
                type='MultiLossFactory',
                num_joints=17,
                num_stages=2,
                ae_loss_type='exp',
                with_ae_loss=[True, False],
                push_loss_factor=[0.001, 0.001],
                pull_loss_factor=[0.001, 0.001],
                with_heatmaps_loss=[True, True],
                heatmaps_loss_factor=[1.0, 1.0]))
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[4],
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (0, 0)
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[3],
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (1, 1)
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[2],
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (0, 0)
    # test tag_per_joint & ae loss
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=False,
        with_ae_loss=[False, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[False, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    assert out[1].shape == torch.Size([1, 17, 64, 64])
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=False,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 18, 32, 32])
    assert out[1].shape == torch.Size([1, 17, 64, 64])
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, True],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])
    # cat_output
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, True],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])
    head = AEHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, True],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0]))
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
