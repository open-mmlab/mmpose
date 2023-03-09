# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models import (DeepposeRegressionHead, TopdownHeatmapMSMUHead,
                           TopdownHeatmapMultiStageHead,
                           TopdownHeatmapSimpleHead, ViPNASHeatmapSimpleHead)


def test_vipnas_simple_head():
    """Test simple head."""
    with pytest.raises(TypeError):
        # extra
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            extra=[],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(TypeError):
        head = ViPNASHeatmapSimpleHead(
            out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=-1,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    _ = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = ViPNASHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    # test final_conv_kernel
    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 1},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    assert head.final_layer.padding == (0, 0)
    _ = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 0},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )))
    assert len(head.final_layer) == 4

    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 256, 256])

    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 32, 32])

    head = ViPNASHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out.shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_top_down_simple_head():
    """Test simple head."""
    with pytest.raises(TypeError):
        # extra
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            extra=[],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(TypeError):
        head = TopdownHeatmapSimpleHead(
            out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=-1,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    _ = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopdownHeatmapSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    # test final_conv_kernel
    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 1},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    assert head.final_layer.padding == (0, 0)
    _ = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 0},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )))
    assert len(head.final_layer) == 4

    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 256, 256])

    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 32, 32])

    head = TopdownHeatmapSimpleHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out.shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_top_down_multistage_head():
    """Test multistage head."""
    with pytest.raises(TypeError):
        # the number of layers should match
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            extra=[],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=-1,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    _ = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    with pytest.raises(AssertionError):
        # inputs should be list
        head = TopdownHeatmapMultiStageHead(
            out_channels=3,
            in_channels=512,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
        input_shape = (1, 512, 32, 32)
        inputs = _demo_inputs(input_shape)
        out = head(inputs)

    # test final_conv_kernel
    head = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 3},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    head.init_weights()
    assert head.multi_final_layers[0].padding == (1, 1)
    head = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 1},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    assert head.multi_final_layers[0].padding == (0, 0)
    _ = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        extra={'final_conv_kernel': 0},
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    head = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert len(out) == 1
    assert out[0].shape == torch.Size([1, 3, 256, 256])

    head = TopdownHeatmapMultiStageHead(
        out_channels=3,
        in_channels=512,
        num_deconv_layers=0,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_top_down_msmu_head():
    """Test multi-stage multi-unit head."""
    with pytest.raises(AssertionError):
        # inputs should be list
        head = TopdownHeatmapMSMUHead(
            out_shape=(64, 48),
            unit_channels=256,
            num_stages=2,
            num_units=2,
            loss_keypoint=(
                [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
                [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # inputs should be list[list, ...]
        head = TopdownHeatmapMSMUHead(
            out_shape=(64, 48),
            unit_channels=256,
            num_stages=2,
            num_units=2,
            loss_keypoint=(
                [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
                [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [inputs] * 2
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # len(inputs) should equal to num_stages
        head = TopdownHeatmapMSMUHead(
            out_shape=(64, 48),
            unit_channels=256,
            num_stages=2,
            num_units=2,
            loss_keypoint=(
                [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
                [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 2] * 3
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # len(inputs[0]) should equal to num_units
        head = TopdownHeatmapMSMUHead(
            out_shape=(64, 48),
            unit_channels=256,
            num_stages=2,
            num_units=2,
            loss_keypoint=(
                [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
                [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 3] * 2
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # input channels should equal to param unit_channels
        head = TopdownHeatmapMSMUHead(
            out_shape=(64, 48),
            unit_channels=256,
            num_stages=2,
            num_units=2,
            loss_keypoint=(
                [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
                [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
        input_shape = (1, 128, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 2] * 2
        _ = head(inputs)

    head = TopdownHeatmapMSMUHead(
        out_shape=(64, 48),
        unit_channels=256,
        out_channels=17,
        num_stages=2,
        num_units=2,
        loss_keypoint=(
            [dict(type='JointsMSELoss', use_target_weight=True)] * 2 +
            [dict(type='JointsOHKMMSELoss', use_target_weight=True)]) * 2)
    input_shape = (1, 256, 32, 32)
    inputs = _demo_inputs(input_shape)
    inputs = [[inputs] * 2] * 2
    out = head(inputs)
    assert len(out) == 2 * 2
    assert out[0].shape == torch.Size([1, 17, 64, 48])

    head.init_weights()


def test_fc_head():
    """Test fc head."""
    head = DeepposeRegressionHead(
        in_channels=2048,
        num_joints=17,
        loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True))

    head.init_weights()

    input_shape = (1, 2048)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 17, 2])

    loss = head.get_loss(out, out, torch.ones_like(out))
    assert torch.allclose(loss['reg_loss'], torch.tensor(0.))

    _ = head.inference_model(inputs)
    _ = head.inference_model(inputs, [])

    acc = head.get_accuracy(out, out, torch.ones_like(out))
    assert acc['acc_pose'] == 1.

    # Test fc head with out_sigma set to True(Default False)
    head = DeepposeRegressionHead(
        in_channels=2048,
        num_joints=17,
        out_sigma=True,
        loss_keypoint=dict(type='RLELoss', use_target_weight=True))

    head.init_weights()

    input_shape = (1, 2048)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 17, 4])

    img_metas = [dict(center=np.zeros(2), scale=np.zeros(2), image_file='')]
    result = head.decode(
        img_metas, out.detach().cpu().numpy(), img_size=(64, 64))
    assert 'preds' in result and result['preds'].shape == (1, 17, 3)
    assert 'boxes' in result and result['boxes'].shape == (1, 6)

    target = out[:, :, 0:2]

    _ = head.get_loss(out, target, torch.ones_like(target))
    _ = head.inference_model(inputs)
    _ = head.inference_model(inputs, [])

    acc = head.get_accuracy(out, target, torch.ones_like(target))
    assert acc['acc_pose'] == 1.


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
