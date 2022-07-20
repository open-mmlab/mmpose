# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmpose.models.backbones import TCN
from mmpose.models.backbones.tcn import BasicTemporalBlock


def test_basic_temporal_block():
    with pytest.raises(AssertionError):
        # padding( + shift) should not be larger than x.shape[2]
        block = BasicTemporalBlock(1024, 1024, dilation=81)
        x = torch.rand(2, 1024, 150)
        x_out = block(x)

    with pytest.raises(AssertionError):
        # when use_stride_conv is True, shift + kernel_size // 2 should
        # not be larger than x.shape[2]
        block = BasicTemporalBlock(
            1024, 1024, kernel_size=5, causal=True, use_stride_conv=True)
        x = torch.rand(2, 1024, 3)
        x_out = block(x)

    # BasicTemporalBlock with causal == False
    block = BasicTemporalBlock(1024, 1024)
    x = torch.rand(2, 1024, 241)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 235])

    # BasicTemporalBlock with causal == True
    block = BasicTemporalBlock(1024, 1024, causal=True)
    x = torch.rand(2, 1024, 241)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 235])

    # BasicTemporalBlock with residual == False
    block = BasicTemporalBlock(1024, 1024, residual=False)
    x = torch.rand(2, 1024, 241)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 235])

    # BasicTemporalBlock, use_stride_conv == True
    block = BasicTemporalBlock(1024, 1024, use_stride_conv=True)
    x = torch.rand(2, 1024, 81)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 27])

    # BasicTemporalBlock with use_stride_conv == True and causal == True
    block = BasicTemporalBlock(1024, 1024, use_stride_conv=True, causal=True)
    x = torch.rand(2, 1024, 81)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 27])


def test_tcn_backbone():
    with pytest.raises(AssertionError):
        # num_blocks should equal len(kernel_sizes) - 1
        TCN(in_channels=34, num_blocks=3, kernel_sizes=(3, 3, 3))

    with pytest.raises(AssertionError):
        # kernel size should be odd
        TCN(in_channels=34, kernel_sizes=(3, 4, 3))

    # Test TCN with 2 blocks (use_stride_conv == False)
    model = TCN(in_channels=34, num_blocks=2, kernel_sizes=(3, 3, 3))
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 2
    assert feat[0].shape == (2, 1024, 235)
    assert feat[1].shape == (2, 1024, 217)

    # Test TCN with 4 blocks and weight norm clip
    max_norm = 0.1
    model = TCN(
        in_channels=34,
        num_blocks=4,
        kernel_sizes=(3, 3, 3, 3, 3),
        max_norm=max_norm)
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 4
    assert feat[0].shape == (2, 1024, 235)
    assert feat[1].shape == (2, 1024, 217)
    assert feat[2].shape == (2, 1024, 163)
    assert feat[3].shape == (2, 1024, 1)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            norm = module.weight.norm().item()
            np.testing.assert_allclose(
                np.maximum(norm, max_norm), max_norm, rtol=1e-4)

    # Test TCN with 4 blocks (use_stride_conv == True)
    model = TCN(
        in_channels=34,
        num_blocks=4,
        kernel_sizes=(3, 3, 3, 3, 3),
        use_stride_conv=True)
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 4
    assert feat[0].shape == (2, 1024, 27)
    assert feat[1].shape == (2, 1024, 9)
    assert feat[2].shape == (2, 1024, 3)
    assert feat[3].shape == (2, 1024, 1)

    # Check that the model w. or w/o use_stride_conv will have the same
    # output and gradient after a forward+backward pass
    model1 = TCN(
        in_channels=34,
        stem_channels=4,
        num_blocks=1,
        kernel_sizes=(3, 3),
        dropout=0,
        residual=False,
        norm_cfg=None)
    model2 = TCN(
        in_channels=34,
        stem_channels=4,
        num_blocks=1,
        kernel_sizes=(3, 3),
        dropout=0,
        residual=False,
        norm_cfg=None,
        use_stride_conv=True)
    for m in model1.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.constant_(m.weight, 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    for m in model2.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.constant_(m.weight, 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    input1 = torch.rand((1, 34, 9))
    input2 = input1.clone()
    outputs1 = model1(input1)
    outputs2 = model2(input2)
    for output1, output2 in zip(outputs1, outputs2):
        assert torch.isclose(output1, output2).all()

    criterion = nn.MSELoss()
    target = torch.rand(output1.shape)
    loss1 = criterion(output1, target)
    loss2 = criterion(output2, target)
    loss1.backward()
    loss2.backward()
    for m1, m2 in zip(model1.modules(), model2.modules()):
        if isinstance(m1, nn.Conv1d):
            assert torch.isclose(m1.weight.grad, m2.weight.grad).all()
