import pytest
import torch

from mmpose.models.backbones import TCN
from mmpose.models.backbones.tcn import BasicTemporalBlock


def test_basic_temporal_block():
    with pytest.raises(AssertionError):
        # padding( + shift) should not be larger than x.shape[2]
        block = BasicTemporalBlock(1024, 1024, dilation=81)
        x = torch.rand(2, 1024, 150)
        x_out = block(x)

    with pytest.raises(AssertionError):
        # when train_with_stride_conv is True, shift + kernel_size // 2 should
        # not be larger than x.shape[2]
        block = BasicTemporalBlock(
            1024,
            1024,
            kernel_size=5,
            causal=True,
            train_with_stride_conv=True)
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

    # BasicTemporalBlock, train_with_stride_conv == True
    block = BasicTemporalBlock(1024, 1024, train_with_stride_conv=True)
    x = torch.rand(2, 1024, 81)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 1024, 27])

    # BasicTemporalBlock with train_with_stride_conv == True and causal == True
    block = BasicTemporalBlock(
        1024, 1024, train_with_stride_conv=True, causal=True)
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

    # Test TCN with 2 blocks (train_with_stride_conv == False)
    model = TCN(in_channels=34, num_blocks=2, kernel_sizes=(3, 3, 3))
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 2
    assert feat[0].shape == (2, 1024, 235)
    assert feat[1].shape == (2, 1024, 217)

    # Test TCN with 4 blocks (train_with_stride_conv == False)
    model = TCN(in_channels=34, num_blocks=4, kernel_sizes=(3, 3, 3, 3, 3))
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 4
    assert feat[0].shape == (2, 1024, 235)
    assert feat[1].shape == (2, 1024, 217)
    assert feat[2].shape == (2, 1024, 163)
    assert feat[3].shape == (2, 1024, 1)

    # Test TCN with 4 blocks (train_with_stride_conv == True)
    model = TCN(
        in_channels=34,
        num_blocks=4,
        kernel_sizes=(3, 3, 3, 3, 3),
        train_with_stride_conv=True)
    pose2d = torch.rand((2, 34, 243))
    feat = model(pose2d)
    assert len(feat) == 4
    assert feat[0].shape == (2, 1024, 27)
    assert feat[1].shape == (2, 1024, 9)
    assert feat[2].shape == (2, 1024, 3)
    assert feat[3].shape == (2, 1024, 1)
