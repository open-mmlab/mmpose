# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones.utils import (InvertedResidual, SELayer,
                                           channel_shuffle, make_divisible)


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def test_make_divisible():
    # test min_value is None
    result = make_divisible(34, 8, None)
    assert result == 32

    # test when new_value > min_ratio * value
    result = make_divisible(10, 8, min_ratio=0.9)
    assert result == 16

    # test min_value = 0.8
    result = make_divisible(33, 8, min_ratio=0.8)
    assert result == 32


def test_channel_shuffle():
    x = torch.randn(1, 24, 56, 56)
    with pytest.raises(AssertionError):
        # num_channels should be divisible by groups
        channel_shuffle(x, 7)

    groups = 3
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    out = channel_shuffle(x, groups)
    # test the output value when groups = 3
    for b in range(batch_size):
        for c in range(num_channels):
            c_out = c % channels_per_group * groups + c // channels_per_group
            for i in range(height):
                for j in range(width):
                    assert x[b, c, i, j] == out[b, c_out, i, j]


def test_inverted_residual():

    with pytest.raises(AssertionError):
        # stride must be in [1, 2]
        InvertedResidual(16, 16, 32, stride=3)

    with pytest.raises(AssertionError):
        # se_cfg must be None or dict
        InvertedResidual(16, 16, 32, se_cfg=list())

    with pytest.raises(AssertionError):
        # in_channeld and out_channels must be the same if
        # with_expand_conv is False
        InvertedResidual(16, 16, 32, with_expand_conv=False)

    # Test InvertedResidual forward, stride=1
    block = InvertedResidual(16, 16, 32, stride=1)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert getattr(block, 'se', None) is None
    assert block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward, stride=2
    block = InvertedResidual(16, 16, 32, stride=2)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert not block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 28, 28))

    # Test InvertedResidual forward with se layer
    se_cfg = dict(channels=32)
    block = InvertedResidual(16, 16, 32, stride=1, se_cfg=se_cfg)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert isinstance(block.se, SELayer)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward, with_expand_conv=False
    block = InvertedResidual(32, 16, 32, with_expand_conv=False)
    x = torch.randn(1, 32, 56, 56)
    x_out = block(x)
    assert getattr(block, 'expand_conv', None) is None
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with GroupNorm
    block = InvertedResidual(
        16, 16, 32, norm_cfg=dict(type='GN', num_groups=2))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    for m in block.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with HSigmoid
    block = InvertedResidual(16, 16, 32, act_cfg=dict(type='HSigmoid'))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with checkpoint
    block = InvertedResidual(16, 16, 32, with_cp=True)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert block.with_cp
    assert x_out.shape == torch.Size((1, 16, 56, 56))
