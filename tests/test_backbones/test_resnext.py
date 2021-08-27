# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpose.models.backbones import ResNeXt
from mmpose.models.backbones.resnext import Bottleneck as BottleneckX


def test_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckX(64, 64, groups=32, width_per_group=4, style='tensorflow')

    # Test ResNeXt Bottleneck structure
    block = BottleneckX(
        64, 256, groups=32, width_per_group=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    # Test ResNeXt Bottleneck forward
    block = BottleneckX(64, 64, base_channels=16, groups=32, width_per_group=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnext():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

    # Test ResNeXt with group 32, width_per_group 4
    model = ResNeXt(
        depth=50, groups=32, width_per_group=4, out_indices=(0, 1, 2, 3))
    for m in model.modules():
        if isinstance(m, BottleneckX):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNeXt with group 32, width_per_group 4 and layers 3 out forward
    model = ResNeXt(depth=50, groups=32, width_per_group=4, out_indices=(3, ))
    for m in model.modules():
        if isinstance(m, BottleneckX):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 2048, 7, 7])
