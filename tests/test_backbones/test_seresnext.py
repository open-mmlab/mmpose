# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpose.models.backbones import SEResNeXt
from mmpose.models.backbones.seresnext import SEBottleneck as SEBottleneckX


def test_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        SEBottleneckX(64, 64, groups=32, width_per_group=4, style='tensorflow')

    # Test SEResNeXt Bottleneck structure
    block = SEBottleneckX(
        64, 256, groups=32, width_per_group=4, stride=2, style='pytorch')
    assert block.width_per_group == 4
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128
    assert block.conv2.out_channels == block.mid_channels

    # Test SEResNeXt Bottleneck structure (groups=1)
    block = SEBottleneckX(
        64, 256, groups=1, width_per_group=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 1
    assert block.conv2.out_channels == 64
    assert block.mid_channels == 64
    assert block.conv2.out_channels == block.mid_channels

    # Test SEResNeXt Bottleneck forward
    block = SEBottleneckX(
        64, 64, base_channels=16, groups=32, width_per_group=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_seresnext():
    with pytest.raises(KeyError):
        # SEResNeXt depth should be in [50, 101, 152]
        SEResNeXt(depth=18)

    # Test SEResNeXt with group 32, width_per_group 4
    model = SEResNeXt(
        depth=50, groups=32, width_per_group=4, out_indices=(0, 1, 2, 3))
    for m in model.modules():
        if isinstance(m, SEBottleneckX):
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

    # Test SEResNeXt with group 32, width_per_group 4 and layers 3 out forward
    model = SEResNeXt(
        depth=50, groups=32, width_per_group=4, out_indices=(3, ))
    for m in model.modules():
        if isinstance(m, SEBottleneckX):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 2048, 7, 7])
