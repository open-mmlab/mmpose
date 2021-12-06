# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones import HRNet
from mmpose.models.backbones.hrnet import HRModule
from mmpose.models.backbones.resnet import BasicBlock, Bottleneck


def is_block(modules):
    """Check if is HRModule building block."""
    if isinstance(modules, (HRModule, )):
        return True
    return False


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (_BatchNorm, )):
        return True
    return False


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def test_hrmodule():
    # Test HRModule forward
    block = HRModule(
        num_branches=1,
        blocks=BasicBlock,
        num_blocks=(4, ),
        in_channels=[
            64,
        ],
        num_channels=(64, ))

    x = torch.randn(2, 64, 56, 56)
    x_out = block([x])
    assert x_out[0].shape == torch.Size([2, 64, 56, 56])


def test_hrnet_backbone():
    extra = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256)))

    model = HRNet(extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([2, 32, 56, 56])

    # Test HRNet zero initialization of residual
    model = HRNet(extra, in_channels=3, zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert all_zeros(m.norm3)
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([2, 32, 56, 56])

    # Test HRNet with the first three stages frozen
    frozen_stages = 3
    model = HRNet(extra, in_channels=3, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    if frozen_stages >= 0:
        assert model.norm1.training is False
        assert model.norm2.training is False
        for layer in [model.conv1, model.norm1, model.conv2, model.norm2]:
            for param in layer.parameters():
                assert param.requires_grad is False

    for i in range(1, frozen_stages + 1):
        if i == 1:
            layer = getattr(model, 'layer1')
        else:
            layer = getattr(model, f'stage{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

        if i < 4:
            layer = getattr(model, f'transition{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
