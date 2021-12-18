# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models.backbones import HRFormer
from mmpose.models.backbones.hrformer import (HRTransformerModule,
                                              LocalWindowTransformerBlock)
from mmpose.models.backbones.resnet import Bottleneck


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
    # Test HighResolutionTransformerModule forward
    module = HRTransformerModule(
        num_branches=2,
        blocks=LocalWindowTransformerBlock,
        num_blocks=(2, 2),
        in_channels=[32, 64],
        num_channels=(32, 64),
        num_heads=[1, 2],
        num_mlp_ratios=[4, 4],
        num_window_sizes=[7, 7],
        multiscale_output=True,
        with_cp=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_cfg=None,
        drop_paths=[0, 0])

    x = [torch.randn(2, 32, 56, 56), torch.randn(2, 64, 28, 28)]
    x_out = module(x)
    assert x_out[0].shape == torch.Size([2, 32, 56, 56])
    assert x_out[1].shape == torch.Size([2, 64, 28, 28])


def test_hrformer_backbone():
    extra = dict(
        drop_path_rate=0.1,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(2, ),
            num_channels=(32, ),
            num_heads=[2],
            num_mlp_ratios=[4]),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='TRANSFORMER_BLOCK',
            num_blocks=(2, 2),
            num_channels=(32, 64),
            num_heads=[1, 2],
            num_mlp_ratios=[4, 4],
            num_window_sizes=[7, 7]),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='TRANSFORMER_BLOCK',
            num_blocks=(2, 2, 2),
            num_channels=(32, 64, 128),
            num_heads=[1, 2, 4],
            num_mlp_ratios=[4, 4, 4],
            num_window_sizes=[7, 7, 7]),
        stage4=dict(
            num_modules=2,
            num_branches=4,
            block='TRANSFORMER_BLOCK',
            num_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 128, 256),
            num_heads=[1, 2, 4, 8],
            num_mlp_ratios=[4, 4, 4, 4],
            num_window_sizes=[7, 7, 7, 7]))

    model = HRFormer(extra=extra, in_channels=3)

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([2, 32, 56, 56])

    # Test HRNet zero initialization of residual
    model = HRFormer(extra, in_channels=3, zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert all_zeros(m.norm3)
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([2, 32, 56, 56])
