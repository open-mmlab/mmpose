# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/mit-han-lab/litepose
# By Junyan Li, lijunyan668@outlook.com
import logging

import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .mobilenet_v2 import InvertedResidual
from .utils import load_checkpoint, make_divisible


class SepConv2d(nn.Module):
    """Separable Conv2d Layer.

    Args:
        in_channels (int): The input channels of the layer.
        out_channels (int): The output channels of the layer.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Default: 3.
        stride (int): Stride of the middle (first) 3x3 convolution.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(SepConv2d, self).__init__()
        conv = [
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None)
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        output = self.conv(x)
        return output


@BACKBONES.register_module()
class LitePose(BaseBackbone):
    """LitePose backbone.

    "Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation"
    More details can be found in the `paper
    <https://arxiv.org/abs/2205.01271>`__ .

    Args:
        num_blocks (list(int)): Searched block number for each stage.
        strides (list(int)): Searched stride config for each stage.
        channels (list(int)): Searched channel config for each stage.
        block_settings (list(list(int))): Searched block config for each block.
        input_channel (int): Output channel number for the first
            input stem stage.
        width_mult (float): width multiplier for each block.
            Default: 1.0.
        round_nearest (int): round to the nearest number.
            Default: 8.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
    """

    def __init__(self,
                 num_blocks,
                 strides,
                 channels,
                 block_settings,
                 input_channel,
                 width_mult=1.0,
                 round_nearest=8,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6')):
        assert len(num_blocks) == len(strides)
        assert len(num_blocks) == len(channels)
        assert len(num_blocks) == len(block_settings)
        super().__init__()
        # building first layer
        input_channel = make_divisible(input_channel * width_mult,
                                       round_nearest)
        self.first = nn.Sequential(
            ConvModule(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=32,
                out_channels=input_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(num_blocks)):
            n = num_blocks[id_stage]
            s = strides[id_stage]
            c = channels[id_stage]
            c = make_divisible(c * width_mult, round_nearest)
            block_setting = block_settings[id_stage]
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(
                    InvertedResidual(
                        input_channel,
                        c,
                        kernel_size=k,
                        stride=stride,
                        expand_ratio=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        return x_list
