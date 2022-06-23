# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class convbnrelu(nn.Sequential):

    def __init__(self, inp, oup, ker=3, stride=1, groups=1):
        super(convbnrelu, self).__init__(
            nn.Conv2d(
                inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneck, self).__init__()
        feature_dim = _make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim), nn.ReLU6(inplace=True))
        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                feature_dim,
                ker,
                stride,
                ker // 2,
                groups=feature_dim,
                bias=False), nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace=True))
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes))
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes

    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out


class SepConv2d(nn.Module):

    def __init__(self, inp, oup, ker=3, stride=1):
        super(SepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
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
    """

    def __init__(self,
                 num_blocks,
                 strides,
                 channels,
                 block_settings,
                 input_channel,
                 width_mult=1.0,
                 round_nearest=8):
        assert len(num_blocks) == len(strides)
        assert len(num_blocks) == len(channels)
        assert len(num_blocks) == len(block_settings)
        super().__init__()
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel))
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(num_blocks)):
            n = num_blocks[id_stage]
            s = strides[id_stage]
            c = channels[id_stage]
            c = _make_divisible(c * width_mult, round_nearest)
            block_setting = block_settings[id_stage]
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(
                    InvBottleneck(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)

    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        return x_list
