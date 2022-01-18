# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class LSTM(nn.Module):
    """LSTM (Long Short-Term Memory) for LSTM Pose Machine.

    Args:
        out_channels (int):  Number of output channels. Default: 17.
        stem_channels (int): Number of channels of stem features. Default: 32.
        hidden_channels (int): Number of channels of hidden state. Default: 48.
    """

    def __init__(self, out_channels=17, stem_channels=32, hidden_channels=48):

        self.conv_fx = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.conv_fh = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.conv_ix = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.conv_ih = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.conv_gx = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.conv_gh = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.conv_ox = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.conv_oh = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def init_forward(self, x):
        """Forward function."""
        gx = self.conv_gx(x)
        ix = self.conv_ix(x)
        ox = self.conv_ox(x)

        gx = self.tanh(gx)
        ix = self.sigmoid(ix)
        ox = self.sigmoid(ox)

        cell_1 = self.tanh(gx * ix)
        hidden_1 = ox * cell_1

        return cell_1, hidden_1

    def forward(self, heatmap, feature, centermap, hidden_t, cell_t):
        """Forward function."""
        x_t = torch.cat([heatmap, feature, centermap], dim=1)

        fx = self.conv_fx(x_t)
        fh = self.conv_fh(hidden_t)
        f_sum = fx + fh
        f_t = self.sigmoid(f_sum)

        ix = self.conv_ix(x_t)
        ih = self.conv_ih(hidden_t)
        i_sum = ix + ih
        i_t = self.sigmoid(i_sum)

        gx = self.conv_gx(x_t)
        gh = self.conv_gh(hidden_t)
        g_sum = gx + gh
        g_t = self.tanh(g_sum)

        ox = self.conv_ox(x_t)
        oh = self.conv_oh(hidden_t)
        o_sum = ox + oh
        o_t = self.sigmoid(o_sum)

        cell_t = f_t * cell_t + i_t * g_t
        hidden_t = o_t * self.tanh(cell_t)

        return cell_t, hidden_t


@BACKBONES.register_module()
class LSTM_PM(BaseBackbone):
    """LSTM Pose Machine backbone.

    `LSTM Pose Machines
    <https://arxiv.org/abs/1712.06316>`__

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        out_channels (int):  Number of output channels. Default: 17.
        stem_channels (int): Number of channels of stem features. Default: 32.
        hidden_channels (int): Number of channels of hidden state. Default: 48.
        num_stages (int): Numerber of stages for propagation. Default: 9.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict | None): The config dict for norm layers. Default: None.

    TODO: check it after the format of inputs is decided.
    Example:
        >>> from mmpose.models import LSTM_PM
        >>> import torch
        >>> self = LSTM_PM(num_stages=3)
        >>> self.eval()
        >>> images = torch.rand(1, 21, 368, 368)
        >>> centermap = torch.rand(1, 1, 368, 368)
        >>> heatmaps = self.forward(images, centermap)
        >>> for heatmap in heatmaps:
        ...     print(tuple(heatmap.shape))
        (1, 32, 46, 46)
        (1, 32, 46, 46)
        (1, 32, 46, 46)
        (1, 32, 46, 46)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=17,
                 stem_channels=32,
                 hidden_channels=48,
                 num_stages=7,
                 conv_cfg=None,
                 norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_channels = stem_channels
        self.hidden_channels = hidden_channels
        self.num_stages = num_stages

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = self._make_conv1(self.in_channels)
        self.conv2 = self._make_conv2(self.in_channels)
        self.conv3 = self._make_conv3(self.hidden_channels)
        self.lstm = LSTM(self.out_channels, self.stem_channels,
                         self.hidden_channels)

        # TODO: May be generated in dataset as the last channel of target
        self.pool_centermap = nn.AvgPool2d(kernel_size=9, stride=8)

    def _make_stem_layers(self, in_channels):
        """Make stem layers."""
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                128,
                kernel_size=9,
                stride=1,
                padding=4,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(
            ConvModule(
                128,
                128,
                kernel_size=9,
                stride=1,
                padding=4,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(
            ConvModule(
                128,
                128,
                kernel_size=9,
                stride=1,
                padding=4,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(
            ConvModule(
                128,
                self.stem_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))

        return layers

    def _make_conv1(self, in_channels):
        """Make conv1 for the initial image."""
        layers = self._make_stem_layers(in_channels)
        layers.append(
            ConvModule(
                32,
                512,
                kernel_size=9,
                stride=1,
                padding=4,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            ConvModule(
                512,
                512,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=512,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0))

        self.conv1 = nn.Sequential(*layers)

    def _make_conv2(self, in_channels):
        """Make conv2 for feature extraction."""
        layers = self._make_stem_layers(in_channels)
        return nn.Sequential(*layers)

    def _make_conv3(self, in_channels):
        """Make conv3 for output."""
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                128,
                kernel_size=11,
                stride=1,
                padding=5,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            ConvModule(
                128,
                128,
                kernel_size=11,
                stride=1,
                padding=5,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            ConvModule(
                128,
                128,
                kernel_size=11,
                stride=1,
                padding=5,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            ConvModule(
                128,
                128,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=128,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0))

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, images, centermap):
        """Forward function."""
        heatmaps = []

        image = images[:, :self.in_channels, :, :]
        # Stage1
        initial_heatmap = self.conv1(image)
        feature = self.conv2(image)
        centermap = self.pool_centermap(centermap)

        x = torch.cat([initial_heatmap, feature, centermap], dim=1)
        cell, hidden = self.lstm.init_forward(x)
        heatmap = self.conv3(hidden)

        heatmaps.append(initial_heatmap)
        heatmaps.append(heatmap)

        for i in range(1, self.num_stages):
            image = images[:, self.in_channels * i:self.in_channels *
                           (i + 1), :, :]
            features = self.conv2(image)
            centermap = self.pool_centermap(centermap)
            cell, hidden = self.lstm(heatmap, features, centermap, hidden,
                                     cell)
            heatmap = self.conv3(hidden)

            heatmaps.append(heatmap)
        return heatmaps
