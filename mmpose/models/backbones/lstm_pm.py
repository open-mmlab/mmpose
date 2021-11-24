# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,\
                      constant_init, normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck, get_expansion
from .utils import load_checkpoint


class Init_LSTM(nn.Module):
    """Initiate LSTM (Long Short-Term Memory).

    Args:
        out_channels (int):  Number of output channels. Default: 17.
        stem_channels (int): Number of channels of stem features. Default: 32.
        hidden_channels (int): Number of channels of hidden state. Default: 48.
    """

    def __init__(self,
                 out_channels=17,
                 stem_channels=32,
                 hidden_channels=48):

        self.conv_gx = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv_ix = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv_ox = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=out_channels + stem_channels + 1,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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


class LSTM(nn.Module):
    """LSTM (Long Short-Term Memory) for LSTM Pose Mechine.

    Args:
        out_channels (int):  Number of output channels. Default: 17.
        stem_channels (int): Number of channels of stem features. Default: 32.
        hidden_channels (int): Number of channels of hidden state. Default: 48.
    """

    def __init__(self,
                 out_channels=17,
                 stem_channels=32,
                 hidden_channels=48):

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

    def forward(self, heatmap, feature, centermap, hidden_t_1, cell_t_1):
        """Forward function."""
        x_t = torch.cat([heatmap, feature, centermap], dim=1)

        fx = self.conv_fx(x_t)
        fh = self.conv_fh(hidden_t_1)
        f_sum = fx + fh
        f_t = self.sigmoid(f_sum)

        ix = self.conv_ix(x_t)
        ih = self.conv_ih(hidden_t_1)
        i_sum = ix + ih
        i_t = self.sigmoid(i_sum)

        gx = self.conv_gx(x_t)
        gh = self.conv_gh(hidden_t_1)
        g_sum = gx + gh
        g_t = self.tanh(g_sum)

        ox = self.conv_ox(x_t)
        oh = self.conv_oh(hidden_t_1)
        o_sum = ox + oh
        o_t = self.sigmoid(o_sum)

        cell_t = f_t * cell_t_1 + i_t * g_t
        hidden_t = o_t * self.tanh(cell_t)

        return cell_t, hidden_t

@BACKBONES.register_module()
class LSTM_PM(nn.Module):
    """LSTM Pose Mechine backbone.

        `LSTM Pose Machines
        <https://arxiv.org/abs/1712.06316>`__

        Args:
            in_channels (int): Number of input image channels. Default: 3.
            out_channels (int):  Number of output channels. Default: 17.
            stem_channels (int): Number of channels of stem features. Default: 32.
            hidden_channels (int): Number of channels of hidden state. Default: 48.
            num_stages (int): Numerber of stages for propogation. Default: 9.
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

        self.convnet1 = self._make_convnet1(self.in_channels)
        self.convnet2 = self._make_convnet2(self.in_channels)
        self.convnet3 = self._make_convnet3()
        self.init_lstm = Init_LSTM(self.out_channels, self.stem_channels, self.hidden_channels)
        self.lstm = LSTM(self.out_channels, self.stem_channels, self.hidden_channels)

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

    def _make_convnet1(self, in_channels):
        """ConvNet1 for the initial image."""
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

        self.convnet1 = nn.Sequential(*layers)

    def _make_convnet2(self, in_channels):
        """ConvNet2 for feature extraction."""
        layers = self._make_stem_layers(in_channels)
        return nn.Sequential(*layers)

    def _make_convnet3(self):
        """ConvNet3 for output."""
        layers = []
        layers.append(
            ConvModule(
                self.hidden_channels,
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

    def stage1(self, image, cmap):
        """Forward function of the first stage."""
        initial_heatmap = self.convnet1(image)
        feature = self.convnet2(image)
        centermap = self.pool_centermap(cmap)

        x = torch.cat([initial_heatmap, feature, centermap], dim=1)
        cell_1, hidden_1 = self.init_lstm(x)
        heatmap = self.convnet3(hidden_1)
        return initial_heatmap, heatmap, cell_1, hidden_1

    def stage2(self, image, cmap, heatmap, cell_t_1, hidden_t_1):
        """Forward function of the propagation stages."""
        features = self.convnet2(image)
        centermap = self.pool_centermap(cmap)
        cell_t, hidden_t = self.lstm(heatmap, features, centermap, hidden_t_1, cell_t_1)
        current_heatmap = self.convnet3(hidden_t)
        return current_heatmap, cell_t, hidden_t

    def forward(self, images, centermap):
        """Forward function."""
        heatmaps = []

        image = images[:, :self.in_channels, :, :]
        initial_heatmap, heatmap, cell, hidden = self.stage1(image, centermap)
        heatmaps.append(initial_heatmap)
        heatmaps.append(heatmap)

        for i in range(1, self.num_stages):
            image = images[:, self.in_channels * i: self.in_channels * (i + 1), :, :]
            heatmap, cell, hidden = self.stage2(image, centermap, heatmap, cell, hidden)
            heat_maps.append(heatmap)
        return heatmaps
