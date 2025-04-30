# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class CpmBlock(BaseModule):
    """CpmBlock for Convolutional Pose Machine.

    Args:
        in_channels (int): Input channels of this block.
        channels (list): Output channels of each conv module.
        kernels (list): Kernel sizes of each conv module.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 channels=(128, 128, 128),
                 kernels=(11, 11, 11),
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert len(channels) == len(kernels)
        layers = []
        for i in range(len(channels)):
            if i == 0:
                input_channels = in_channels
            else:
                input_channels = channels[i - 1]
            layers.append(
                ConvModule(
                    input_channels,
                    channels[i],
                    kernels[i],
                    padding=(kernels[i] - 1) // 2,
                    norm_cfg=norm_cfg))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Model forward function."""
        out = self.model(x)
        return out


@MODELS.register_module()
class CPM(BaseBackbone):
    """CPM backbone.

    Convolutional Pose Machines.
    More details can be found in the `paper
    <https://arxiv.org/abs/1602.00134>`__ .

    Args:
        in_channels (int): The input channels of the CPM.
        out_channels (int): The output channels of the CPM.
        feat_channels (int): Feature channel of each CPM stage.
        middle_channels (int): Feature channel of conv after the middle stage.
        num_stages (int): Number of stages.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='Normal', std=0.001, layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]``

    Example:
        >>> from mmpose.models import CPM
        >>> import torch
        >>> self = CPM(3, 17)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        feat_channels=128,
        middle_channels=32,
        num_stages=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
    ):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)

        assert in_channels == 3

        self.num_stages = num_stages
        assert self.num_stages >= 1

        self.stem = nn.Sequential(
            ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 32, 5, padding=2, norm_cfg=norm_cfg),
            ConvModule(32, 512, 9, padding=4, norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, norm_cfg=norm_cfg),
            ConvModule(512, out_channels, 1, padding=0, act_cfg=None))

        self.middle = nn.Sequential(
            ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.cpm_stages = nn.ModuleList([
            CpmBlock(
                middle_channels + out_channels,
                channels=[feat_channels, feat_channels, feat_channels],
                kernels=[11, 11, 11],
                norm_cfg=norm_cfg) for _ in range(num_stages - 1)
        ])

        self.middle_conv = nn.ModuleList([
            nn.Sequential(
                ConvModule(
                    128, middle_channels, 5, padding=2, norm_cfg=norm_cfg))
            for _ in range(num_stages - 1)
        ])

        self.out_convs = nn.ModuleList([
            nn.Sequential(
                ConvModule(
                    feat_channels,
                    feat_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg),
                ConvModule(feat_channels, out_channels, 1, act_cfg=None))
            for _ in range(num_stages - 1)
        ])

    def forward(self, x):
        """Model forward function."""
        stage1_out = self.stem(x)
        middle_out = self.middle(x)
        out_feats = []

        out_feats.append(stage1_out)

        for ind in range(self.num_stages - 1):
            single_stage = self.cpm_stages[ind]
            out_conv = self.out_convs[ind]

            inp_feat = torch.cat(
                [out_feats[-1], self.middle_conv[ind](middle_out)], 1)
            cpm_feat = single_stage(inp_feat)
            out_feat = out_conv(cpm_feat)
            out_feats.append(out_feat)

        return out_feats
