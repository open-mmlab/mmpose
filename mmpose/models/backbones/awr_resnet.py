# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class AWRResNet(ResNet):
    """AWR ResNet backbone.

    Using a specialized stem scheme.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=False, **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for depth."""
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
