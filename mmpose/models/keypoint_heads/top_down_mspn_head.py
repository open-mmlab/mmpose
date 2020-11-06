import copy as cp

import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, Linear,
                      build_activation_layer, build_norm_layer, constant_init,
                      kaiming_init, normal_init)

from ..registry import HEADS


class PredictHeatmap(nn.Module):
    """Predict the heat map for an input feature.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        use_prm (bool): Whether to use pose refine machine. Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 unit_channels,
                 out_channels,
                 out_shape,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.use_prm = use_prm
        if use_prm:
            self.prm = PRM(out_channels, norm_cfg=norm_cfg)
        self.conv_layers = nn.Sequential(
            ConvModule(
                unit_channels,
                unit_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            ConvModule(
                unit_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False))

    def forward(self, feature):
        feature = self.conv_layers(feature)
        output = nn.functional.interpolate(
            feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class PRM(nn.Module):
    """Pose Refine Machine.

    For more details about PRM, refer to Learning Delicate
    Local Representations for Multi-Person Pose Estimation (ECCV 2020).
    Args:
        out_channels (int): Channel number of the output. Equals to
            the number of key points.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, out_channels, norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.out_channels = out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_path = nn.Sequential(
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            build_activation_layer(dict(type='Sigmoid')))

        self.bottom_path = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            DepthwiseSeparableConvModule(
                self.out_channels,
                1,
                kernel_size=9,
                stride=1,
                padding=4,
                norm_cfg=norm_cfg,
                inplace=False), build_activation_layer(dict(type='Sigmoid')))
        self.conv_bn_relu_prm_1 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False)

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out

        out_2 = self.global_pooling(out_1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_2 = self.middle_path(out_2)
        out_2 = out_2.unsqueeze(2)
        out_2 = out_2.unsqueeze(3)

        out_3 = self.bottom_path(out_1)
        out = out_1 * (1 + out_2 * out_3)

        return out


@HEADS.register_module()
class TopDownMSPNHead(nn.Module):
    """Heads for multi-stage Residual Steps Networks.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        num_stages (int): Number of stages.
        num_units (int): Number of units in each stage.
        use_prm (bool): Whether to use pose refine machine.
            Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 out_shape,
                 unit_channels=256,
                 out_channels=17,
                 num_stages=4,
                 num_units=4,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()

        self.out_shape = out_shape
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.predict_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            for j in range(self.num_units):
                self.predict_layers.append(
                    PredictHeatmap(
                        unit_channels,
                        out_channels,
                        out_shape,
                        use_prm,
                        norm_cfg=norm_cfg))

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        assert len(x) == self.num_stages
        for i in range(self.num_stages):
            for j in range(self.num_units):
                y = self.predict_layers[i * self.num_units + j](x[i][j])
                out.append(y)

        return out

    def init_weights(self):
        """Initialize model weights."""

        for m in self.predict_layers.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
