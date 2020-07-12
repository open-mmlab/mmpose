import torch.nn as nn
from mmcv.cnn import build_conv_layer, normal_init

from ..registry import HEADS


@HEADS.register_module()
class BottomUpSimpleHead(nn.Module):
    """Bottom-up simple head.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        with_ae_loss (list[bool]): Option to use ae loss or not.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 tag_per_joint=True,
                 with_ae_loss=None,
                 extra=None):
        super(BottomUpSimpleHead, self).__init__()

        self.in_channels = in_channels
        dim_tag = num_joints if tag_per_joint else 1
        out_channels = num_joints + dim_tag \
            if with_ae_loss[0] else num_joints

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            else:
                padding = 0
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        self.final_layers = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        final_outputs = []
        y = self.final_layers(x)
        final_outputs.append(y)
        return final_outputs

    def init_weights(self):
        for m in self.final_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
