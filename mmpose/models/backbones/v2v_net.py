# ------------------------------------------------------------------------------
# Copyright and License Information
# Adapted from
# https://github.com/microsoft/voxelpose-pytorch/blob/main/lib/models/v2v_net.py
# Original Licence: MIT License
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class Basic3DBlock(BaseModule):
    """A basic 3D convolutional block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the convolution operation
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: dict(type='Conv3d')
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN3d')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 init_cfg=None):
        super(Basic3DBlock, self).__init__(init_cfg=init_cfg)
        self.block = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True)

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class Res3DBlock(BaseModule):
    """A residual 3D convolutional block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the convolution operation
            Default: 3
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: dict(type='Conv3d')
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN3d')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 init_cfg=None):
        super(Res3DBlock, self).__init__(init_cfg=init_cfg)
        self.res_branch = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=True),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True))

        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)

    def forward(self, x):
        """Forward function."""
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(BaseModule):
    """A 3D max-pool block.

    Args:
        pool_size (int): Pool size of the 3D max-pool layer
    """

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        """Forward function."""
        return F.max_pool3d(
            x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(BaseModule):
    """A 3D upsample block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        kernel_size (int): Kernel size of the transposed convolution operation.
            Default: 2
        stride (int):  Kernel size of the transposed convolution operation.
            Default: 2
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2,
                 init_cfg=None):
        super(Upsample3DBlock, self).__init__(init_cfg=init_cfg)
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0), nn.BatchNorm3d(out_channels), nn.ReLU(True))

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class EncoderDecorder(BaseModule):
    """An encoder-decoder block.

    Args:
        in_channels (int): Input channels of this block
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, in_channels=32, init_cfg=None):
        super(EncoderDecorder, self).__init__(init_cfg=init_cfg)

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(in_channels, in_channels * 2)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(in_channels * 2, in_channels * 4)

        self.mid_res = Res3DBlock(in_channels * 4, in_channels * 4)

        self.decoder_res2 = Res3DBlock(in_channels * 4, in_channels * 4)
        self.decoder_upsample2 = Upsample3DBlock(in_channels * 4,
                                                 in_channels * 2, 2, 2)
        self.decoder_res1 = Res3DBlock(in_channels * 2, in_channels * 2)
        self.decoder_upsample1 = Upsample3DBlock(in_channels * 2, in_channels,
                                                 2, 2)

        self.skip_res1 = Res3DBlock(in_channels, in_channels)
        self.skip_res2 = Res3DBlock(in_channels * 2, in_channels * 2)

    def forward(self, x):
        """Forward function."""
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


@MODELS.register_module()
class V2VNet(BaseBackbone):
    """V2VNet.

    Please refer to the `paper <https://arxiv.org/abs/1711.07399>`
        for details.

    Args:
        input_channels (int):
            Number of channels of the input feature volume.
        output_channels (int):
            Number of channels of the output volume.
        mid_channels (int):
            Input and output channels of the encoder-decoder block.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: ``dict(
                type='Normal',
                std=0.001,
                layer=['Conv3d', 'ConvTranspose3d']
            )``
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 mid_channels=32,
                 init_cfg=dict(
                     type='Normal',
                     std=0.001,
                     layer=['Conv3d', 'ConvTranspose3d'])):
        super(V2VNet, self).__init__(init_cfg=init_cfg)

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, mid_channels // 2, 7),
            Res3DBlock(mid_channels // 2, mid_channels),
        )

        self.encoder_decoder = EncoderDecorder(in_channels=mid_channels)

        self.output_layer = nn.Conv3d(
            mid_channels, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward function."""
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return (x, )
