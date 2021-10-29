# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

# from mmcv.cnn import ConvModule


class Basic3DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d')):
        super(Basic3DBlock, self).__init__()
        # self.block = ConvModule(in_channels, out_channels, kernel_size,
        #                         stride=1,
        #                         padding=((kernel_size-1)//2),
        #                         conv_cfg=conv_cfg,
        #                         norm_cfg=norm_cfg, bias=True)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2)),
            nn.BatchNorm3d(out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d')):
        super(Res3DBlock, self).__init__()
        # self.res_branch = nn.Sequential(
        #     ConvModule(in_channels, out_channels, kernel_size,
        #                stride=1, padding=((kernel_size - 1) // 2),
        #                conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=True),
        #     ConvModule(out_channels, out_channels, kernel_size,
        #                stride=1, padding=((kernel_size - 1) // 2),
        #                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
        #                act_cfg=None, bias=True))
        self.res_branch = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(True),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, stride=1,
                padding=1), nn.BatchNorm3d(out_channels))

        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            # self.skip_con =
            # ConvModule(in_channels, out_channels, 1,
            #          stride=1, padding=0, conv_cfg=conv_cfg,
            #          norm_cfg=norm_cfg, act_cfg=None,
            #          bias=True)
            self.skip_con = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), nn.BatchNorm3d(out_channels))

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(
            x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Upsample3DBlock, self).__init__()
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
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self, in_channels=32):
        super(EncoderDecorder, self).__init__()

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


@BACKBONES.register_module()
class V2VNet(BaseBackbone):

    def __init__(self, input_channels, output_channels, mid_channels=32):
        super(V2VNet, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, mid_channels // 2, 7),
            Res3DBlock(mid_channels // 2, mid_channels),
        )

        self.encoder_decoder = EncoderDecorder(in_channels=mid_channels)

        self.output_layer = nn.Conv3d(
            mid_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
