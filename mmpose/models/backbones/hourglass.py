import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import BasicBlock, ResLayer
from .utils import load_checkpoint


class HourglassModule(nn.Module):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 int=None,
                 out=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_block, cur_channel, cur_channel, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_block,
            cur_channel,
            next_channel,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_block,
                next_channel,
                next_channel,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            cur_block,
            next_channel,
            cur_channel,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = nn.Upsample(scale_factor=2)

        self.int = int
        self.out = out

        if int is not None:
            self.adjust_channel_int = ResLayer(
                BasicBlock, 1, int, cur_channel, stride=1, norm_cfg=norm_cfg)

        if out is not None:
            self.adjust_channel_out = ResLayer(
                BasicBlock, 1, cur_channel, out, stride=1, norm_cfg=norm_cfg)

    def forward(self, x):
        """Model forward function."""
        if self.int is not None:
            x = self.adjust_channel_int(x)

        # up1 256*64*64
        up1 = self.up1(x)
        # up1 256*64*64
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        out = up1 + up2
        if self.out is not None:
            out = self.adjust_channel_out(out)
        return out


@BACKBONES.register_module()
class HourglassNet(BaseBackbone):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times=5,
                 num_stacks=2,
                 stage_channels=(256, 256, 384, 384, 384, 512),
                 stage_blocks=(2, 2, 2, 2, 2, 4),
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        # cur_channel = stage_channels[0]

        self.stem_1 = nn.Sequential(
            ConvModule(3, 64, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 1, 64, 128, stride=2, norm_cfg=norm_cfg))

        self.stem_2 = nn.Sequential(
            ResLayer(BasicBlock, 2, 128, 128, stride=1, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 1, 128, 256, stride=1, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks),
            HourglassModule(downsample_times, stage_channels, stage_blocks),
            HourglassModule(downsample_times, stage_channels, stage_blocks),
            HourglassModule(downsample_times, stage_channels, stage_blocks)
        ])

        self.adjust_channel = nn.ModuleList([
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None),
                ConvModule(256, 128, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None),
                ConvModule(256, 128, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None),
                ConvModule(256, 128, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None),
                ConvModule(256, 512, 1, norm_cfg=norm_cfg, act_cfg=None))
        ])

        self.final_convs = nn.ModuleList([
            nn.Sequential(
                ConvModule(128, 1 * 16, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(128, 2 * 16, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(128, 4 * 16, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(512, 64 * 16, 1, norm_cfg=norm_cfg, act_cfg=None))
        ])

        self.remap_1 = nn.ModuleList([
            nn.Sequential(
                ConvModule(1 * 16, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(2 * 16, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(4 * 16, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
        ])

        self.remap_2 = nn.ModuleList([
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
            nn.Sequential(
                ConvModule(256, 256, 1, norm_cfg=norm_cfg, act_cfg=None)),
        ])

        # self.inters = ResLayer(
        #     BasicBlock,
        #     num_stacks - 1,
        #     cur_channel,
        #     cur_channel,
        #     norm_cfg=norm_cfg)

        # self.conv1x1s = nn.ModuleList([
        #     ConvModule(
        #         cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
        #     for _ in range(num_stacks - 1)
        # ])

        # self.out_convs = nn.ModuleList([
        #     ConvModule(
        #         cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
        #     for _ in range(num_stacks)
        # ])

        # self.final_convs = nn.ModuleList([
        #     nn.Sequential(
        #         ConvModule(
        #             feat_channel,
        #             feat_channel,
        #             1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=None),
        #         nn.Conv2d(feat_channel, 16, 1),
        #     ),
        #     nn.Sequential(
        #         ConvModule(
        #             feat_channel,
        #             feat_channel,
        #             1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=None),
        #         nn.Conv2d(feat_channel, 16 * 2, 1),
        #     ),
        #     nn.Sequential(
        #         ConvModule(
        #             feat_channel,
        #             feat_channel,
        #             1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=None),
        #         nn.Conv2d(feat_channel, 16 * 4, 1),
        #     ),
        #     nn.Sequential(
        #         ConvModule(
        #             feat_channel,
        #             feat_channel,
        #             1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=None),
        #         nn.Conv2d(feat_channel, 16 * 8, 1),
        #     ),
        #     nn.Sequential(
        #         ConvModule(
        #             feat_channel,
        #             feat_channel,
        #             1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=None),
        #         nn.Conv2d(feat_channel, 16 * 64, 1),
        #     )
        # ])

        # self.remap_convs = nn.ModuleList([
        #     ConvModule(
        #         feat_channel,
        #         cur_channel,
        #         1, norm_cfg=norm_cfg, act_cfg=None)
        #     for _ in range(num_stacks - 1)
        # ])

        # self.remap_final = nn.ModuleList([
        #     ConvModule(16, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None),
        #     ConvModule(
        #         16 * 2, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None),
        #     ConvModule(
        #         16 * 4, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None),
        #     ConvModule(
        #         16 * 8, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None),
        # ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""

        mid_feat = self.stem_1(x)
        inter_feat = self.stem_2(mid_feat)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            adjust = self.adjust_channel[ind]
            final = self.final_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            res = adjust(hourglass_feat)
            out_feat = final(res)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                # import pdb
                # pdb.set_trace()
                r1 = self.remap_1[ind](out_feat)
                cat = torch.cat([res, mid_feat], axis=1)
                r2 = self.remap_2[ind](cat)

                inter_feat = r2 + r1
                mid_feat = res

        return out_feats
