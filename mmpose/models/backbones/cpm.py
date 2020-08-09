import torch
import torch.nn as nn
from mmcv.cnn import VGG, ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from .utils import load_checkpoint


class CpmBlock(nn.Module):
    """CpmBlock for Convolutional Pose Machine.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ConvModule(in_channels, out_channels, 7, padding=3),
            ConvModule(out_channels, out_channels, 7, padding=3),
            ConvModule(out_channels, out_channels, 7, padding=3),
            ConvModule(out_channels, out_channels, 7, padding=3),
            ConvModule(out_channels, out_channels, 7, padding=3))

    def forward(self, x):
        """Model forward function."""
        out = self.model(x)
        return out


@BACKBONES.register_module()
class CPM(BaseBackbone):
    """CPM backbone.

    Convolutional Pose Machines.
    More details can be found in the `paper
    <https://arxiv.org/abs/1602.00134>`_ .

    Args:
        num_stages (int): Number of stages.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import CPM
        >>> import torch
        >>> self = CPM()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 feat_channels=128,
                 num_stages=6):
        super().__init__()

        assert in_channels == 3

        self.num_stages = num_stages
        assert self.num_stages >= 1

        vgg = VGG(
            depth=19,
            with_bn=False,
            num_classes=-1,
            num_stages=4,
            dilations=(1, 1, 1, 1),
            out_indices=(3, ),
            frozen_stages=-1,
            bn_eval=False,
            bn_frozen=False,
            ceil_mode=False,
            with_last_pool=False)
        self.vgg = nn.Sequential(*list(vgg.features.children())[:-4])
        self.stage1 = nn.Sequential(
            ConvModule(512, 256, 3, padding=1),
            ConvModule(256, 256, 3, padding=1),
            ConvModule(256, 256, 3, padding=1),
            ConvModule(256, 256, 3, padding=1),
            ConvModule(256, feat_channels, 3, padding=1))

        self.out_stage1 = nn.Sequential(
            ConvModule(feat_channels, 512, 1),
            ConvModule(512, out_channels, 1, act_cfg=None))

        self.cpm_stages = nn.ModuleList([
            CpmBlock(feat_channels + out_channels, feat_channels)
            for _ in range(num_stages - 1)
        ])

        self.out_convs = nn.ModuleList([
            nn.Sequential(
                ConvModule(feat_channels, feat_channels, 1, padding=0),
                ConvModule(feat_channels, out_channels, 1, act_cfg=None))
            for _ in range(num_stages - 1)
        ])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        if isinstance(pretrained, str):
            # initialize the VGG stem
            logger = get_root_logger()
            load_checkpoint(self.vgg, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stage1(self.vgg(x))
        out_feats = []

        out_feats.append(self.out_stage1(inter_feat))

        for ind in range(self.num_stages - 1):
            single_stage = self.cpm_stages[ind]
            out_conv = self.out_convs[ind]

            inp_feat = torch.cat([out_feats[-1], inter_feat], 1)
            cpm_feat = single_stage(inp_feat)
            out_feat = out_conv(cpm_feat)
            out_feats.append(out_feat)

        return out_feats
