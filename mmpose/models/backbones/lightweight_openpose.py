import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import load_checkpoint


class CpmLayer(nn.Module):
    """A CPM-type layer.

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = ConvModule(
            in_channels, out_channels, 1, padding=0, norm_cfg=None)
        self.feat = nn.Sequential(
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ELU'),
                norm_cfg=None))
        self.out_conv = ConvModule(
            out_channels, out_channels, 3, padding=1, norm_cfg=None)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.out_conv(x + self.feat(x))
        return x


class InitialStage(nn.Module):
    """The initial stage.

    Args:
        in_channels (int): The input channels.
        mid_channels (int): The middle-layer channels.
        out_channels_cm (int): The output channels for CM (
            confidence map, or heatmap).
        out_channels_paf (int): The output channels for PAF (
            part-affinity field).
    """

    def __init__(self, in_channels, mid_channels, out_channels_cm,
                 out_channels_paf):
        super().__init__()
        self.feat = nn.Sequential(
            ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=None),
            ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=None),
            ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=None))
        self.cm_out_conv = nn.Sequential(
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=1,
                padding=0,
                norm_cfg=None),
            ConvModule(
                mid_channels,
                out_channels_cm,
                kernel_size=1,
                padding=0,
                norm_cfg=None,
                act_cfg=None))
        self.paf_out_conv = nn.Sequential(
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=1,
                padding=0,
                norm_cfg=None),
            ConvModule(
                mid_channels,
                out_channels_paf,
                kernel_size=1,
                padding=0,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, x):
        features = self.feat(x)
        cm_output = self.cm_out_conv(features)
        paf_output = self.paf_out_conv(features)
        return [cm_output, paf_output]


class RefinementStageBlock(nn.Module):
    """The block for the refinement stage.

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = ConvModule(
            in_channels, out_channels, 1, padding=0, norm_cfg=None)
        self.feat = nn.Sequential(
            ConvModule(
                out_channels, out_channels, 3, padding=1, norm_cfg=None),
            ConvModule(
                out_channels,
                out_channels,
                3,
                dilation=2,
                padding=2,
                norm_cfg=None))

    def forward(self, x):
        pre_features = self.pre_conv(x)
        features = self.feat(pre_features)
        return pre_features + features


class RefinementStage(nn.Module):
    """The refinement stage.

    Args:
        in_channels (int): The input channels.
        mid_channels (int): The middle-layer channels.
        out_channels_cm (int): The output channels for CM (
            confidence map, or heatmap).
        out_channels_paf (int): The output channels for PAF (
            part-affinity field).
    """

    def __init__(self, in_channels, mid_channels, out_channels_cm,
                 out_channels_paf):
        super().__init__()
        self.feat = nn.Sequential(
            RefinementStageBlock(in_channels, mid_channels),
            RefinementStageBlock(mid_channels, mid_channels),
            RefinementStageBlock(mid_channels, mid_channels),
            RefinementStageBlock(mid_channels, mid_channels),
            RefinementStageBlock(mid_channels, mid_channels))
        self.cm_out_conv = nn.Sequential(
            ConvModule(
                mid_channels, mid_channels, 1, padding=0, norm_cfg=None),
            ConvModule(
                mid_channels,
                out_channels_cm,
                1,
                padding=0,
                norm_cfg=None,
                act_cfg=None))
        self.paf_out_conv = nn.Sequential(
            ConvModule(
                mid_channels, mid_channels, 1, padding=0, norm_cfg=None),
            ConvModule(
                mid_channels,
                out_channels_paf,
                1,
                padding=0,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, x):
        features = self.feat(x)
        cm_output = self.cm_out_conv(features)
        paf_output = self.paf_out_conv(features)
        return [cm_output, paf_output]


@BACKBONES.register_module()
class LightweightOpenPoseNetwork(BaseBackbone):
    """Lightweight OpenPose backbone Network.

    Real-time 2D Multi-Person Pose Estimation on
    CPU: Lightweight OpenPose

    More details can be found in the `paper
    <https://arxiv.org/pdf/1811.12004>`__ .

    Args:
        in_channels (int): The input channels.
        out_channels_cm (int): The output channels for CM (
            confidence map, or heatmap).
        out_channels_paf (int): The output channels for PAF (
            part-affinity field).
        stem_feat_channels (int): Feature channel of the stem network.
        num_stages (int): Number of stages.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import LightweightOpenPoseNetwork
        >>> import torch
        >>> self = LightweightOpenPoseNetwork(3, 19, 38)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
    """

    def __init__(self,
                 in_channels,
                 out_channels_cm=19,
                 out_channels_paf=38,
                 stem_feat_channels=128,
                 num_stages=2,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        assert in_channels == 3

        self.num_stages = num_stages
        assert self.num_stages >= 1

        self.features = nn.Sequential(
            ConvModule(
                in_channels,
                32,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                bias=False),
            # conv_dw(32, 64)
            ConvModule(
                32, 32, 3, padding=1, groups=32, bias=False,
                norm_cfg=norm_cfg),
            ConvModule(32, 64, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(64, 128, stride=2)
            ConvModule(
                64,
                64,
                3,
                stride=2,
                padding=1,
                groups=64,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(64, 128, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(128, 128)
            ConvModule(
                128,
                128,
                3,
                padding=1,
                groups=128,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(128, 128, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(128, 256, stride=2)
            ConvModule(
                128,
                128,
                3,
                stride=2,
                padding=1,
                groups=128,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(128, 256, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(256, 256)
            ConvModule(
                256,
                256,
                3,
                padding=1,
                groups=256,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(256, 256, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(256, 512)
            ConvModule(
                256,
                256,
                3,
                padding=1,
                groups=256,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(256, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(512, 512, dilation=2, padding=2)
            ConvModule(
                512,
                512,
                3,
                padding=2,
                dilation=2,
                groups=512,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(512, 512)
            ConvModule(
                512,
                512,
                3,
                padding=1,
                groups=512,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(512, 512)
            ConvModule(
                512,
                512,
                3,
                padding=1,
                groups=512,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(512, 512)
            ConvModule(
                512,
                512,
                3,
                padding=1,
                groups=512,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg),
            # conv_dw(512, 512)
            ConvModule(
                512,
                512,
                3,
                padding=1,
                groups=512,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, bias=False, norm_cfg=norm_cfg))

        self.cpm = CpmLayer(512, stem_feat_channels)

        self.initial_stage = InitialStage(stem_feat_channels, 512,
                                          out_channels_cm, out_channels_paf)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_stages - 1):
            self.refinement_stages.append(
                RefinementStage(
                    stem_feat_channels + out_channels_cm + out_channels_paf,
                    stem_feat_channels, out_channels_cm, out_channels_paf))

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
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        stem_feat = self.cpm(self.features(x))

        cm_outputs = []
        paf_outputs = []

        cm_output, paf_output = self.initial_stage(stem_feat)
        cm_outputs.append(cm_output)
        paf_outputs.append(paf_output)

        for refinement_stage in self.refinement_stages:
            cm_output, paf_output = refinement_stage(
                torch.cat([stem_feat, cm_outputs[-1], paf_outputs[-1]], dim=1))
            cm_outputs.append(cm_output)
            paf_outputs.append(paf_output)

        return [*cm_outputs, *paf_outputs]
