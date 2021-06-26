import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .cpm import CpmBlock
from .utils import load_checkpoint


@BACKBONES.register_module()
class OpenPoseNetwork(BaseBackbone):
    """OpenPose backbone Network.

    Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.
    More details can be found in the `paper
    <https://arxiv.org/pdf/1611.08050>`__ .

    Based on the officially released model
    'https://github.com/CMU-Perceptual-Computing-Lab/openpose/
    blob/master/models/pose/coco/pose_deploy_linevec.prototxt'

    Args:
        in_channels (int): The input channels.
        out_channels_heatmap (int): The output channels for heatmap.
        out_channels_heatmap (int): The output channels for paf.
        feat_channels (int): Feature channel of each stage.
        num_stages (int): Number of stages.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmpose.models import OpenPoseNetwork
        >>> import torch
        >>> self = OpenPoseNetwork(3, 19, 38)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 19, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
    """

    def __init__(self,
                 in_channels,
                 out_channels_heatmap=19,
                 out_channels_paf=38,
                 feat_channels=128,
                 num_stages=6,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        assert in_channels == 3

        self.num_stages = num_stages
        assert self.num_stages >= 1

        self.stem = nn.Sequential(
            ConvModule(in_channels, 64, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(64, 64, 3, padding=1, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvModule(64, 128, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(128, 128, 3, padding=1, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvModule(128, 256, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvModule(256, 512, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(512, 256, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(256, feat_channels, 3, padding=1, norm_cfg=norm_cfg))

        # stage 0
        self.heatmap_stages = nn.ModuleList([
            CpmBlock(feat_channels,
                     [feat_channels, feat_channels, feat_channels, 512],
                     [3, 3, 3, 1], norm_cfg)
        ])

        # stage 1 to n-1
        for _ in range(1, self.num_stages):
            self.heatmap_stages.append(
                CpmBlock(
                    feat_channels + out_channels_heatmap + out_channels_paf, [
                        feat_channels, feat_channels, feat_channels,
                        feat_channels, feat_channels, feat_channels
                    ], [7, 7, 7, 7, 7, 1], norm_cfg))

        # stage 0
        self.paf_stages = nn.ModuleList([
            CpmBlock(feat_channels,
                     [feat_channels, feat_channels, feat_channels, 512],
                     [3, 3, 3, 1], norm_cfg)
        ])

        # stage 1 to n-1
        for _ in range(1, self.num_stages):
            self.paf_stages.append(
                CpmBlock(
                    feat_channels + out_channels_heatmap + out_channels_paf, [
                        feat_channels, feat_channels, feat_channels,
                        feat_channels, feat_channels, feat_channels
                    ], [7, 7, 7, 7, 7, 1], norm_cfg))

        # stage 0
        self.heatmap_out_convs = nn.ModuleList(
            [ConvModule(512, out_channels_heatmap, 1, act_cfg=None)])
        # stage 1 to n-1
        for _ in range(1, self.num_stages):
            self.heatmap_out_convs.append(
                ConvModule(
                    feat_channels, out_channels_heatmap, 1, act_cfg=None))

        # stage 0
        self.paf_out_convs = nn.ModuleList(
            [ConvModule(512, out_channels_paf, 1, act_cfg=None)])
        # stage 1 to n-1
        for _ in range(1, self.num_stages):
            self.paf_out_convs.append(
                ConvModule(feat_channels, out_channels_paf, 1, act_cfg=None))

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
        stem_feat = self.stem(x)
        out_feats = []
        out_feats.append(stem_feat)

        heatmap_outputs = []
        paf_outputs = []

        for ind in range(self.num_stages):
            heatmap_stage = self.heatmap_stages[ind]
            paf_stage = self.paf_stages[ind]

            heatmap_out_conv = self.heatmap_out_convs[ind]
            paf_out_conv = self.paf_out_convs[ind]

            heatmap_output = heatmap_out_conv(heatmap_stage(out_feats[-1]))
            heatmap_outputs.append(heatmap_output)
            paf_output = paf_out_conv(paf_stage(out_feats[-1]))
            paf_outputs.append(paf_output)

            out_feat = torch.cat([stem_feat, heatmap_output, paf_output], 1)

            out_feats.append(out_feat)

        return [*heatmap_outputs, *paf_outputs]
