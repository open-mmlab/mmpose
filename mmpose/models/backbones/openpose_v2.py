import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import load_checkpoint


class MconvBlock(nn.Module):
    """MconvBlock for replacing convolutions of 7x7 kernel.

    Args:
        in_channels (int): Input channels of this block.
        channels (list): Output channels of each conv module.
        kernels (list): Kernel sizes of each conv module.
    """

    def __init__(self,
                 in_channels,
                 channels=(96, 96, 96),
                 kernels=(3, 3, 3),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU')):
        super().__init__()

        assert len(channels) == len(kernels)

        self.num_layers = len(channels)

        self.model = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                input_channels = in_channels
            else:
                input_channels = channels[i - 1]
            self.model.append(
                ConvModule(
                    input_channels,
                    channels[i],
                    kernels[i],
                    padding=(kernels[i] - 1) // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, x):
        """Model forward function."""
        feat = []
        feat.append(x)
        for i in range(self.num_layers):
            feat.append(self.model[i](feat[-1]))
        out = torch.cat([*feat[1:]], 1)
        return out


class MconvStage(nn.Module):
    """MconvStage.

    Args:
        in_channels (int): Input channels of this block.
        channels (list): Output channels of each conv module.
        kernels (list): Kernel sizes of each conv module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=5,
                 block_channels=(96, 96, 96),
                 block_kernels=(3, 3, 3),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU')):
        super().__init__()

        layers = []
        for i in range(num_blocks):

            if i == 0:
                input_channels = in_channels
            else:
                input_channels = sum(block_channels)

            layers.append(
                MconvBlock(
                    input_channels,
                    channels=block_channels,
                    kernels=block_kernels,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='PReLU')))

        layers.append(
            ConvModule(
                sum(block_channels),
                out_channels,
                kernel_size=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Model forward function."""
        out = self.model(x)
        return out


@BACKBONES.register_module()
class OpenPoseNetworkV2(BaseBackbone):
    """OpenPose backbone Network.

    Open{P}ose: realtime multi-person 2{D} pose estimation
    using {P}art {A}ffinity {F}ields.
    More details can be found in the `paper
    <https://arxiv.org/abs/1812.08008>`__ .

    Based on the officially released model
    'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
    'openpose/master/models/pose/body_25/pose_deploy.prototxt'

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
        >>> from mmpose.models import OpenPoseNetworkV2
        >>> import torch
        >>> self = OpenPoseNetworkV2(3)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 38, 46, 46)
        (1, 19, 46, 46)
    """

    def __init__(self,
                 in_channels,
                 out_channels_cm=19,
                 out_channels_paf=38,
                 stem_feat_channels=128,
                 num_stages=6,
                 stage_types=('PAF', 'PAF', 'PAF', 'PAF', 'PAF', 'CM'),
                 num_blocks=5,
                 block_channels=96,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        assert in_channels == 3
        assert num_stages == len(stage_types)

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
            ConvModule(
                512, 512, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                512, 256, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                256,
                stem_feat_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.stages = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i, stage_type in enumerate(stage_types):
            if i == 0:
                input_channels = stem_feat_channels
            else:
                if stage_types[i - 1] == 'CM':
                    input_channels = stem_feat_channels + out_channels_cm
                else:
                    # stage_types[i-1] == 'PAF':
                    input_channels = stem_feat_channels + out_channels_paf

            if stage_type.upper() == 'CM':
                self.stages.append(
                    MconvStage(
                        input_channels,
                        256,
                        num_blocks=num_blocks,
                        block_channels=[block_channels] * num_blocks,
                        block_kernels=[3] * num_blocks,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='PReLU')))
                self.out_convs.append(
                    ConvModule(256, out_channels_cm, 1, act_cfg=None))

            elif stage_type.upper() == 'PAF':
                self.stages.append(
                    MconvStage(
                        input_channels,
                        256,
                        num_blocks=num_blocks,
                        block_channels=[block_channels] * num_blocks,
                        block_kernels=[3] * num_blocks,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='PReLU')))
                self.out_convs.append(
                    ConvModule(256, out_channels_paf, 1, act_cfg=None))

            else:
                raise ValueError("stage_type should be either 'CM' or 'PAF'.")

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

        outputs = []

        for ind in range(self.num_stages):
            single_stage = self.stages[ind]
            single_out_conv = self.out_convs[ind]
            output = single_out_conv(single_stage(out_feats[-1]))
            outputs.append(output)

            out_feat = torch.cat([stem_feat, output], 1)
            out_feats.append(out_feat)

        return [*outputs]
