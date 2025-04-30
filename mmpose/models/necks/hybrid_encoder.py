# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmpose.models.utils import (DetrTransformerEncoder, RepVGGBlock,
                                 SinePositionalEncoding)
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType


class CSPRepLayer(BaseModule):
    """CSPRepLayer, a layer that combines Cross Stage Partial Networks with
    RepVGG Blocks.

    Args:
        in_channels (int): Number of input channels to the layer.
        out_channels (int): Number of output channels from the layer.
        num_blocks (int): The number of RepVGG blocks to be used in the layer.
            Defaults to 3.
        widen_factor (float): Expansion factor for intermediate channels.
            Determines the hidden channel size based on out_channels.
            Defaults to 1.0.
        norm_cfg (dict): Configuration for normalization layers.
            Defaults to Batch Normalization with trainable parameters.
        act_cfg (dict): Configuration for activation layers.
            Defaults to SiLU (Swish) with in-place operation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 widen_factor: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * widen_factor)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@MODELS.register_module()
class HybridEncoder(BaseModule):
    """Hybrid encoder neck introduced in `RT-DETR` by Lyu et al (2023),
    combining transformer encoders with a Feature Pyramid Network (FPN) and a
    Path Aggregation Network (PAN).

    Args:
        encoder_cfg (ConfigType): Configuration for the transformer encoder.
        projector (OptConfigType, optional): Configuration for an optional
            projector module. Defaults to None.
        num_encoder_layers (int, optional): Number of encoder layers.
            Defaults to 1.
        in_channels (List[int], optional): Input channels of feature maps.
            Defaults to [512, 1024, 2048].
        feat_strides (List[int], optional): Strides of feature maps.
            Defaults to [8, 16, 32].
        hidden_dim (int, optional): Hidden dimension of the MLP.
            Defaults to 256.
        use_encoder_idx (List[int], optional): Indices of encoder layers to
            use. Defaults to [2].
        pe_temperature (int, optional): Positional encoding temperature.
            Defaults to 10000.
        widen_factor (float, optional): Expansion factor for CSPRepLayer.
            Defaults to 1.0.
        deepen_factor (float, optional): Depth multiplier for CSPRepLayer.
            Defaults to 1.0.
        spe_learnable (bool, optional): Whether positional encoding is
            learnable. Defaults to False.
        output_indices (Optional[List[int]], optional): Indices of output
            layers. Defaults to None.
        norm_cfg (OptConfigType, optional): Configuration for normalization
            layers. Defaults to Batch Normalization.
        act_cfg (OptConfigType, optional): Configuration for activation
            layers. Defaults to SiLU (Swish) with in-place operation.

    .. _`RT-DETR`: https://arxiv.org/abs/2304.08069
    """

    def __init__(self,
                 encoder_cfg: ConfigType = dict(),
                 projector: OptConfigType = None,
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 use_encoder_idx: List[int] = [2],
                 pe_temperature: int = 10000,
                 widen_factor: float = 1.0,
                 deepen_factor: float = 1.0,
                 spe_learnable: bool = False,
                 output_indices: Optional[List[int]] = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.output_indices = output_indices

        # channel projection
        self.input_proj = ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        # encoder transformer
        if len(use_encoder_idx) > 0:
            pos_enc_dim = self.hidden_dim // 2
            self.encoder = ModuleList([
                DetrTransformerEncoder(num_encoder_layers, encoder_cfg)
                for _ in range(len(use_encoder_idx))
            ])

        self.sincos_pos_enc = SinePositionalEncoding(
            pos_enc_dim,
            learnable=spe_learnable,
            temperature=self.pe_temperature,
            spatial_dim=2)

        # top-down fpn
        lateral_convs = list()
        fpn_blocks = list()
        for idx in range(len(in_channels) - 1, 0, -1):
            lateral_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * deepen_factor),
                    act_cfg=act_cfg,
                    widen_factor=widen_factor))
        self.lateral_convs = ModuleList(lateral_convs)
        self.fpn_blocks = ModuleList(fpn_blocks)

        # bottom-up pan
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(in_channels) - 1):
            downsample_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * deepen_factor),
                    act_cfg=act_cfg,
                    widen_factor=widen_factor))
        self.downsample_convs = ModuleList(downsample_convs)
        self.pan_blocks = ModuleList(pan_blocks)

        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        proj_feats = [
            self.input_proj[i](inputs[i]) for i in range(len(inputs))
        ]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(
                    0, 2, 1).contiguous()

                if torch.onnx.is_in_onnx_export():
                    pos_enc = getattr(self, f'pos_enc_{i}')
                else:
                    pos_enc = self.sincos_pos_enc(size=(h, w))
                    pos_enc = pos_enc.transpose(-1, -2).reshape(1, h * w, -1)
                memory = self.encoder[i](
                    src_flatten, query_pos=pos_enc, key_padding_mask=None)

                proj_feats[enc_ind] = memory.permute(
                    0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(
                feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # Conv
            out = self.pan_blocks[idx](  # CSPRepLayer
                torch.cat([downsample_feat, feat_high], axis=1))
            outs.append(out)

        if self.output_indices is not None:
            outs = [outs[i] for i in self.output_indices]

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)

    def switch_to_deploy(self, test_cfg):
        """Switch to deploy mode."""

        if getattr(self, 'deploy', False):
            return

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = test_cfg['input_size']
                h = int(h / 2**(3 + enc_ind))
                w = int(w / 2**(3 + enc_ind))
                pos_enc = self.sincos_pos_enc(size=(h, w))
                pos_enc = pos_enc.transpose(-1, -2).reshape(1, h * w, -1)
                self.register_buffer(f'pos_enc_{i}', pos_enc)

        self.deploy = True
