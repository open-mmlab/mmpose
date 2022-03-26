# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x


@BACKBONES.register_module()
class PoseFormer(BaseBackbone):

    def __init__(self,
                 num_frame=9,
                 num_joints=17,
                 in_chans=2,
                 spatial_embed_dim=32,
                 spatial_depth=4,
                 temporal_depth=4,
                 num_heads=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        """
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels,
                            2D joints have 2 channels: (x,y)
            spatial_embed_dim (int): embedding dimension of
                                     the spatial transformer
            spatial_depth (int): depth of the spatial transformer
            temporal_depth (int): depth of the temporal transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale
                              of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        # Temporal embed_dim is num_joints * spatial embedding dim ratio
        temporal_embed_dim = spatial_embed_dim * num_joints
        # output dimension is num_joints * 3

        # Spatial patch embedding
        self.spatial_patch_to_embedding = nn.Linear(in_chans,
                                                    spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frame, temporal_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        spatial_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, spatial_depth)
        ]

        self.spatial_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dims=spatial_embed_dim,
                num_heads=num_heads,
                feedforward_channels=int(mlp_ratio * spatial_embed_dim),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=spatial_dpr[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg) for i in range(spatial_depth)
        ])

        temporal_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, temporal_depth)
        ]

        self.temporal_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dims=temporal_embed_dim,
                num_heads=num_heads,
                feedforward_channels=int(mlp_ratio * temporal_embed_dim),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=temporal_dpr[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg) for i in range(temporal_depth)
        ])

        self.spatial_norm = build_norm_layer(norm_cfg, spatial_embed_dim)[1]
        self.temporal_norm = build_norm_layer(norm_cfg, temporal_embed_dim)[1]

        # An easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(
            in_channels=num_frame, out_channels=1, kernel_size=1)

    def spatial_forward_features(self, x):
        # b is batch size, f is number of frames, p is number of joints
        b, _, f, p = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(b * f, p, -1)

        x = self.spatial_patch_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.spatial_norm(x)

        _, w, c = x.shape
        x = x.view(b, f, w * c)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.temporal_blocks:
            x = blk(x)

        x = self.temporal_norm(x)
        # x size [b, f, emb_dim]
        # Then take weighted mean on frame dimension
        # We only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Now x is [batch_size, 2 channels, receptive frames, joint_num]
        x = self.spatial_forward_features(x)
        x = self.forward_features(x)

        return x
