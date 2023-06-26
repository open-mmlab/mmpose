# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init
from mmengine.model.weight_init import trunc_normal_

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class Attention(BaseModule):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = mode

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_count_s = None
        self.attn_count_t = None

    def forward(self, x, seq_len=1):
        B, N, C = x.shape

        if self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[
                2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seq_len=seq_len)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[
                2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seq_len=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seq_len, self.num_heads, N,
                       C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        kt = k.reshape(-1, seq_len, self.num_heads, N,
                       C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        vt = v.reshape(-1, seq_len, self.num_heads, N,
                       C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, N, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x


class AttentionBlock(BaseModule):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 mlp_out_ratio=1.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 st_mode='st'):
        super().__init__()

        self.st_mode = st_mode
        self.norm1_s = nn.LayerNorm(dim, eps=1e-06)
        self.norm1_t = nn.LayerNorm(dim, eps=1e-06)

        self.attn_s = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            mode='spatial')
        self.attn_t = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            mode='temporal')

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = nn.LayerNorm(dim, eps=1e-06)
        self.norm2_t = nn.LayerNorm(dim, eps=1e-06)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim), nn.Dropout(drop))
        self.mlp_t = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim), nn.Dropout(drop))

    def forward(self, x, seq_len=1):
        if self.st_mode == 'st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seq_len))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seq_len))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seq_len))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seq_len))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        else:
            raise NotImplementedError(self.st_mode)
        return x


@MODELS.register_module()
class DSTFormer(BaseBackbone):
    """Dual-stream Spatio-temporal Transformer Module.

    Args:
        in_channels (int): Number of input channels.
        feat_size: Number of feature channels. Default: 256.
        depth: The network depth. Default: 5.
        num_heads: Number of heads in multi-Head self-attention blocks.
            Default: 8.
        mlp_ratio (int, optional): The expansion ratio of FFN. Default: 4.
        num_keypoints: num_keypoints (int): Number of keypoints. Default: 17.
        seq_len: The sequence length. Default: 243.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout ratio of input. Default: 0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        att_fuse: Whether to fuse the results of attention blocks.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmpose.models import DSTFormer
        >>> import torch
        >>> self = DSTFormer(in_channels=3)
        >>> self.eval()
        >>> inputs = torch.rand(1, 2, 17, 3)
        >>> level_outputs = self.forward(inputs)
        >>> print(tuple(level_outputs.shape))
        (1, 2, 17, 512)
    """

    def __init__(self,
                 in_channels,
                 feat_size=256,
                 depth=5,
                 num_heads=8,
                 mlp_ratio=4,
                 num_keypoints=17,
                 seq_len=243,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 att_fuse=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.feat_size = feat_size

        self.joints_embed = nn.Linear(in_channels, feat_size)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        self.blocks_st = nn.ModuleList([
            AttentionBlock(
                dim=feat_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                st_mode='st') for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            AttentionBlock(
                dim=feat_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                st_mode='ts') for i in range(depth)
        ])

        self.norm = nn.LayerNorm(feat_size, eps=1e-06)

        self.temp_embed = nn.Parameter(torch.zeros(1, seq_len, 1, feat_size))
        self.spat_embed = nn.Parameter(
            torch.zeros(1, num_keypoints, feat_size))

        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.spat_embed, std=.02)

        self.att_fuse = att_fuse
        if self.att_fuse:
            self.attn_regress = nn.ModuleList(
                [nn.Linear(feat_size * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.attn_regress[i].weight.data.fill_(0)
                self.attn_regress[i].bias.data.fill_(0.5)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[None, :]
        assert len(x.shape) == 4

        B, F, K, C = x.shape
        x = x.reshape(-1, K, C)
        BF = x.shape[0]
        x = self.joints_embed(x)  # (BF, K, feat_size)
        x = x + self.spat_embed
        _, K, C = x.shape
        x = x.reshape(-1, F, K, C) + self.temp_embed[:, :F, :, :]
        x = x.reshape(BF, K, C)  # (BF, K, feat_size)
        x = self.pos_drop(x)

        for idx, (blk_st,
                  blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.attn_regress[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, K = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)  # (BF, K, feat_size)
        x = x.reshape(B, F, K, -1)
        return x

    def init_weights(self):
        """Initialize the weights in backbone."""
        super(DSTFormer, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
