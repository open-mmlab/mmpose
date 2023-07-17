import math
from collections import OrderedDict
from functools import partial
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import checkpoint, load_checkpoint
from mmengine.utils import to_2tuple

from mmpose.models.backbones.base_backbone import BaseBackbone
from mmpose.registry import MODELS
from mmpose.utils import get_root_logger


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x))
        x = self.drop(x)
        return x


class CMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop_rate=0.,
                 init_cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        return x


class CBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__()
        self.pos_embed = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            padding=1,
            groups=embed_dims)
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.conv1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.conv2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.attn = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=5,
            padding=2,
            groups=embed_dims)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.cffn = CMlp(
            embed_dims, mlp_hidden_dim, act_cfg=act_cfg, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.cffn(self.norm2(x)))
        return x


class Attention(nn.Module):

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.,
        proj_drop_rate=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dims=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dims)
        self.proj = nn.Conv2d(
            in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class MSA(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__()
        self.pos_embed = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            padding=1,
            groups=embed_dims)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(embed_dims, num_heads, qkv_bias, qk_scale,
                              attn_drop_rate, proj_drop_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class WindowMSA(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=14,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__()
        self.windows_size = window_size
        self.pos_embed = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            padding=1,
            groups=embed_dims)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(embed_dims, num_heads, qkv_bias, qk_scale,
                              attn_drop_rate, proj_drop_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_dropout(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop_rate=drop_rate)

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H_pad, W_pad, _ = x.shape

        x_windows = self.window_partition(
            x)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        x = self.window_reverse(attn_windows, H_pad, W_pad)  # B H' W' C

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2).reshape(B, C, H, W)
        return x


@MODELS.register_module()
class UniFormer(BaseBackbone):
    """ Vision Transformer
    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929

    Args:
        depths (tuple[int]): number of block in each layer
        img_size (int, tuple): input image size. Default: 224.
        in_channels (int): number of input channels. Default: 3.
        num_classes (int): number of classes for classification head. Default 80.
        embed_dims (tuple[int]): embedding dimension. Default to [64, 128, 320, 512].
        head_dim (int): dimension of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): if True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
        representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
        drop_rate (float): dropout rate. Default: 0.
        attn_drop_rate (float): attention dropout rate. Default: 0.
        drop_path_rate (float): stochastic depth rate. Default: 0.
        norm_layer (nn.Module): normalization layer
        pretrained (str, optional): model pretrained path. Default: None.
        use_checkpoint (bool): whether use torch.utils.checkpoint
        checkpoint_num (list): index for using checkpoint in every stage
        use_windows (bool): whether use window MHRA
        use_hybrid (bool): whether use hybrid MHRA
        window_size (int): size of window (>14). Default: 14.
        init_cfg (dict or list[dict], optional)
    """

    def __init__(self,
                 depths=(3, 4, 8, 3),
                 img_size=224,
                 in_channels=3,
                 num_classes=80,
                 embed_dims=(64, 128, 320, 512),
                 head_dim=64,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 use_checkpoint=False,
                 checkpoint_num=[0, 0, 0, 0],
                 use_window=False,
                 use_hybrid=False,
                 window_size=14,
                 init_cfg=[dict(type='Pretrained', checkpoint='/root/mmpose/projects/uniformer/uniformer_image/uniformer_base_in1k.pth')]):         
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.use_window = use_window
        # print(f'Use Checkpoint: {self.use_checkpoint}')
        # print(f'Checkpoint Number: {self.checkpoint_num}')
        self.logger = get_root_logger()
        self.logger.info(
            f'Use torch.utils.checkpoint: {self.use_checkpoint}, checkpoint number: {self.checkpoint_num}'
        )
        self.num_features = self.embed_dims = embed_dims  # num_features for consistency with other models
        norm_cfg = norm_cfg or dict(type='LN', eps=1e-6)

        self.patch_embed1 = PatchEmbed(img_size=img_size,
                                       patch_size=4, in_channels=in_channels, embed_dims=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, 
                                       patch_size=2, in_channels=embed_dims[0], embed_dims=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8,
                                       patch_size=2, in_channels=embed_dims[1], embed_dims=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16,
                                       patch_size=2, in_channels=embed_dims[2], embed_dims=embed_dims[3])

        self.drop_after_pos = nn.Dropout(drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dims]
        self.blocks1 = nn.ModuleList([
            CBlock(
                embed_dims=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg) for i in range(depths[0])
        ])
        self.norm1 = build_norm_layer(norm_cfg, embed_dims[0])[1]
        self.blocks2 = nn.ModuleList([
            CBlock(
                embed_dims=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i + depths[0]],
                norm_cfg=norm_cfg) for i in range(depths[1])
        ])
        self.norm2 = build_norm_layer(norm_cfg, embed_dims[1])[1]
        if self.use_window:
            self.logger.info('Use local window for all blocks in stage3')
            self.blocks3 = nn.ModuleList([
                WindowMSA(
                    embed_dims=embed_dims[2],
                    num_heads=num_heads[2],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i + depths[0] + depths[1]],
                    norm_cfg=norm_cfg) for i in range(depths[2])
            ])
        elif use_hybrid:
            self.logger.info('Use hybrid window for blocks in stage3')
            block3 = []
            for i in range(depths[2]):
                if (i + 1) % 4 == 0:
                    block3.append(
                        MSA(
                            diembed_dimsm=embed_dims[2],
                            num_heads=num_heads[2],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[i + depths[0] + depths[1]],
                            norm_cfg=norm_cfg))
                else:
                    block3.append(
                        MSA(
                            embed_dims=embed_dims[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[i + depths[0] + depths[1]],
                            norm_cfg=norm_cfg))
            self.blocks3 = nn.ModuleList(block3)
        else:
            self.logger.info('Use global window for all blocks in stage3')
            self.blocks3 = nn.ModuleList([
                MSA(
                    embed_dims=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i + depths[0] + depths[1]],
                    norm_cfg=norm_cfg) for i in range(depths[2])
            ])
        self.norm3 = build_norm_layer(norm_cfg, embed_dims[2])[1]
        self.blocks4 = nn.ModuleList([
            MSA(
                embed_dims=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i + depths[0] + depths[1] + depths[2]],
                norm_cfg=norm_cfg) for i in range(depths[3])
        ])
        self.norm4 = build_norm_layer(norm_cfg, embed_dims[3])[1]

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(embed_dims,
                                              representation_size)),
                             ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        self.apply(self._init_weights)
        self.init_weights(init_cfg=init_cfg)

    def init_weights(self, init_cfg):
        if (isinstance(self.init_cfg, dict) and self.init_cfg['type']=='Pretrained'):
            pretrained_path = init_cfg['checkpoint']
            load_checkpoint(
                self,
                pretrained_path,
                map_location='cpu',
                strict=False,
                logger=self.logger)
            self.logger.info(f'Load pretrained model from: {pretrained_path}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dims,
            num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        out = []
        x = self.patch_embed1(x)
        x = self.drop_after_pos(x)
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_out = self.norm1(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed2(x)
        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_out = self.norm2(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed3(x)
        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed4(x)
        for i, blk in enumerate(self.blocks4):
            if self.use_checkpoint and i < self.checkpoint_num[3]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        return tuple(out)
