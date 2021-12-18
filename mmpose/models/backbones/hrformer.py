# Copyright (c) OpenMMLab. All rights reserved.

import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn import functional as F

from ..builder import BACKBONES
from .hrnet import BasicBlock, Bottleneck, HRModule, HRNet


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn  # + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        # query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        query = F.pad(query, (0, 0, pad_r // 2, pad_r - pad_r // 2, pad_b // 2,
                              pad_b - pad_b // 2))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            # x = x[:, :H, :W, :].contiguous()
            x = x[:, pad_b // 2:pad_b // 2 + H,
                  pad_r // 2:pad_r // 2 + W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

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


class MlpDWBN(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_cfg=dict(type='GELU', inplace=True),
            dw_act_cfg=dict(type='GELU', inplace=True),
            drop=0.0,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            dropout_layer=dict(type='DropPath', drop_prob=0.),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = build_conv_layer(
            conv_cfg,
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act1 = build_activation_layer(act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, hidden_channels)[1]
        self.dw3x3 = build_conv_layer(
            conv_cfg,
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels,
        )
        self.act2 = build_activation_layer(dw_act_cfg)

        self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        self.fc2 = build_conv_layer(
            conv_cfg,
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act3 = build_activation_layer(act_cfg)
        self.norm3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.drop = nn.Dropout(drop, inplace=True)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        x_ = self.fc1(x_)
        x_ = self.norm1(x_)
        x_ = self.act1(x_)
        x_ = self.dw3x3(x_)
        x_ = self.norm2(x_)
        x_ = self.act2(x_)
        x_ = self.drop(x_)
        x_ = self.fc2(x_)
        x_ = self.norm3(x_)
        x_ = self.act3(x_)
        x_ = self.drop(x_)

        x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        return self.dropout_layer(x_)


class LocalWindowTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            num_heads,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_cfg=dict(type='GELU'),
            dw_act_cfg=None,
            norm_cfg=dict(type='LN', eps=1e-6),
            conv_cfg=None,
            mlp_norm_cfg=dict(type='BN', requires_grad=True),
    ):
        super().__init__()
        self.dim = in_channels
        self.out_dim = out_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.attn = ShiftWindowMSA(
            self.dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop_rate=attn_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path),
        )
        self.norm1 = build_norm_layer(norm_cfg, self.dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, self.out_dim)[1]

        mlp_hidden_dim = int(self.dim * mlp_ratio)
        dw_act_cfg = dw_act_cfg or act_cfg
        self.mlp = MlpDWBN(
            in_channels=self.dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=self.out_dim,
            act_cfg=act_cfg,
            dw_act_cfg=dw_act_cfg,
            drop=drop,
            conv_cfg=conv_cfg,
            norm_cfg=mlp_norm_cfg,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # reshape
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        # Attention
        x = x + self.attn(self.norm1(x), (H, W))
        # FFN
        x = x + self.mlp(self.norm2(x), (H, W))
        # reshape
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x


class HRTransformerModule(HRModule):

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 num_heads=None,
                 num_window_sizes=None,
                 num_mlp_ratios=None,
                 drop_paths=0.0,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):

        super(HRModule, self).__init__()
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.upsample_cfg = upsample_cfg
        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.drop_paths = drop_paths

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        """Make one branch."""
        # LocalWindowTransformerBlock does not support down sample layer yet.
        assert stride == 1 and self.in_channels[branch_index] == num_channels[
            branch_index]
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    mlp_norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                ))
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths,
                ))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners']),
                        ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1],
                                ))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True),
                                ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)


@BACKBONES.register_module()
class HRFormer(HRNet):
    """HRFormer backbone.

    High Resolution Transformer Backbone
    """

    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck,
        'TRANSFORMER_BLOCK': LocalWindowTransformerBlock
    }

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 frozen_stages=-1):
        # generate drop path rate list
        depth_s2 = (
            extra['stage2']['num_blocks'][0] * extra['stage2']['num_modules'])
        depth_s3 = (
            extra['stage3']['num_blocks'][0] * extra['stage3']['num_modules'])
        depth_s4 = (
            extra['stage4']['num_blocks'][0] * extra['stage4']['num_modules'])
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = extra['drop_path_rate']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]

        upsample_cfg = extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })
        extra['upsample'] = upsample_cfg

        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, zero_init_residual, frozen_stages)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['num_window_sizes']
        num_mlp_ratios = layer_config['num_mlp_ratios']
        drop_path_rates = layer_config['drop_path_rates']

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg,
                    num_heads=num_heads,
                    num_window_sizes=num_window_sizes,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_path_rates[num_blocks[0] *
                                               i:num_blocks[0] * (i + 1)],
                ))

        return nn.Sequential(*hr_modules), in_channels
