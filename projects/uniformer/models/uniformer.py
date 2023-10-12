from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import checkpoint, load_checkpoint
from mmengine.utils import to_2tuple

from mmpose.models.backbones.base_backbone import BaseBackbone
from mmpose.registry import MODELS
from mmpose.utils import get_root_logger


class Mlp(BaseModule):
    """Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
            Defaults to None.
        out_features (int): Number of output features.
            Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x))
        x = self.drop(x)
        return x


class CMlp(BaseModule):
    """Multilayer perceptron via convolution.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
            Defaults to None.
        out_features (int): Number of output features.
            Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x))
        x = self.drop(x)
        return x


class CBlock(BaseModule):
    """Convolution Block.

    Args:
        embed_dim (int): Number of input features.
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        drop (float): Dropout rate.
            Defaults to 0.0.
        drop_paths (float): Stochastic depth rates.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim: int,
                 mlp_ratio: float = 4.,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pos_embed = nn.Conv2d(
            embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 1)
        self.attn = nn.Conv2d(
            embed_dim, embed_dim, 5, padding=2, groups=embed_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is
        # better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(BaseModule):
    """Self-Attention.

    Args:
        embed_dim (int): Number of input features.
        num_heads (int): Number of attention heads.
            Defaults to 8.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Attention dropout rate.
            Defaults to 0.0.
        proj_drop_rate (float): Dropout rate.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop_rate: float = 0.,
                 proj_drop_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually
        # to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        img_size (int): Number of input size.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        in_channels (int): Number of input features.
            Defaults to 3.
        embed_dims (int): Number of output features.
            Defaults to 768.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class SABlock(BaseModule):
    """Self-Attention Block.

    Args:
        embed_dim (int): Number of input features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float): Stochastic depth rates.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pos_embed = nn.Conv2d(
            embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class WindowSABlock(BaseModule):
    """Self-Attention Block.

    Args:
        embed_dim (int): Number of input features.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the partition window. Defaults to 14.
        mlp_ratio (float): Ratio of mlp hidden dimension
            to embedding dimension. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float): Stochastic depth rates.
            Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: int = 14,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.windows_size = window_size
        self.pos_embed = nn.Conv2d(
            embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        # self.norm2 = build_dropout(norm_cfg, embed_dims)[1]
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
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
        windows = x.permute(0, 1, 3, 2, 4,
                            5).contiguous().view(-1, window_size, window_size,
                                                 C)
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
    """The implementation of Uniformer with downstream pose estimation task.

    UniFormer: Unifying Convolution and Self-attention for Visual Recognition
      https://arxiv.org/abs/2201.09450
    UniFormer: Unified Transformer for Efficient Spatiotemporal Representation
      Learning https://arxiv.org/abs/2201.04676

    Args:
        depths (List[int]): number of block in each layer.
            Default to [3, 4, 8, 3].
        img_size (int, tuple): input image size. Default: 224.
        in_channels (int): number of input channels. Default: 3.
        num_classes (int): number of classes for classification head. Default
            to 80.
        embed_dims (List[int]): embedding dimensions.
            Default to [64, 128, 320, 512].
        head_dim (int): dimension of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): if True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        representation_size (Optional[int]): enable and set representation
            layer (pre-logits) to this value if set
        drop_rate (float): dropout rate. Default: 0.
        attn_drop_rate (float): attention dropout rate. Default: 0.
        drop_path_rate (float): stochastic depth rate. Default: 0.
        norm_layer (nn.Module): normalization layer
        use_checkpoint (bool): whether use torch.utils.checkpoint
        checkpoint_num (list): index for using checkpoint in every stage
        use_windows (bool): whether use window MHRA
        use_hybrid (bool): whether use hybrid MHRA
        window_size (int): size of window (>14). Default: 14.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        depths: List[int] = [3, 4, 8, 3],
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 80,
        embed_dims: List[int] = [64, 128, 320, 512],
        head_dim: int = 64,
        mlp_ratio: int = 4.,
        qkv_bias: bool = True,
        qk_scale: float = None,
        representation_size: Optional[int] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        checkpoint_num=(0, 0, 0, 0),
        use_window: bool = False,
        use_hybrid: bool = False,
        window_size: int = 14,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super(UniFormer, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.use_window = use_window
        self.logger = get_root_logger()
        self.logger.info(f'Use torch.utils.checkpoint: {self.use_checkpoint}')
        self.logger.info(
            f'torch.utils.checkpoint number: {self.checkpoint_num}')
        self.num_features = self.embed_dims = embed_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3])

        self.drop_after_pos = nn.Dropout(drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dims]
        self.blocks1 = nn.ModuleList([
            CBlock(
                embed_dim=embed_dims[0],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        self.blocks2 = nn.ModuleList([
            CBlock(
                embed_dim=embed_dims[1],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i + depths[0]]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        if self.use_window:
            self.logger.info('Use local window for all blocks in stage3')
            self.blocks3 = nn.ModuleList([
                WindowSABlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i + depths[0] + depths[1]])
                for i in range(depths[2])
            ])
        elif use_hybrid:
            self.logger.info('Use hybrid window for blocks in stage3')
            block3 = []
            for i in range(depths[2]):
                if (i + 1) % 4 == 0:
                    block3.append(
                        SABlock(
                            embed_dim=embed_dims[2],
                            num_heads=num_heads[2],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[i + depths[0] + depths[1]]))
                else:
                    block3.append(
                        WindowSABlock(
                            embed_dim=embed_dims[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[i + depths[0] + depths[1]]))
            self.blocks3 = nn.ModuleList(block3)
        else:
            self.logger.info('Use global window for all blocks in stage3')
            self.blocks3 = nn.ModuleList([
                SABlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i + depths[0] + depths[1]])
                for i in range(depths[2])
            ])
        self.norm3 = norm_layer(embed_dims[2])
        self.blocks4 = nn.ModuleList([
            SABlock(
                embed_dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i + depths[0] + depths[1] + depths[2]])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

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
        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            pretrained = self.init_cfg['checkpoint']
            load_checkpoint(
                self,
                pretrained,
                map_location='cpu',
                strict=False,
                logger=self.logger)
            self.logger.info(f'Load pretrained model from {pretrained}')

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
