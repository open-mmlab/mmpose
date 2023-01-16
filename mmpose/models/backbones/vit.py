# Copyright (c) OpenMMLab. All rights reserved.
# Modified from the ViT implementation in mmseg
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.transformer import PatchEmbed
from .base_backbone import BaseBackbone


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
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
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
    """

    def __init__(
        self,
        img_size=(256, 192),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        num_fcs=2,
        with_cp=False,
        final_norm=False,
    ):
        super(VisionTransformer, self).__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.with_cp = with_cp

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=None,
            init_cfg=None)

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            _, self.last_norm = build_norm_layer(norm_cfg, embed_dims)
        else:
            self.last_norm = nn.Identity()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # convert the weights from MAE to mmcls
            new_ckpt = OrderedDict()

            for k, v in state_dict.items():
                if k.startswith('blocks'):
                    new_key = k.replace('blocks', 'layers')
                    if 'norm' in new_key:
                        new_key = new_key.replace('norm', 'ln')
                    elif 'mlp.fc1' in new_key:
                        new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
                    elif 'mlp.fc2' in new_key:
                        new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
                    elif 'attn.qkv' in new_key:
                        new_key = new_key.replace('attn.qkv.',
                                                  'attn.attn.in_proj_')
                    elif 'attn.proj' in new_key:
                        new_key = new_key.replace('attn.proj',
                                                  'attn.attn.out_proj')
                    new_ckpt[new_key] = v
                elif k.startswith('patch_embed'):
                    new_key = k.replace('patch_embed.proj',
                                        'patch_embed.projection')
                    new_ckpt[new_key] = v
                else:
                    new_key = k
                    new_ckpt[new_key] = v

            state_dict = new_ckpt

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size))

            self.load_state_dict(state_dict, False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _pos_embeding(self, patched_img, hw_shape, pos_embed, cls_token=False):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if not cls_token:
            if x_len != (pos_len - 1):
                if pos_len == (self.img_size[0] // self.patch_size) * (
                        self.img_size[1] // self.patch_size):
                    pos_h = self.img_size[0] // self.patch_size
                    pos_w = self.img_size[1] // self.patch_size
                else:
                    raise ValueError(
                        'Unexpected shape of pos_embed, got {}.'.format(
                            pos_embed.shape))
                pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                                  (pos_h, pos_w))
            return self.drop_after_pos(patched_img + pos_embed[:, 1:] +
                                       pos_embed[:, 0:1])
        else:
            if x_len != pos_len:
                if pos_len == (self.img_size[0] // self.patch_size) * (
                        self.img_size[1] // self.patch_size):
                    pos_h = self.img_size[0] // self.patch_size
                    pos_w = self.img_size[1] // self.patch_size
                else:
                    raise ValueError(
                        'Unexpected shape of pos_embed, got {}.'.format(
                            pos_embed.shape))
                pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                                  (pos_h, pos_w))
            return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for up sampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=input_shpae,
            align_corners=False,
            mode='bicubic')
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, x):
        B = x.shape[0]

        x, hw_shape = self.patch_embed(x)
        x = self._pos_embeding(x, hw_shape, self.pos_embed, cls_token=False)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.last_norm(x)
            if i in self.out_indices:
                out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
