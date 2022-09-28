# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, normal_init)

from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..backbones.resnet import BasicBlock
from ..builder import HEADS

try:
    from mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


class AdaptiveActivationBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):

        super(AdaptiveActivationBlock, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups

        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())

        self.transform_matrix_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=in_channels,
            out_channels=6 * groups,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=True)

        self.adapt_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=groups,
            deform_groups=groups)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        B, _, H, W = x.size()
        residual = x

        affine_matrix = self.transform_matrix_conv(x)
        affine_matrix = affine_matrix.view(B, 2, 3, self.groups, H, W)
        affine_matrix = affine_matrix.permute(0, 4, 5, 3, 1, 2).contiguous()
        offset = torch.matmul(affine_matrix, self.regular_matrix)
        offset = offset.view(B, H, W, self.groups * 18).permute(0, 3, 1, 2)

        x = self.adapt_conv(x, offset.contiguous())
        x = self.norm(x)
        x = self.act(x + residual)

        return x


@HEADS.register_module()
class OneStageHeatmapOffsetHead(nn.Module):
    """Simple deconv head.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 num_joints=17,
                 use_keypoint_heatmap=True,
                 num_heatmap_filters=32,
                 num_offset_filters=255,
                 use_adapt_act=False,
                 use_separate_regression=True,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 heatmap_loss_cfg=None,
                 offset_loss_cfg=None):
        super().__init__()

        self.in_channels = in_channels

        # set up input transform
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints * int(use_keypoint_heatmap),
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints if use_separate_regression else 1
        assert num_offset_filters % groups == 0
        if use_adapt_act:
            _block = AdaptiveActivationBlock
        else:
            _block = partial(
                ConvModule, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))

        self.offset_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_offset_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            _block(num_offset_filters, num_offset_filters, groups=groups),
            _block(num_offset_filters, num_offset_filters, groups=groups),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_offset_filters,
                out_channels=2 * num_joints,
                kernel_size=1,
                groups=groups))

        # set up losses
        self.heatmap_loss = build_loss(copy.deepcopy(heatmap_loss_cfg))
        self.offset_loss = build_loss(copy.deepcopy(offset_loss_cfg))

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate bottom-up masked mse loss.

        Note:
            - batch_size: N
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            targets (List(torch.Tensor[N,C,H,W])): Multi-scale targets.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale targets.
        """

        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.heatmap_loss(
                pred_heatmap, heatmaps[idx], masks[idx])
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        final_outputs = []
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        final_outputs.append((heatmap, offset))
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.offset_conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
