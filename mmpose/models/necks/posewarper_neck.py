# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from mmcv.ops import DeformConv2d
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.utils.ops import resize
from ..backbones.resnet import BasicBlock, Bottleneck
from ..builder import NECKS


@NECKS.register_module()
class PoseWarper(nn.Module):
    """PoseWarper neck. Paper ref: Bertasius, Gedas, et al. "Learning temporal
    pose estimation from sparsely-labeled videos." arXiv preprint
    arXiv:1906.04016 (2019).

    <`https://arxiv.org/abs/1906.04016`>

    Args:
        in_channels (int): Number of intput channels from backbone
        out_channels (int): Number of output channels
        inner_channels (int): Number of intermediate channels
        deform_groups (int): Number of groups in the deformable conv
        dilations (list|tuple): different dilations of the offset conv layers
        extra (dict): config of the conv layer to get heatmap
        res_blocks (dict): config of residual blocks
        offsets (dict): config of offset conv layer
        deform_conv (dict): config of defomrable conv layer
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
    """
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 in_channels,
                 out_channels,
                 inner_channels,
                 deform_groups,
                 dilations=(3, 6, 12, 18, 24),
                 extra=None,
                 res_blocks=None,
                 offsets=None,
                 deform_conv=None,
                 in_index=0,
                 input_transform=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.deform_groups = deform_groups
        self.dilations = dilations
        self.extra = extra
        self.res_blocks = res_blocks
        self.offsets = offsets
        self.deform_conv = deform_conv
        self.in_index = in_index
        self.input_transform = input_transform

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            self.final_layer = build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

        # bulid chain of residual blocks
        block_type = self.res_blocks.get('block', 'BASIC')
        block = self.blocks_dict[block_type]
        num_blocks = self.res_blocks.get('num_blocks', 20)

        res_layers = []
        downsample = nn.Sequential(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=out_channels,
                out_channels=inner_channels,
                kernel_size=1,
                stride=1,
                bias=False), build_norm_layer(dict(type='BN'), inner_channels))
        res_layers.append(
            block(
                in_channels=out_channels,
                out_channels=inner_channels,
                downsample=downsample))

        for _ in range(1, num_blocks):
            res_layers.append(block(inner_channels, inner_channels))
        self.offset_feats = nn.Sequential(*res_layers)

        # bulid offset layers
        num_offset_layers = len(dilations)
        kernel = offsets.get('kernel', 3)
        target_offset_channels = 2 * kernel * kernel * deform_groups

        offset_layers = []
        for i in range(num_offset_layers):
            offset_layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=inner_channels,
                    out_channels=target_offset_channels,
                    kernel_size=kernel,
                    stride=1,
                    dilation=dilations[i],
                    padding=dilations[i],
                    bias=False,
                ))
        self.offset_layers = nn.ModuleList(offset_layers)

        # bulid deformable conv layers
        kernel = deform_conv.get('kernel', 3)

        deform_conv_layers = []
        for i in range(num_offset_layers):
            deform_conv_layers.append(
                DeformConv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=int(kernel) / 2 * dilations[i],
                    dilation=dilations[i],
                    deform_groups=deform_groups,
                ))
        self.deform_conv_layers = nn.ModuleList(deform_conv_layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, DeformConv2d):
                filler = torch.zeros([
                    m.weight.size(0),
                    m.weight.size(1),
                    m.weight.size(2),
                    m.weight.size(3)
                ],
                                     dtype=torch.float32,
                                     device=m.weight.device)
                for k in range(m.weight.size(0)):
                    filler[k, k,
                           int(m.weight.size(2) / 2),
                           int(m.weight.size(3) / 2)] = 1.0
                m.weight = torch.nn.Parameter(filler)
                m.weight.requires_grad = True

        # posewarper offset layer weight initialization
        for m in self.offset_layers.modules():
            constant_init(m, 0)

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

    def forward(self, inputs):
        # inputs (list): each element of the list is a batch of frame data
        num_frames = len(inputs)
        num_offset_layers = len(self.offset_layers)

        # batch_size, _, heatmap_height, heatmap_width = inputs[0].shape
        for i in range(num_frames):
            inputs[i] = self._transform_inputs(inputs[i])
            inputs[i] = self.final_layer(inputs[i])

        ref_feature = inputs[0]
        # calculate difference features
        if num_frames == 2:  # train mode, input two frames
            diff_feature = ref_feature - inputs[1]
            diff_feature = self.offset_feats(diff_feature)

            warped_heatmap = 0
            for j in range(num_offset_layers):
                offset = (self.offset_layers[j](diff_feature))
                warped_heatmap_tmp = self.deform_conv_layers[j](ref_feature,
                                                                offset)
                warped_heatmap += warped_heatmap_tmp / num_offset_layers

            return warped_heatmap

        else:  # test mode, multi frames
            if num_frames == 5:
                weight_frame = [0.3, 0.25, 0.25, 0.1, 0.1]
            elif num_frames == 3:
                weight_frame = [0.4, 0.3, 0.3]
            elif num_frames == 7:
                weight_frame = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
            elif num_frames % 2 == 0:
                raise ValueError('Number of frames must be odd number')
            else:
                weight_frame = [1.0 / num_frames for _ in range(num_frames)]

            diff_features = []
            for i in range(num_frames):
                diff_feature = ref_feature - inputs[i]
                diff_features.append(self.offset_feats(diff_feature))

            output_heatmap = 0
            for i in range(num_frames):
                warped_heatmap = 0
                for j in range(num_offset_layers):
                    offset = (self.offset_layers[j](diff_features[i]))
                    warped_heatmap_tmp = self.deform_conv_layers[j](
                        ref_feature, offset)
                    warped_heatmap += warped_heatmap_tmp / num_offset_layers

                output_heatmap += warped_heatmap * weight_frame[i]

            return output_heatmap
