# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
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
class PoseWarperNeck(nn.Module):
    """PoseWarper neck. Paper ref: Bertasius, Gedas, et al. "Learning temporal
    pose estimation from sparsely-labeled videos." arXiv preprint
    arXiv:1906.04016 (2019).

    <`https://arxiv.org/abs/1906.04016`>

    Args:
        in_channels (int): Number of input channels from backbone
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
        freeze_trans_layer (bool): Whether to freeze the transition layer
            (stop grad and set eval mode). Default: False.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        im2col_step: the argument `im2col_step` in deform_conv
        least_mmcv_version (str): should use version newer than this version
    """
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 in_channels,
                 out_channels,
                 inner_channels,
                 deform_groups=17,
                 dilations=(3, 6, 12, 18, 24),
                 extra=None,
                 res_blocks={},
                 offsets={},
                 deform_conv={},
                 in_index=0,
                 input_transform=None,
                 freeze_trans_layer=False,
                 norm_eval=False,
                 im2col_step=80,
                 least_mmcv_version='1.3.16'):
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
        self.freeze_trans_layer = freeze_trans_layer
        self.norm_eval = norm_eval
        self.im2col_step = im2col_step
        self.least_mmcv_version = least_mmcv_version

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        identity_trans_layer = False
        if extra is not None and 'trans_conv_kernel' in extra:
            assert extra['trans_conv_kernel'] in [0, 1, 3]
            if extra['trans_conv_kernel'] == 3:
                padding = 1
            elif extra['trans_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_trans_layer = True
            kernel_size = extra['trans_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_trans_layer:
            self.trans_layer = nn.Identity()
        else:
            self.trans_layer = build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

        # build chain of residual blocks
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
                bias=False),
            build_norm_layer(dict(type='BN'), inner_channels)[1])
        res_layers.append(
            block(
                in_channels=out_channels,
                out_channels=inner_channels,
                downsample=downsample))

        for _ in range(1, num_blocks):
            res_layers.append(block(inner_channels, inner_channels))
        self.offset_feats = nn.Sequential(*res_layers)

        # build offset layers
        self.num_offset_layers = len(dilations)
        assert self.num_offset_layers > 0, 'Number of offset layers ' \
            'should be larger than 0.'

        kernel = offsets.get('kernel', 3)
        target_offset_channels = 2 * kernel * kernel * deform_groups

        offset_layers = [
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=inner_channels,
                out_channels=target_offset_channels,
                kernel_size=kernel,
                stride=1,
                dilation=dilations[i],
                padding=dilations[i],
                bias=False,
            ) for i in range(self.num_offset_layers)
        ]
        self.offset_layers = nn.ModuleList(offset_layers)

        # build deformable conv layers
        assert mmcv.__version__ > self.least_mmcv_version, \
            f'Current MMCV version: {mmcv.__version__}, ' \
            f'but MMCV > {self.least_mmcv_version} is required, see ' \
            f'https://github.com/open-mmlab/mmcv/issues/1440, ' \
            f'Please install the latest MMCV.'

        kernel = deform_conv.get('kernel', 3)
        deform_conv_layers = [
            DeformConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=1,
                padding=int(kernel / 2) * dilations[i],
                dilation=dilations[i],
                deform_groups=deform_groups,
                im2col_step=self.im2col_step,
            ) for i in range(self.num_offset_layers)
        ]
        self.deform_conv_layers = nn.ModuleList(deform_conv_layers)

        self.freeze_layers()

    def freeze_layers(self):
        if self.freeze_trans_layer:
            self.trans_layer.eval()

            for param in self.trans_layer.parameters():
                param.requires_grad = False

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

    def forward(self, inputs, frame_weight):
        assert isinstance(inputs, (list, tuple)), 'PoseWarperNeck inputs ' \
            'should be list or tuple, even though the length is 1, ' \
            'for unified processing.'

        output_heatmap = 0
        if len(inputs) > 1:
            inputs = [self._transform_inputs(input) for input in inputs]
            inputs = [self.trans_layer(input) for input in inputs]

            # calculate difference features
            diff_features = [
                self.offset_feats(inputs[0] - input) for input in inputs
            ]

            for i in range(len(inputs)):
                if frame_weight[i] == 0:
                    continue
                warped_heatmap = 0
                for j in range(self.num_offset_layers):
                    offset = (self.offset_layers[j](diff_features[i]))
                    warped_heatmap_tmp = self.deform_conv_layers[j](inputs[i],
                                                                    offset)
                    warped_heatmap += warped_heatmap_tmp / \
                        self.num_offset_layers

                output_heatmap += warped_heatmap * frame_weight[i]

        else:
            inputs = inputs[0]
            inputs = self._transform_inputs(inputs)
            inputs = self.trans_layer(inputs)

            num_frames = len(frame_weight)
            batch_size = inputs.size(0) // num_frames
            ref_x = inputs[:batch_size]
            ref_x_tiled = ref_x.repeat(num_frames, 1, 1, 1)

            offset_features = self.offset_feats(ref_x_tiled - inputs)

            warped_heatmap = 0
            for j in range(self.num_offset_layers):
                offset = self.offset_layers[j](offset_features)

                warped_heatmap_tmp = self.deform_conv_layers[j](inputs, offset)
                warped_heatmap += warped_heatmap_tmp / self.num_offset_layers

            for i in range(num_frames):
                if frame_weight[i] == 0:
                    continue
                output_heatmap += warped_heatmap[i * batch_size:(i + 1) *
                                                 batch_size] * frame_weight[i]

        return output_heatmap

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self.freeze_layers()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
