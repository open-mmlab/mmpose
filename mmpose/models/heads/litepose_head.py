# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.models.backbones.litepose import SepConv2d
from mmpose.models.builder import HEADS


@HEADS.register_module()
class LitePoseHead(nn.Module):

    def __init__(self, deconv_setting, num_deconv_layers, num_deconv_kernels,
                 num_joints, tag_per_joint, with_heatmaps_loss, with_ae_loss,
                 channels):
        super().__init__()
        self.deconv_setting = deconv_setting
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_kernels = num_deconv_kernels
        self.num_joints = num_joints
        self.tag_per_joint = tag_per_joint
        self.with_heatmaps_loss = with_heatmaps_loss
        self.with_ae_loss = with_ae_loss
        self.channel = channels
        self.filters = deconv_setting
        self.inplanes = self.channel[-1]
        self.deconv_refined, self.deconv_raw, self.deconv_bnrelu = \
            self._make_deconv_layers(
                self.num_deconv_layers,
                self.filters,
                self.num_deconv_kernels,
            )
        self.final_refined, self.final_raw, self.final_channel = \
            self._make_final_layers(self.filters)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_final_layers(self, num_filters):
        dim_tag = self.num_joints if self.tag_per_joint else 1
        final_raw = []
        final_refined = []
        final_channel = []
        for i in range(1, self.num_deconv_layers):
            # input_channels = num_filters[i] + self.channel[-i-3]
            oup_joint = self.num_joints if self.with_heatmaps_loss[i -
                                                                   1] else 0
            oup_tag = dim_tag if self.with_ae_loss[i - 1] else 0
            final_refined.append(
                SepConv2d(num_filters[i], oup_joint + oup_tag, ker=5))
            final_raw.append(
                SepConv2d(self.channel[-i - 3], oup_joint + oup_tag, ker=5))
            final_channel.append(oup_joint + oup_tag)

        return nn.ModuleList(final_refined), nn.ModuleList(
            final_raw), final_channel

    def _make_deconv_layers(self, num_layers, num_filters, num_kernels):
        deconv_refined = []
        deconv_raw = []
        deconv_bnrelu = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            # inplanes = self.inplanes + self.channel[-i-2]
            layers = []
            deconv_refined.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            deconv_raw.append(
                nn.ConvTranspose2d(
                    in_channels=self.channel[-i - 2],
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            deconv_bnrelu.append(nn.Sequential(*layers))

        return nn.ModuleList(deconv_refined), nn.ModuleList(
            deconv_raw), nn.ModuleList(deconv_bnrelu)

    def forward(self, x_list):
        final_outputs = []
        input_refined = x_list[-1]
        input_raw = x_list[-2]
        for i in range(self.num_deconv_layers):
            next_input_refined = self.deconv_refined[i](input_refined)
            next_input_raw = self.deconv_raw[i](input_raw)
            input_refined = self.deconv_bnrelu[i](
                next_input_refined + next_input_raw)
            input_raw = x_list[-i - 3]
            if i > 0:
                final_refined = self.final_refined[i - 1](input_refined)
                final_raw = self.final_raw[i - 1](input_raw)
                final_outputs.append(final_refined + final_raw)
        return final_outputs
