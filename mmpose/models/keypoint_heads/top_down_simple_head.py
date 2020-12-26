import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from mmpose.models.builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class TopDownSimpleHead(nn.Module):
    """Top-down model head of simple baseline paper ref: Bin Xiao. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopDownSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 loss_keypoint=None):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

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
                in_channels=num_deconv_filters[-1]
                if num_deconv_layers > 0 else in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

    def get_loss(self, output, target, target_weight, train_cfg):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_outputs: O
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW] | torch.Tensor[NxOxKxHxW]):
                Output heatmaps.
            target (torch.Tensor[NxKxHxW] | torch.Tensor[NxOxKxHxW]):
                Target heatmaps.
            target_weight (torch.Tensor[NxKx1] | torch.Tensor[NxOxKx1]):
                Weights across different joint types.
            train_cfg (dict): Config for training. Default: None.
        """

        losses = dict()

        if isinstance(output, list):
            if target.dim() == 5 and target_weight.dim() == 4:
                # target: [N, O, K, H, W]
                # target_weight: [N, O, K, 1]
                assert target.size(1) == len(output)
            if isinstance(self.loss, nn.Sequential):
                assert len(self.loss) == len(output)
            if 'loss_weights' in train_cfg and train_cfg[
                    'loss_weights'] is not None:
                assert len(train_cfg['loss_weights']) == len(output)
            for i in range(len(output)):
                if target.dim() == 5 and target_weight.dim() == 4:
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                else:
                    target_i = target
                    target_weight_i = target_weight
                if isinstance(self.loss, nn.Sequential):
                    loss_func = self.loss[i]
                else:
                    loss_func = self.loss

                loss_i = loss_func(output[i], target_i, target_weight_i)
                if 'loss_weights' in train_cfg and train_cfg['loss_weights']:
                    loss_i = loss_i * train_cfg['loss_weights'][i]
                if 'mse_loss' not in losses:
                    losses['mse_loss'] = loss_i
                else:
                    losses['mse_loss'] += loss_i
        else:
            assert not isinstance(self.loss, nn.Sequential)
            assert target.dim() == 4 and target_weight.dim() == 3
            # target: [N, K, H, W]
            # target_weight: [N, K, 1]
            losses['mse_loss'] = self.loss(output, target, target_weight)

        return losses

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
