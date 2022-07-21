# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from torch import Tensor, nn

from mmpose.core.utils.tensor_utils import _to_numpy
from mmpose.core.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                      SampleList)
from mmpose.metrics.utils import pose_pck_accuracy
from mmpose.registry import KEYPOINT_CODECS, MODELS
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int): Number of channels in the input feature map
        out_channels (int): Number of channels in the output heatmap
        num_deconv_layers (int): Number of deconv layers. Defaults to 3
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple]): The kernel size of
            each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        input_transform (str): Transformation of input features which should
            be one of the following options:

                - ``'resize_concat'``: Resize multiple feature maps specified
                    by ``input_index`` to the same size as the first one and
                    concat these feature maps
                - ``'select'``: Select feature map(s) specified by
                    ``input_index``. Multiple selected features will be
                    bundled into a tuple

            Defaults to ``'select'``
        input_index (int | sequence[int]): The feature map index used in the
            input transformation. See also ``input_transform``. Defaults to 0
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 has_final_layer: bool = True,
                 input_transform: str = 'select',
                 input_index: Union[int, Sequence[int]] = 0,
                 align_corners: bool = False,
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Get model input channels according to feature
        in_channels = self._get_in_channels()

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'unmatched values {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got unmatched'
                    f' values {conv_out_channels} and {conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity

        if has_final_layer:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self._transform_inputs(feats)

        assert isinstance(x, Tensor), (
            'Selecting multiple features as the inputs is not supported in '
            f'{self.__class__.__name__}')

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x

    def predict(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
                test_cfg: ConfigType) -> SampleList:
        """Predict results from features."""
        heatmaps = self.forward(feats)
        preds = self.decode(heatmaps, batch_data_samples, test_cfg)
        return preds

    def loss(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
             train_cfg: ConfigType) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pred_heatmaps = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        target_weights = torch.cat(
            [d.gt_instance.target_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_heatmaps, gt_heatmaps, target_weights)
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses.update(loss_kpts=loss)

        # calculate accuracy
        _, avg_acc, _ = pose_pck_accuracy(
            output=_to_numpy(pred_heatmaps),
            target=_to_numpy(gt_heatmaps),
            mask=_to_numpy(target_weights).squeeze(-1) > 0)

        losses.update(acc_pose=float(avg_acc))

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """

        version = local_meta.get('version', None)
        if version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            v = state_dict.pop(_k)
            k = _k.lstrip(prefix)
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layerse, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weghts of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = _k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            elif k_parts[0] == 'loss':
                # loss.xxx -> loss_module.xxx
                k_new = 'loss_module.' + '.'.join(k_parts[1:])
            else:
                k_new = k

            state_dict[prefix + k_new] = v
