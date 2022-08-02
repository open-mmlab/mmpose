# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from mmpose.core.utils.tensor_utils import to_numpy
from mmpose.core.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                      SampleList)
from mmpose.metrics.utils import simcc_pck_accuracy
from mmpose.registry import KEYPOINT_CODECS, MODELS
from .. import HeatmapHead
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SimCCHead(BaseHead):
    """Top-down heatmap head introduced in `SimCC`_ by Li et al (2022). The
    head is composed of a few deconvolutional layers followed by a fully-
    connected layer to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        input_size (tuple): Input image size in shape [w, h]
        heatmap_size (tuple): Size of the output heatmap
        simcc_split_ratio (float): Split ratio of pixels
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
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
            input transformation. See also ``input_transform``. Defaults to -1
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`SimCC`: https://arxiv.org/abs/2107.03332
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        has_final_layer: bool = True,
        input_transform: str = 'select',
        input_index: Union[int, Sequence[int]] = -1,
        align_corners: bool = False,
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple(
                [s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.simplebaseline_head = HeatmapHead(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                has_final_layer=has_final_layer,
                input_transform=input_transform,
                input_index=input_index,
                align_corners=align_corners)

            if has_final_layer:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]
        else:
            self.simplebaseline_head = None
            in_channels = self._get_in_channels()

            if self.input_transform == 'resize_concat':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[0]
            elif self.input_transform == 'select':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[input_index]

        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        x = self._transform_inputs(feats)

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        # flatten the output heatmap
        x = torch.flatten(x, 2)

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> SampleList:
        """Predict results from features."""

        batch_pred_x, batch_pred_y = self.forward(feats)
        preds = self.decode((batch_pred_x, batch_pred_y), batch_data_samples,
                            test_cfg)

        # Whether to visualize the predicted simcc representations
        if test_cfg.get('output_heatmaps', False):
            for pred_x, pred_y, data_sample in zip(batch_pred_x, batch_pred_y,
                                                   preds):
                # Store the simcc predictions in the data sample
                data_sample.pred_instances.simcc_x = pred_x[None, :]
                data_sample.pred_instances.simcc_y = pred_y[None, :]

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([d.gt_instances.simcc_x for d in batch_data_samples],
                         dim=0)
        gt_y = torch.cat([d.gt_instances.simcc_y for d in batch_data_samples],
                         dim=0)
        target_coords = torch.cat(
            [d.gt_instances.keypoints for d in batch_data_samples], dim=0)
        keypoint_weights = torch.cat(
            [d.gt_instances.keypoint_weights for d in batch_data_samples],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(target_coords),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        losses.update(acc_pose=float(avg_acc))

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
