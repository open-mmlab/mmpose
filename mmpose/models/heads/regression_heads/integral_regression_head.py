# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.data import PixelData
from torch import Tensor, nn

from mmpose.core.utils.tensor_utils import to_numpy
from mmpose.core.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                      SampleList,)
from mmpose.metrics.utils import keypoint_pck_accuracy
from mmpose.registry import KEYPOINT_CODECS, MODELS
from .. import HeatmapHead
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class IntegralRegressionHead(BaseHead):
    """Top-down integral regression head introduced in `DSNT`_ by Nibali et
    al(2018). The head contains a differentiable spatial to numerical transform
    (DSNT) layer that do soft-argmax operation on the predicted heatmaps to
    regress the coordinates.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        in_featuremap_size (int | sequence[int]): Size of input feature map
        num_joints (int): Number of joints
        out_sigma (bool): Predict the sigma (the variance of the joint
            location) together with the joint location. Introduced in `RLE`_
            by Li et al(2021). Default: False
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`SmoothL1Loss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`DSNT`: https://arxiv.org/abs/1801.07372
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 has_final_layer: bool = True,
                 input_transform: str = 'select',
                 input_index: Union[int, Sequence[int]] = 0,
                 align_corners: bool = False,
                 loss: ConfigType = dict(
                     type='SmoothL1Loss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_joints = num_joints
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
            self.heatmap_size = tuple([s * (2**num_deconv)
                                       for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.simplebaseline_head = HeatmapHead(
                in_channels=in_channels,
                out_channels=num_joints,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                has_final_layer=has_final_layer,
                input_transform=input_transform,
                input_index=input_index,
                align_corners=align_corners
            )

            if has_final_layer:
                in_channels = num_joints
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

        W, H = self.heatmap_size
        self.linspace_x = torch.arange(0.0, 1.0 * H, 1).reshape(H, 1).repeat(
            [1, W]) / H
        self.linspace_y = torch.arange(0.0, 1.0 * W, 1).reshape(1, W).repeat(
            [H, 1]) / W

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

    def _linear_expectation(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""

        N, C, _, _ = heatmaps.shape
        heatmaps = heatmaps.mul(linspace).reshape(N, C, -1)
        expectation = torch.sum(heatmaps, dim=2, keepdim=True)

        return expectation

    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""

        _, C, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, C, H * W)
        heatmaps = F.softmax(featmaps, dim=2)

        return heatmaps.reshape(-1, C, H, W)

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        if self.simplebaseline_head is None:
            feats = self._transform_inputs(feats)
        else:
            feats = self.simplebaseline_head(feats)

        heatmaps = self._flat_softmax(feats)

        pred_x = self._linear_expectation(heatmaps, self.linspace_x)
        pred_y = self._linear_expectation(heatmaps, self.linspace_y)
        coords = torch.cat([pred_x, pred_y], dim=-1)

        return coords, heatmaps

    def predict(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> SampleList:
        """Predict results from outputs."""

        batch_coords, batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_coords, batch_data_samples, test_cfg)

        # Whether to visualize the predicted heatmaps
        if test_cfg.get('output_heatmaps', False):
            for heatmaps, data_sample in zip(batch_heatmaps, preds):
                # Store the heatmap predictions in the data sample
                if 'pred_fileds' not in data_sample:
                    data_sample.pred_fields = PixelData()
                data_sample.pred_fields.heatmaps = heatmaps

        return preds

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, pred_heatmaps = self.forward(inputs)
        target_coords = torch.cat(
            [d.gt_instances.keypoints for d in batch_data_samples])
        keypoint_weights = torch.cat(
            [d.gt_instances.keypoint_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()

        # TODO: multi-loss calculation
        if 'use_dsnt' in train_cfg:
            loss = self.loss_module(pred_coords, pred_heatmaps, target_coords,
                                    keypoint_weights)
        else:
            loss = self.loss_module(pred_coords, target_coords,
                                    keypoint_weights)

        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(target_coords),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        losses.update(acc_pose=float(avg_acc))

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
