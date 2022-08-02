# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmpose.core.utils.tensor_utils import to_numpy
from mmpose.core.utils.typing import ConfigType, OptConfigType, OptSampleList
from mmpose.metrics.utils import keypoint_pck_accuracy
from mmpose.registry import MODELS
from . import IntegralRegressionHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class DSNTHead(IntegralRegressionHead):
    """Top-down integral regression head introduced in `DSNT`_ by Nibali et
    al(2018). The head contains a differentiable spatial to numerical transform
    (DSNT) layer that do soft-argmax operation on the predicted heatmaps to
    regress the coordinates.

    This head is used for algorithms that require supervision of heatmaps.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        in_featuremap_size (int | sequence[int]): Size of input feature map
        num_joints (int): Number of joints
        out_sigma (bool): Predict the sigma (the variance of the joint
            location) together with the joint location. Introduced in `RLE`_
            by Li et al(2021). Default: False
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`DSNTLoss`
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
                     type='DSNTLoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        super().__init__(
            in_channels=in_channels,
            in_featuremap_size=in_featuremap_size,
            num_joints=num_joints,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            has_final_layer=has_final_layer,
            input_transform=input_transform,
            input_index=input_index,
            align_corners=align_corners,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg)

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, pred_heatmaps = self.forward(inputs)
        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()

        # TODO: multi-loss calculation
        loss = self.loss_module(pred_coords, pred_heatmaps, keypoint_labels,
                                keypoint_weights)

        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        losses.update(acc_pose=float(avg_acc))

        return losses
