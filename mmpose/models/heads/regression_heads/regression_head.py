# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.core.utils.tensor_utils import _to_numpy
from mmpose.core.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                      SampleList)
from mmpose.metrics.utils import keypoint_pck_accuracy
from mmpose.registry import KEYPOINT_CODECS, MODELS
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RegressionHead(BaseHead):
    """Top-down regression head introduced in `Deeppose`_ by Toshev et al
    (2014). The head is composed of fully-connected layers to predict the
    coordinates directly.


    Args:
        in_channels (int | sequence[int]): Number of input channels
        num_joints (int): Number of joints
        output_sigma (bool): Predict the sigma (the variance of the joint
            location) together with the joint location. Introduced in `RLE`_
            by Li et al(2021). Default: False
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`SmoothL1Loss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Deeppose`: https://arxiv.org/abs/1312.4659
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_joints: int,
                 output_sigma: bool,
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
        self.output_sigma = output_sigma
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
        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define fully-connected layers
        if self.output_sigma:
            self.fc = nn.Linear(self.in_channels, self.num_joints * 4)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        x = self._transform_inputs(feats)

        x = self.fc(x)

        if self.output_sigma:
            return x.reshape(-1, self.num_joints, 4)
        else:
            return x.reshape(-1, self.num_joints, 2)

    def predict(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
                test_cfg: ConfigType) -> SampleList:
        """Predict results from outputs."""

        batch_coords = self.forward(feats)
        preds = self.decode(batch_coords, batch_data_samples)

        return preds

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: OptSampleList,
             train_cfg: ConfigType) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs = self.forward(inputs)
        target_coords = torch.stack(
            [d.gt_fields.keypoints for d in batch_data_samples])
        target_weights = torch.cat(
            [d.gt_instance.target_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_outputs, target_coords, target_weights)
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses.update(loss_kpts=loss)

        # calculate accuracy
        pred_coords = pred_outputs[:, :, :2]

        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=_to_numpy(pred_coords),
            gt=_to_numpy(target_coords),
            mask=_to_numpy(target_weights).squeeze(-1) > 0,
            thr=0.05,
            normalize=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        losses.update(acc_pose=float(avg_acc))

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
