# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from ..base_head import BaseHead


@MODELS.register_module()
class MotionRegressionHead(BaseHead):
    """Regression head of `MotionBERT`_ by Zhu et al (2022).

    Args:
        in_channels (int): Number of input channels. Default: 256.
        out_channels (int): Number of output channels. Default: 3.
        embedding_size (int): Number of embedding channels. Default: 512.
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`MSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`MotionBERT`: https://arxiv.org/abs/2210.06551
    """

    _version = 2

    def __init__(self,
                 in_channels: int = 256,
                 out_channels: int = 3,
                 embedding_size: int = 512,
                 loss: ConfigType = dict(
                     type='MSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Define fully-connected layers
        self.pre_logits = nn.Sequential(
            OrderedDict([('fc', nn.Linear(in_channels, embedding_size)),
                         ('act', nn.Tanh())]))
        self.fc = nn.Linear(
            embedding_size,
            out_channels) if embedding_size > 0 else nn.Identity()

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: Output coordinates (and sigmas[optional]).
        """
        x = feats  # (B, F, K, in_channels)
        x = self.pre_logits(x)  # (B, F, K, embedding_size)
        x = self.fc(x)  # (B, F, K, out_channels)

        return x

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from outputs.

        Returns:
            preds (sequence[InstanceData]): Prediction results.
                Each contains the following fields:

                - keypoints: Predicted keypoints of shape (B, N, K, D).
                - keypoint_scores: Scores of predicted keypoints of shape
                  (B, N, K).
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_coords = self.forward(_feats)
            _batch_coords_flip = torch.stack([
                flip_coordinates(
                    _batch_coord_flip,
                    flip_indices=flip_indices,
                    shift_coords=test_cfg.get('shift_coords', True),
                    input_size=(1, 1))
                for _batch_coord_flip in self.forward(_feats_flip)
            ],
                                             dim=0)
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords = self.forward(feats)

        # Restore global position with camera_param and factor
        camera_param = batch_data_samples[0].metainfo.get('camera_param', None)
        if camera_param is not None:
            w = torch.stack([
                torch.from_numpy(np.array([b.metainfo['camera_param']['w']]))
                for b in batch_data_samples
            ])
            h = torch.stack([
                torch.from_numpy(np.array([b.metainfo['camera_param']['h']]))
                for b in batch_data_samples
            ])
        else:
            w = torch.stack([
                torch.empty((0), dtype=torch.float32)
                for _ in batch_data_samples
            ])
            h = torch.stack([
                torch.empty((0), dtype=torch.float32)
                for _ in batch_data_samples
            ])

        factor = batch_data_samples[0].metainfo.get('factor', None)
        if factor is not None:
            factor = torch.stack([
                torch.from_numpy(b.metainfo['factor'])
                for b in batch_data_samples
            ])
        else:
            factor = torch.stack([
                torch.empty((0), dtype=torch.float32)
                for _ in batch_data_samples
            ])

        preds = self.decode((batch_coords, w, h, factor))

        return preds

    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs = self.forward(inputs)

        lifting_target_label = torch.stack([
            d.gt_instance_labels.lifting_target_label
            for d in batch_data_samples
        ])
        lifting_target_weight = torch.stack([
            d.gt_instance_labels.lifting_target_weight
            for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_outputs, lifting_target_label,
                                lifting_target_weight.unsqueeze(-1))

        losses.update(loss_pose3d=loss)

        # calculate accuracy
        mpjpe_err = keypoint_mpjpe(
            pred=to_numpy(pred_outputs),
            gt=to_numpy(lifting_target_label),
            mask=to_numpy(lifting_target_weight) > 0)

        mpjpe_pose = torch.tensor(
            mpjpe_err, device=lifting_target_label.device)
        losses.update(mpjpe=mpjpe_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='TruncNormal', layer=['Linear'], std=0.02)]
        return init_cfg
