# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmpose.core.utils.typing import ConfigType, OptSampleList, SampleList


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head. A subclass should override :meth:`predict` and :meth:`loss`.

    Args:
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.
    """

    @abstractmethod
    def forward(self, feats: Tuple[Tensor]):
        """Forward the network."""

    @abstractmethod
    def predict(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
                test_cfg: ConfigType) -> SampleList:
        """Predict results from features."""

    @abstractmethod
    def loss(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList,
             train_cfg: ConfigType) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    def _get_in_channels(self):
        """Get the input channel number of the network according to the feature
        channel numbers and the input transform type."""

        feat_channels = self.in_channels
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels]

        if self.input_transform == 'resize_concat':
            if isinstance(self.input_index, int):
                in_channels = feat_channels[self.input_index]
            else:
                in_channels = sum(feat_channels[i] for i in self.input_index)
        elif self.input_transform == 'select':
            if isinstance(self.input_index, int):
                in_channels = feat_channels[self.input_index]
            else:
                in_channels = [feat_channels[i] for i in self.input_index]
        else:
            raise (ValueError,
                   f'Invalid input transform mode "{self.input_transform}"')

        return in_channels

    def _transform_inputs(
            self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Transform multi scale features into the network input."""
        if self.input_transform == 'resize_concat':
            inputs = [feats[i] for i in self.input_index]
            resized_inputs = [
                F.interpolate(
                    x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(resized_inputs, dim=1)
        elif self.input_transform == 'select':
            if isinstance(self.input_index, int):
                inputs = feats[self.input_index]
            else:
                inputs = tuple(feats[i] for i in self.input_index)
        else:
            raise (ValueError,
                   f'Invalid input transform mode "{self.input_transform}"')

        return inputs
