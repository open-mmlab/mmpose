# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (Features, InstanceList, OptConfigType,
                                 OptSampleList, Predictions)


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
    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        """Predict results from features."""

    @abstractmethod
    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    def _get_in_channels(self) -> Union[int, List[int]]:
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

    def _transform_inputs(self, feats: Tuple[Tensor]
                          ) -> Union[Tensor, Tuple[Tensor]]:
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

    def decode(self, batch_outputs: Union[Tensor,
                                          Tuple[Tensor]]) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = self.decoder.batch_decode(
                batch_outputs)

        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints = []
            batch_scores = []
            for outputs in batch_output_np:
                keypoints, scores = self.decoder.decode(outputs)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)

        preds = [
            InstanceData(keypoints=keypoints, keypoint_scores=scores)
            for keypoints, scores in zip(batch_keypoints, batch_scores)
        ]

        return preds
