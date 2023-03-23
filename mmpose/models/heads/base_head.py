# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

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

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)

        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints = []
            batch_scores = []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs,
                                                   self.decoder.decode)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)

        preds = [
            InstanceData(keypoints=keypoints, keypoint_scores=scores)
            for keypoints, scores in zip(batch_keypoints, batch_scores)
        ]

        return preds
