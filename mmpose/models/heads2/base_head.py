# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor

from mmpose.core.utils.typing import OptSampleList, SampleList


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head. A subclass should override :meth:`predict` and :meth:`loss`.

    Args:
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.
    """

    @abstractmethod
    def predict(self, feats: Tuple[Tensor],
                batch_data_samples: OptSampleList) -> SampleList:
        """Predict results from features."""

    @abstractmethod
    def loss(self, feats: Tuple[Tensor],
             batch_data_samples: OptSampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
