# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmengine.model import BaseModule


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)
