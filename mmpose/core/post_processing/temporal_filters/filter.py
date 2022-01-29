# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class TemporalFilter(metaclass=ABCMeta):

    def __init__(self, window_size=1):
        self._window_size = window_size

    @property
    def window_size(self):
        return self._window_size

    @abstractmethod
    def __call__(self, x):
        """Apply filter to a pose sequence.

        Args:
            x (Tensor): shape (T, K, C), the pose sequence of a single target

        Returns:
            Tensor: shape(T, K, C)
        """
