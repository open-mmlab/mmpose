# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class BaseKeypointCodec(metaclass=ABCMeta):
    """The base class of the keypoint codec.

    A keypoint codec is a module to encode keypoint coordinates to specific
    representation (e.g. heatmap) and vice versa. A subclass should implement
    the methods :meth:`encode` and :meth:`decode`.
    """

    @abstractmethod
    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> Any:
        """Encode keypoints.

        Note:

            - instance number: N
            - keypoint number: K
            - keypoint dimension: C

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
            keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, C)
        """

    @abstractmethod
    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints.

        Args:
            encoded (any): Encoded keypoint representation using the codec

        Returns:
            tuple:
            - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, C)
        """
