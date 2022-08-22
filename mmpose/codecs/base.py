# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
from mmengine.utils import is_method_overridden


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
            - keypoint dimension: D

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        """

    @abstractmethod
    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints.

        Args:
            encoded (any): Encoded keypoint representation using the codec

        Returns:
            tuple:
            - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        """

    def batch_decode(self, batch_encoded: Any
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Decode keypoints.

        Args:
            batch_encoded (any): A batch of encoded keypoint
                representations

        Returns:
            tuple:
            - batch_keypoints (List[np.ndarray]): Each element is keypoint
                coordinates in shape (N, K, D)
            - batch_keypoints (List[np.ndarray]): Each element is keypoint
                visibility in shape (N, K)
        """
        raise NotImplementedError()

    @property
    def support_batch_decoding(self) -> bool:
        """Return whether the codec support decoding from batch data."""
        return is_method_overridden('batch_decode', BaseKeypointCodec,
                                    self.__class__)
