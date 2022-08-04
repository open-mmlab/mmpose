# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from mmpose.utils.typing import ConfigType
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class MultiLevelHeatmapEncoder(BaseKeypointCodec):
    """An encoder wrapper that contain multiple encoders to generate multi-
    level heatmaps.

    Note:

        - encoder number: n
        - instance number: N
        - keypoint number: K
        - keypoint dimension: C
        - image size: [w, h]
        - heatmap size: [W, H]

    Args:
        encoders (List[dict]): The encoder config list that should contain
            at least two encoder configs
    """

    def __init__(self, encoders: List[ConfigType]):
        super().__init__()

        assert isinstance(encoders, list) and len(encoders) > 1, (
            '``encoders`` should be a list of more than 1 configs')

        self.encoders = [KEYPOINT_CODECS.build(cfg) for cfg in encoders]

        self.input_size = self.encoders[0].input_size
        self.heatmap_size = self.encoders[0].heatmap_size

        # check all encoders have the samle ``input_size`` and
        # ``heatmap_size``
        for i, encoder in enumerate(self.encoders):
            assert hasattr(encoder, 'input_size'), (
                f'Encoder[{i}] does not have attribute ``input_size``')

            assert hasattr(encoder, 'heatmap_size'), (
                f'Encoder[{i}] does not have attribute ``heatmap_size``')

            assert encoder.input_size == self.input_size, (
                f'Encoder[{i}] has unmatched input_size ({encoder.input_size})'
                f' with the wrapper ({self.input_size})')

            assert encoder.heatmap_size == self.heatmap_size, (
                f'Encoder[{i}] has unmatched heatmap_size '
                f'({encoder.heatmap_size}) with the wrapper '
                f'({self.heatmap_size})')

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode keypoints with wrapped encoders respectively and combine the
        results.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K*n, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, n, K)
        """

        all_heatmaps = []
        all_keypoint_weights = []

        for encoder in self.encoders:
            _heatmaps, _keypoint_weights = encoder.encode(
                keypoints, keypoints_visible)
            all_heatmaps.append(_heatmaps)
            all_keypoint_weights.append(_keypoint_weights)

        # heatmaps.shape: # [K, H, W] -> [K*n, H, W]
        heatmaps = np.concatenate(all_heatmaps)
        # keypoint_weights.shape: [N, K] -> [N, n, K]
        keyponit_weights = np.stack(all_keypoint_weights, axis=1)

        return heatmaps, keyponit_weights

    def decode(self, encoded: Any):
        raise NotImplementedError(
            f'The encoder wrapper {self.__class__.__name__} does not support '
            'decoding.')
