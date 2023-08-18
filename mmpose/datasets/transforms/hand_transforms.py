# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

from mmpose.codecs import *  # noqa: F401, F403
from mmpose.registry import TRANSFORMS
from .common_transforms import RandomFlip


@TRANSFORMS.register_module()
class HandRandomFlip(RandomFlip):
    """Data augmentation with random image flip. A child class of
    `TopDownRandomFlip`.

    Required Keys:

        - img
        - joints_3d
        - joints_3d_visible
        - center
        - hand_type
        - rel_root_depth
        - ann_info

    Modified Keys:

        - img
        - joints_3d
        - joints_3d_visible
        - center
        - hand_type
        - rel_root_depth

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
    """

    def __init__(self, prob: Union[float, List[float]] = 0.5) -> None:
        super().__init__(prob=prob, direction='horizontal')

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`HandRandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        # base flip augmentation
        results = super().transform(results)

        # flip hand type and root depth
        hand_type = results['hand_type']
        rel_root_depth = results['rel_root_depth']
        flipped = results['flip']
        if flipped:
            hand_type[..., [0, 1]] = hand_type[..., [1, 0]]
            rel_root_depth = -rel_root_depth
        results['hand_type'] = hand_type
        results['rel_root_depth'] = rel_root_depth
        return results
