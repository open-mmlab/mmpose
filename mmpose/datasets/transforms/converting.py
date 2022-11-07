# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class KeypointConverter(BaseTransform):
    """Change the order of keypoints according to the given mapping.

    Required Keys:

        - keypoints
        - keypoints_visible

    Modified Keys:

        - keypoints
        - keypoints_visible

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
        mapping (list): A list containing mapping indexes. Each element has
            format (source_index, target_index)
    """

    def __init__(self, num_keypoints: int, mapping: List[Tuple[int, int]]):
        self.num_keypoints = num_keypoints
        self.mapping = mapping

    def transform(self, results: dict) -> dict:
        num_instances = results['keypoints'].shape[0]

        keypoints = np.zeros((num_instances, self.num_keypoints, 2))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))

        source_index, target_index = zip(*self.mapping)
        keypoints[:, target_index] = results['keypoints'][:, source_index]
        keypoints_visible[:, target_index] = results[
            'keypoints_visible'][:, source_index]

        results['keypoints'] = keypoints
        results['keypoints_visible'] = keypoints_visible
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(num_keypoints={self.num_keypoints}, '\
                    f'mapping={self.mapping})'
        return repr_str
