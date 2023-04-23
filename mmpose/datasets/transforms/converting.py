# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

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

    Example:
        >>> import numpy as np
        >>> # case 1: 1-to-1 mapping
        >>> # (0, 0) means target[0] = source[0]
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         (0, 0), (1, 1), (2, 2), (3, 3)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
        >>> assert np.equal(results['keypoints'],
        >>>                 np.arange(34).reshape(2, 3, 2)).all()
        >>> assert np.equal(results['keypoints_visible'],
        >>>                 np.arange(34).reshape(2, 3, 2) % 2).all()
        >>>
        >>> # case 2: 2-to-1 mapping
        >>> # ((1, 2), 0) means target[0] = (source[1] + source[2]) / 2
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         ((1, 2), 0), (1, 1), (2, 2)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
    """

    def __init__(self, num_keypoints: int,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]]):
        self.num_keypoints = num_keypoints
        self.mapping = mapping
        source_index, target_index = zip(*mapping)

        src1, src2 = [], []
        interpolation = False
        for x in source_index:
            if isinstance(x, (list, tuple)):
                assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                    'length 2'
                src1.append(x[0])
                src2.append(x[1])
                interpolation = True
            else:
                src1.append(x)
                src2.append(x)

        # When paired source_indexes are input,
        # keep a self.source_index2 for interpolation
        if interpolation:
            self.source_index2 = src2

        self.source_index = src1
        self.target_index = target_index
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        num_instances = results['keypoints'].shape[0]

        keypoints = np.zeros((num_instances, self.num_keypoints, 2))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))

        # When paired source_indexes are input,
        # perform interpolation with self.source_index and self.source_index2
        if self.interpolation:
            keypoints[:, self.target_index] = 0.5 * (
                results['keypoints'][:, self.source_index] +
                results['keypoints'][:, self.source_index2])

            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index] * \
                results['keypoints_visible'][:, self.source_index2]
        else:
            keypoints[:,
                      self.target_index] = results['keypoints'][:, self.
                                                                source_index]
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index]

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
