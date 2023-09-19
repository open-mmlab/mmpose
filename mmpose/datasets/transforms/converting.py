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
        self.target_index = list(target_index)
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        num_instances = results['keypoints'].shape[0]

        # Initialize output arrays
        keypoints = np.zeros((num_instances, self.num_keypoints, 2))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))

        if 'keypoints_3d' in results:
            keypoints_3d = np.zeros((num_instances, self.num_keypoints, 3),
                                    dtype=np.float32)
        flip_indices = results.get('flip_indices', None)

        # Create a mask to weight visibility loss
        keypoints_visible_weights = keypoints_visible.copy()
        keypoints_visible_weights[:, self.target_index] = 1.0

        # Interpolate keypoints if pairs of source indexes provided
        if self.interpolation:
            keypoints[:, self.target_index] = 0.5 * (
                results['keypoints'][:, self.source_index] +
                results['keypoints'][:, self.source_index2])

            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index] * \
                results['keypoints_visible'][:, self.source_index2]

            if 'keypoints_3d' in results:
                keypoints_3d[:, self.target_index] = 0.5 * (
                    results['keypoints_3d'][:, self.source_index] +
                    results['keypoints_3d'][:, self.source_index2])

            # Flip keypoints if flip_indices provided
            if flip_indices is not None:
                for i, (x1, x2) in enumerate(
                        zip(self.source_index, self.source_index2)):
                    id = flip_indices[x1] if x1 == x2 else i
                    flip_indices[i] = id if id < self.num_keypoints else i
                flip_indices = flip_indices[:len(self.source_index)]
        # Otherwise just copy from the source index
        else:
            keypoints[:,
                      self.target_index] = results['keypoints'][:, self.
                                                                source_index]
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index]
            if 'keypoints_3d' in results:
                keypoints_3d[:, self.target_index] = results[
                    'keypoints_3d'][:, self.source_index]

        # Update the results dict
        results['keypoints'] = keypoints
        results['keypoints_visible'] = np.stack(
            [keypoints_visible, keypoints_visible_weights], axis=2)
        if 'keypoints_3d' in results:
            results['keypoints_3d'] = keypoints_3d
            results['lifting_target'] = keypoints_3d[results['target_idx']]
            results['lifting_target_visible'] = keypoints_visible[
                results['target_idx']]
        results['flip_indices'] = flip_indices

        return results

    def transform_sigmas(self, sigmas: Union[List, np.ndarray]):
        """Transforms the sigmas based on the mapping."""
        list_input = False
        if isinstance(sigmas, list):
            sigmas = np.array(sigmas)
            list_input = True

        new_sigmas = np.ones(self.num_keypoints, dtype=sigmas.dtype)
        new_sigmas[self.target_index] = sigmas[self.source_index]

        if list_input:
            new_sigmas = new_sigmas.tolist()

        return new_sigmas

    def transform_ann(self, ann_info: Union[dict, list]):
        """Transforms the annotations based on the mapping."""

        list_input = True
        if not isinstance(ann_info, list):
            ann_info = [ann_info]
            list_input = False

        for ann in ann_info:
            if 'keypoints' in ann:
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                new_keypoints = np.zeros((self.num_keypoints, 3),
                                         dtype=keypoints.dtype)
                new_keypoints[self.target_index] = keypoints[self.source_index]
                ann['keypoints'] = new_keypoints.reshape(-1).tolist()
            if 'num_keypoints' in ann:
                ann['num_keypoints'] = self.num_keypoints

        if not list_input:
            ann_info = ann_info[0]

        return ann_info

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(num_keypoints={self.num_keypoints}, '\
                    f'mapping={self.mapping})'
        return repr_str
