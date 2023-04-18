# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ZeroCenterPose(BaseTransform):
    """Zero-center the pose around a given root keypoint. Optionally, the root
    keypoint can be removed from the original pose and stored as a separate
    item.

    Note that the root-centered keypoints may no longer align with some
    annotation information (e.g. flip_pairs, num_keypoints, etc.) due to
    the removal of the root keypoint.

    Args:
        item (str): The name of the pose to apply root-centering.
        root_index (int): Root keypoint index in the pose.
        remove_root (bool): If true, remove the root keypoint from the pose.
            Default: ``False``.
        visible_item (str): The name of the visibility item. Required if
            ``remove_root`` is set to ``True``.
        save_index (bool): If true, store the root position separated from the
            original pose. Default: ``False``.

    Required keys:
        item

    Modified keys:
        item, visible_item, save_index
    """

    def __init__(self,
                 item,
                 root_index,
                 remove_root=False,
                 visible_item=None,
                 save_index=False):
        self.item = item
        self.root_index = root_index
        self.remove_root = remove_root
        if self.remove_root:
            assert visible_item is not None
        self.visible_item = visible_item
        self.save_index = save_index

    def transform(self, results: Dict) -> dict:
        """The transform function of :class:`ZeroCenterPose`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        assert self.item in results
        keypoints = results[self.item]
        self.root_index = self.root_index

        assert keypoints.ndim >= 2 and keypoints.shape[-2] > self.root_index, \
            f'Got invalid joint shape {keypoints.shape}'

        root = keypoints[..., self.root_index:self.root_index + 1, :]
        keypoints = keypoints - root
        results[self.item] = keypoints

        if self.save_index:
            results[f'{self.item}_root'] = root

        if self.remove_root:
            results[self.item] = np.delete(
                results[self.item], self.root_index, axis=-2)
            if self.visible_item is not None:
                assert self.visible_item in results
                visible_arr = results[self.visible_item]
                assert visible_arr.ndim in {1, 2}
                axis_to_remove = -2 if visible_arr.ndim == 2 else -1
                results[self.visible_item] = np.delete(
                    visible_arr, self.root_index, axis=axis_to_remove)
            # Add a flag to avoid latter transforms that rely on the root
            # joint or the original joint index
            results[f'{self.item}_root_removed'] = True

            # Save the root index which is necessary to restore the global pose
            if self.save_index:
                results[f'{self.item}_root_index'] = self.root_index

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(item={self.item}, '
        repr_str += f'root_index={self.root_index}, '
        repr_str += f'remove_root={self.remove_root}, '
        repr_str += f'save_index={self.save_index}, '
        repr_str += f'visible_item={self.visible_item})'
        return repr_str


@TRANSFORMS.register_module()
class NormalizeKeypointsWithImage(BaseTransform):
    """Normalize the 2D keypoint coordinate with image width and height. Range.

    [0, w] is mapped to [-1, 1], while preserving the aspect ratio.

    Args:
        item (str|list[str]): The key name of the pose to normalize.
        normalize_camera (bool): Whether to normalize camera intrinsics.
            Default: False.

    Required keys:
        item, camera_param

    Modified keys:
        item (, camera_param)
    """

    def __init__(self, item, normalize_camera=False, camera_param=None):
        self.item = item
        if isinstance(self.item, str):
            self.item = [self.item]

        self.normalize_camera = normalize_camera

    def transform(self, results: Dict) -> dict:
        """The transform function of :class:`NormalizeKeypointsWithImage`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        camera_param = deepcopy(results['camera_param'])
        assert 'w' in camera_param and 'h' in camera_param
        center = np.array([0.5 * camera_param['w'], 0.5 * camera_param['h']],
                          dtype=np.float32)
        scale = np.array(0.5 * camera_param['w'], dtype=np.float32)

        for item in self.item:
            assert item in results
            results[item] = (results[item] - center) / scale

        if self.normalize_camera:
            assert 'f' in camera_param and 'c' in camera_param
            camera_param['f'] = camera_param['f'] / scale
            camera_param['c'] = (camera_param['c'] - center[:, None]) / scale
            if 'camera_param' not in results:
                results['camera_param'] = dict()
            results['camera_param'].update(camera_param)

        return results
