# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS
from mmpose.structures.keypoint import flip_keypoints_custom_center


@TRANSFORMS.register_module()
class RandomFlipAroundRoot(BaseTransform):
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        keypoints_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``keypoints``. Please
            refer to the docstring of the ``flip_keypoints_custom_center``
            function for more details.
        target_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``lifting_target``.
            Please refer to the docstring of the
            ``flip_keypoints_custom_center`` function for more details.
        flip_prob (float): Probability of flip. Default: 0.5.
        flip_camera (bool): Whether to flip horizontal distortion coefficients.
            Default: ``False``.

    Required keys:
        keypoints
        lifting_target

    Modified keys:
        (keypoints, keypoints_visible, lifting_target, lifting_target_visible,
        camera_param)
    """

    def __init__(self,
                 keypoints_flip_cfg,
                 target_flip_cfg,
                 flip_prob=0.5,
                 flip_camera=False):
        self.keypoints_flip_cfg = keypoints_flip_cfg
        self.target_flip_cfg = target_flip_cfg
        self.flip_prob = flip_prob
        self.flip_camera = flip_camera

    def transform(self, results: Dict) -> dict:
        """The transform function of :class:`ZeroCenterPose`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        keypoints = results['keypoints']
        if 'keypoints_visible' in results:
            keypoints_visible = results['keypoints_visible']
        else:
            keypoints_visible = np.ones(keypoints.shape[:-1], dtype=np.float32)
        lifting_target = results['lifting_target']
        if 'lifting_target_visible' in results:
            lifting_target_visible = results['lifting_target_visible']
        else:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)

        if np.random.rand() <= self.flip_prob:
            if 'flip_indices' not in results:
                flip_indices = list(range(self.num_keypoints))
            else:
                flip_indices = results['flip_indices']

            # flip joint coordinates
            keypoints, keypoints_visible = flip_keypoints_custom_center(
                keypoints, keypoints_visible, flip_indices,
                **self.keypoints_flip_cfg)
            lifting_target, lifting_target_visible = flip_keypoints_custom_center(  # noqa
                lifting_target, lifting_target_visible, flip_indices,
                **self.target_flip_cfg)

            results['keypoints'] = keypoints
            results['keypoints_visible'] = keypoints_visible
            results['lifting_target'] = lifting_target
            results['lifting_target_visible'] = lifting_target_visible

            # flip horizontal distortion coefficients
            if self.flip_camera:
                assert 'camera_param' in results, \
                    'Camera parameters are missing.'
                _camera_param = deepcopy(results['camera_param'])

                assert 'c' in _camera_param
                _camera_param['c'][0] *= -1

                if 'p' in _camera_param:
                    _camera_param['p'][0] *= -1

                results['camera_param'].update(_camera_param)

        return results
