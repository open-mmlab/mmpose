# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class VideoPoseLifting(BaseKeypointCodec):
    r"""Generate keypoint coordinates for pose lifter.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - pose-lifitng target dimension: C

    Args:
        num_keypoints (int): The number of keypoints in the dataset.
        zero_center: Whether to zero-center the target around root. Default:
            ``True``.
        root_index (int): Root keypoint index in the pose. Default: 0.
        remove_root (bool): If true, remove the root keypoint from the pose.
            Default: ``False``.
        save_index (bool): If true, store the root position separated from the
            original pose, only takes effect if ``remove_root`` is ``True``.
            Default: ``False``.
        normalize_camera (bool): Whether to normalize camera intrinsics.
            Default: ``False``.
    """

    auxiliary_encode_keys = {
        'lifting_target', 'lifting_target_visible', 'camera_param'
    }

    def __init__(self,
                 num_keypoints: int,
                 zero_center: bool = True,
                 root_index: int = 0,
                 remove_root: bool = False,
                 save_index: bool = False,
                 normalize_camera: bool = False):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.zero_center = zero_center
        self.root_index = root_index
        self.remove_root = remove_root
        self.save_index = save_index
        self.normalize_camera = normalize_camera

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               lifting_target: Optional[np.ndarray] = None,
               lifting_target_visible: Optional[np.ndarray] = None,
               camera_param: Optional[dict] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
            keypoints_visible (np.ndarray, optional): Keypoint visibilities in
                shape (N, K).
            lifting_target (np.ndarray, optional): 3d target coordinate in
                shape (K, C).
            lifting_target_visible (np.ndarray, optional): Target coordinate in
                shape (K, ).
            camera_param (dict, optional): The camera parameter dictionary.

        Returns:
            encoded (dict): Contains the following items:

                - keypoint_labels (np.ndarray): The processed keypoints in
                  shape (K * D, N) where D is 2 for 2d coordinates.
                - lifting_target_label: The processed target coordinate in
                  shape (K, C) or (K-1, C).
                - lifting_target_weights (np.ndarray): The target weights in
                  shape (K, ) or (K-1, ).
                - trajectory_weights (np.ndarray): The trajectory weights in
                  shape (K, ).

                In addition, there are some optional items it may contain:

                - target_root (np.ndarray): The root coordinate of target in
                  shape (C, ). Exists if ``self.zero_center`` is ``True``.
                - target_root_removed (bool): Indicate whether the root of
                  pose-lifitng target is removed. Exists if
                  ``self.remove_root`` is ``True``.
                - target_root_index (int): An integer indicating the index of
                  root. Exists if ``self.remove_root`` and ``self.save_index``
                  are ``True``.
                - camera_param (dict): The updated camera parameter dictionary.
                  Exists if ``self.normalize_camera`` is ``True``.
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if lifting_target is None:
            lifting_target = keypoints[0]

        # set initial value for `lifting_target_weights`
        # and `trajectory_weights`
        if lifting_target_visible is None:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)
            lifting_target_weights = lifting_target_visible
            trajectory_weights = (1 / lifting_target[:, 2])
        else:
            valid = lifting_target_visible > 0.5
            lifting_target_weights = np.where(valid, 1., 0.).astype(np.float32)
            trajectory_weights = lifting_target_weights

        if camera_param is None:
            camera_param = dict()

        encoded = dict()

        lifting_target_label = lifting_target.copy()
        # Zero-center the target pose around a given root keypoint
        if self.zero_center:
            assert (lifting_target.ndim >= 2 and
                    lifting_target.shape[-2] > self.root_index), \
                f'Got invalid joint shape {lifting_target.shape}'

            root = lifting_target[..., self.root_index, :]
            lifting_target_label = lifting_target_label - root
            encoded['target_root'] = root

            if self.remove_root:
                lifting_target_label = np.delete(
                    lifting_target_label, self.root_index, axis=-2)
                assert lifting_target_weights.ndim in {1, 2}
                axis_to_remove = -2 if lifting_target_weights.ndim == 2 else -1
                lifting_target_weights = np.delete(
                    lifting_target_weights,
                    self.root_index,
                    axis=axis_to_remove)
                # Add a flag to avoid latter transforms that rely on the root
                # joint or the original joint index
                encoded['target_root_removed'] = True

                # Save the root index for restoring the global pose
                if self.save_index:
                    encoded['target_root_index'] = self.root_index

        # Normalize the 2D keypoint coordinate with image width and height
        _camera_param = deepcopy(camera_param)
        assert 'w' in _camera_param and 'h' in _camera_param
        center = np.array([0.5 * _camera_param['w'], 0.5 * _camera_param['h']],
                          dtype=np.float32)
        scale = np.array(0.5 * _camera_param['w'], dtype=np.float32)

        keypoint_labels = (keypoints - center) / scale

        assert keypoint_labels.ndim in {2, 3}
        if keypoint_labels.ndim == 2:
            keypoint_labels = keypoint_labels[None, ...]

        if self.normalize_camera:
            assert 'f' in _camera_param and 'c' in _camera_param
            _camera_param['f'] = _camera_param['f'] / scale
            _camera_param['c'] = (_camera_param['c'] - center[:, None]) / scale
            encoded['camera_param'] = _camera_param

        encoded['keypoint_labels'] = keypoint_labels
        encoded['lifting_target_label'] = lifting_target_label
        encoded['lifting_target_weights'] = lifting_target_weights
        encoded['trajectory_weights'] = trajectory_weights

        return encoded

    def decode(self,
               encoded: np.ndarray,
               target_root: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, C).
            target_root (np.ndarray, optional): The pose-lifitng target root
                coordinate. Default: ``None``.

        Returns:
            keypoints (np.ndarray): Decoded coordinates in shape (N, K, C).
            scores (np.ndarray): The keypoint scores in shape (N, K).
        """
        keypoints = encoded.copy()

        if target_root.size > 0:
            keypoints = keypoints + np.expand_dims(target_root, axis=0)
            if self.remove_root:
                keypoints = np.insert(
                    keypoints, self.root_index, target_root, axis=1)
        scores = np.ones(keypoints.shape[:-1], dtype=np.float32)

        return keypoints, scores
