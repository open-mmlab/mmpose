# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class ImagePoseLifting(BaseKeypointCodec):
    r"""Generate keypoint coordinates for pose lifter.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - target dimension: C

    Args:
        num_keypoints (int): The number of keypoints in the dataset.
        root_index (int): Root keypoint index in the pose.
        remove_root (bool): If true, remove the root keypoint from the pose.
            Default: ``False``.
        save_index (bool): If true, store the root position separated from the
            original pose. Default: ``False``.
        keypoints_mean (np.ndarray, optional): Mean values of keypoints
            coordinates in shape (K, D).
        keypoints_std (np.ndarray, optional): Std values of keypoints
            coordinates in shape (K, D).
        target_mean (np.ndarray, optional): Mean values of target coordinates
            in shape (K, C).
        target_std (np.ndarray, optional): Std values of target coordinates
            in shape (K, C).
    """

    auxiliary_encode_keys = {'target', 'target_visible'}

    def __init__(self,
                 num_keypoints: int,
                 root_index: int,
                 remove_root: bool = False,
                 save_index: bool = False,
                 keypoints_mean: Optional[np.ndarray] = None,
                 keypoints_std: Optional[np.ndarray] = None,
                 target_mean: Optional[np.ndarray] = None,
                 target_std: Optional[np.ndarray] = None):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.root_index = root_index
        self.remove_root = remove_root
        self.save_index = save_index
        if keypoints_mean is not None and keypoints_std is not None:
            assert keypoints_mean.shape == keypoints_std.shape
        if target_mean is not None and target_std is not None:
            assert target_mean.shape == target_std.shape
        self.keypoints_mean = keypoints_mean
        self.keypoints_std = keypoints_std
        self.target_mean = target_mean
        self.target_std = target_std

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               target: Optional[np.ndarray] = None,
               target_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
            keypoints_visible (np.ndarray, optional): Keypoint visibilities in
                shape (N, K).
            target (np.ndarray, optional): Target coordinate in shape (K, C).
            target_visible (np.ndarray, optional): Target coordinate in shape
                (K, ).

        Returns:
            encoded (dict): Contains the following items:

                - keypoint_labels (np.ndarray): The processed keypoints in
                  shape (K * D, N) where D is 2 for 2d coordinates.
                - target_label: The processed target coordinate in shape (K, C)
                  or (K-1, C).
                - target_weights (np.ndarray): The target weights in shape
                  (K, ) or (K-1, ).
                - trajectory_weights (np.ndarray): The trajectory weights in
                  shape (K, ).
                - target_root (np.ndarray): The root coordinate of target in
                  shape (C, ).

                In addition, there are some optional items it may contain:

                - target_root_removed (bool): Indicate whether the root of
                  target is removed. Added if ``self.remove_root`` is ``True``.
                - target_root_index (int): An integer indicating the index of
                  root. Added if ``self.remove_root`` and ``self.save_index``
                  are ``True``.
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if target is None:
            target = keypoints[0]

        # set initial value for `target_weights` and `trajectory_weights`
        if target_visible is None:
            target_visible = np.ones(target.shape[:-1], dtype=np.float32)
            target_weights = target_visible
            trajectory_weights = (1 / target[:, 2])
        else:
            valid = target_visible > 0.5
            target_weights = np.where(valid, 1., 0.).astype(np.float32)
            trajectory_weights = target_weights

        encoded = dict()

        # Zero-center the target pose around a given root keypoint
        assert target.ndim >= 2 and target.shape[-2] > self.root_index, \
            f'Got invalid joint shape {target.shape}'

        root = target[..., self.root_index, :]
        target_label = target - root

        if self.remove_root:
            target_label = np.delete(target_label, self.root_index, axis=-2)
            assert target_weights.ndim in {1, 2}
            axis_to_remove = -2 if target_weights.ndim == 2 else -1
            target_weights = np.delete(
                target_weights, self.root_index, axis=axis_to_remove)
            # Add a flag to avoid latter transforms that rely on the root
            # joint or the original joint index
            encoded['target_root_removed'] = True

            # Save the root index which is necessary to restore the global pose
            if self.save_index:
                encoded['target_root_index'] = self.root_index

        # Normalize the 2D keypoint coordinate with mean and std
        keypoint_labels = keypoints.copy()
        if self.keypoints_mean is not None and self.keypoints_std is not None:
            keypoints_shape = keypoints.shape
            assert self.keypoints_mean.shape == keypoints_shape[1:]

            keypoint_labels = (keypoint_labels -
                               self.keypoints_mean) / self.keypoints_std
        if self.target_mean is not None and self.target_std is not None:
            target_shape = target_label.shape
            assert self.target_mean.shape == target_shape

            target_label = (target_label - self.target_mean) / self.target_std

        # Generate reshaped keypoint coordinates
        assert keypoint_labels.ndim in {2, 3}
        if keypoint_labels.ndim == 2:
            keypoint_labels = keypoint_labels[None, ...]

        N = keypoint_labels.shape[0]
        keypoint_labels = keypoint_labels.transpose(1, 2, 0).reshape(-1, N)

        encoded['keypoint_labels'] = keypoint_labels
        encoded['target_label'] = target_label
        encoded['target_weights'] = target_weights
        encoded['trajectory_weights'] = trajectory_weights
        encoded['target_root'] = root

        return encoded

    def decode(self,
               encoded: np.ndarray,
               restore_global_position: bool = False,
               target_root: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, C).
            restore_global_position (bool): Whether to restore global position.
                Default: ``False``.
            target_root (np.ndarray, optional): The target root coordinate.
                Default: ``None``.

        Returns:
            keypoints (np.ndarray): Decoded coordinates in shape (N, K, C).
            scores (np.ndarray): The keypoint scores in shape (N, K).
        """
        keypoints = encoded.copy()

        if self.target_mean is not None and self.target_std is not None:
            assert self.target_mean.shape == keypoints.shape[1:]
            keypoints = keypoints * self.target_std + self.target_mean

        if restore_global_position:
            assert target_root is not None
            keypoints = keypoints + np.expand_dims(target_root, axis=0)
            if self.remove_root:
                keypoints = np.insert(
                    keypoints, self.root_index, target_root, axis=1)
        scores = np.ones(keypoints.shape[:-1], dtype=np.float32)

        return keypoints, scores
