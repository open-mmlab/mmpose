# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import generate_3d_gaussian_heatmaps
from .utils.post_processing import get_heatmap_3d_maximum


@KEYPOINT_CODECS.register_module()
class Hand3DHeatmap(BaseKeypointCodec):
    r"""Generate target 3d heatmap and relative root depth for hand datasets.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Args:
        image_size (tuple): Size of image. Default: ``[256, 256]``.
        root_heatmap_size (int): Size of heatmap of root head.
            Default: 64.
        heatmap_size (tuple): Size of heatmap. Default: ``[64, 64, 64]``.
        heatmap3d_depth_bound (float): Boundary for 3d heatmap depth.
            Default: 400.0.
        heatmap_size_root (int): Size of 3d heatmap root. Default: 64.
        depth_size (int): Number of depth discretization size, used for
            decoding. Defaults to 64.
        root_depth_bound (float): Boundary for 3d heatmap root depth.
            Default: 400.0.
        use_different_joint_weights (bool): Whether to use different joint
            weights. Default: ``False``.
        sigma (int): Sigma of heatmap gaussian. Default: 2.
        joint_indices (list, optional): Indices of joints used for heatmap
            generation. If None (default) is given, all joints will be used.
            Default: ``None``.
        max_bound (float): The maximal value of heatmap. Default: 1.0.
    """

    auxiliary_encode_keys = {
        'dataset_keypoint_weights', 'rel_root_depth', 'rel_root_valid',
        'hand_type', 'hand_type_valid', 'focal', 'principal_pt'
    }

    instance_mapping_table = {
        'keypoints': 'keypoints',
        'keypoints_visible': 'keypoints_visible',
        'keypoints_cam': 'keypoints_cam',
    }

    label_mapping_table = {
        'keypoint_weights': 'keypoint_weights',
        'root_depth_weight': 'root_depth_weight',
        'type_weight': 'type_weight',
        'root_depth': 'root_depth',
        'type': 'type'
    }

    def __init__(self,
                 image_size: Tuple[int, int] = [256, 256],
                 root_heatmap_size: int = 64,
                 heatmap_size: Tuple[int, int, int] = [64, 64, 64],
                 heatmap3d_depth_bound: float = 400.0,
                 heatmap_size_root: int = 64,
                 root_depth_bound: float = 400.0,
                 depth_size: int = 64,
                 use_different_joint_weights: bool = False,
                 sigma: int = 2,
                 joint_indices: Optional[list] = None,
                 max_bound: float = 1.0):
        super().__init__()

        self.image_size = np.array(image_size)
        self.root_heatmap_size = root_heatmap_size
        self.heatmap_size = np.array(heatmap_size)
        self.heatmap3d_depth_bound = heatmap3d_depth_bound
        self.heatmap_size_root = heatmap_size_root
        self.root_depth_bound = root_depth_bound
        self.depth_size = depth_size
        self.use_different_joint_weights = use_different_joint_weights

        self.sigma = sigma
        self.joint_indices = joint_indices
        self.max_bound = max_bound
        self.scale_factor = (np.array(image_size) /
                             heatmap_size[:-1]).astype(np.float32)

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray],
        dataset_keypoint_weights: Optional[np.ndarray],
        rel_root_depth: np.float32,
        rel_root_valid: np.float32,
        hand_type: np.ndarray,
        hand_type_valid: np.ndarray,
        focal: np.ndarray,
        principal_pt: np.ndarray,
    ) -> dict:
        """Encoding keypoints from input image space to input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
            keypoints_visible (np.ndarray, optional): Keypoint visibilities in
                shape (N, K).
            dataset_keypoint_weights (np.ndarray, optional): Keypoints weight
                in shape (K, ).
            rel_root_depth (np.float32): Relative root depth.
            rel_root_valid (float): Validity of relative root depth.
            hand_type (np.ndarray): Type of hand encoded as a array.
            hand_type_valid (np.ndarray): Validity of hand type.
            focal (np.ndarray): Focal length of camera.
            principal_pt (np.ndarray): Principal point of camera.

        Returns:
            encoded (dict): Contains the following items:

                - heatmaps (np.ndarray): The generated heatmap in shape
                  (K * D, H, W) where [W, H, D] is the `heatmap_size`
                - keypoint_weights (np.ndarray): The target weights in shape
                  (N, K)
                - root_depth (np.ndarray): Encoded relative root depth
                - root_depth_weight (np.ndarray): The weights of relative root
                  depth
                - type (np.ndarray): Encoded hand type
                - type_weight (np.ndarray): The weights of hand type
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:-1], dtype=np.float32)

        if self.use_different_joint_weights:
            assert dataset_keypoint_weights is not None, 'To use different ' \
                'joint weights,`dataset_keypoint_weights` cannot be None.'

        heatmaps, keypoint_weights = generate_3d_gaussian_heatmaps(
            heatmap_size=self.heatmap_size,
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            sigma=self.sigma,
            image_size=self.image_size,
            heatmap3d_depth_bound=self.heatmap3d_depth_bound,
            joint_indices=self.joint_indices,
            max_bound=self.max_bound,
            use_different_joint_weights=self.use_different_joint_weights,
            dataset_keypoint_weights=dataset_keypoint_weights)

        rel_root_depth = (rel_root_depth / self.root_depth_bound +
                          0.5) * self.heatmap_size_root
        rel_root_valid = rel_root_valid * (rel_root_depth >= 0) * (
            rel_root_depth <= self.heatmap_size_root)

        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            root_depth=rel_root_depth * np.ones(1, dtype=np.float32),
            type=hand_type,
            type_weight=hand_type_valid,
            root_depth_weight=rel_root_valid * np.ones(1, dtype=np.float32))
        return encoded

    def decode(self, heatmaps: np.ndarray, root_depth: np.ndarray,
               hand_type: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            heatmaps (np.ndarray): Heatmaps in shape (K, D, H, W)
            root_depth (np.ndarray): Root depth prediction.
            hand_type (np.ndarray): Hand type prediction.

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmap3d = heatmaps.copy()

        keypoints, scores = get_heatmap_3d_maximum(heatmap3d)

        # transform keypoint depth to camera space
        keypoints[..., 2] = (keypoints[..., 2] / self.depth_size -
                             0.5) * self.heatmap3d_depth_bound

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        # Restore the keypoint scale
        keypoints[..., :2] = keypoints[..., :2] * self.scale_factor

        # decode relative hand root depth
        # transform relative root depth to camera space
        rel_root_depth = ((root_depth / self.root_heatmap_size - 0.5) *
                          self.root_depth_bound)

        hand_type = (hand_type > 0).reshape(1, -1).astype(int)

        return keypoints, scores, rel_root_depth, hand_type
