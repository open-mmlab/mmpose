# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_root)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints


@KEYPOINT_CODECS.register_module(force=True)
class DecoupledHeatmap(BaseKeypointCodec):
    """Encode/decode keypoints with the method introduced in the paper CID.

    See the paper Contextual Instance Decoupling for Robust Multi-Person
    Pose Estimation`_ by Wang et al (2022) for details

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:
        - heatmaps (np.ndarray): The coupled heatmap in shape
            (1+K, H, W) where [W, H] is the `heatmap_size`.
        - instance_heatmaps (np.ndarray): The decoupled heatmap in shape
            (M*K, H, W) where M is the number of instances.
        - keypoint_weights (np.ndarray): The weight for heatmaps in shape
            (M*K).
        - instance_coords (np.ndarray): The coordinates of instance roots
            in shape (M, 2)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        root_type (str): The method to generate the instance root. Options
            are:

            - ``'kpt_center'``: Average coordinate of all visible keypoints.
            - ``'bbox_center'``: Center point of bounding boxes outlined by
                all visible keypoints.

            Defaults to ``'kpt_center'``

        heatmap_min_overlap (float): The threshold of diagonal
            length of instance bounding box. Small instances will not be
            used in training. Defaults to 32
        background_weight (float): Loss weight of background pixels.
            Defaults to 0.1
        encode_max_instances (int): The maximum number of instances
            to encode for each sample. Defaults to 30

    .. _`CID`: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_
    Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_
    CVPR_2022_paper.html
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        root_type: str = 'kpt_center',
        heatmap_min_overlap: float = 0.7,
        encode_max_instances: int = 30,
    ):
        super().__init__()

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.root_type = root_type
        self.encode_max_instances = encode_max_instances
        self.heatmap_min_overlap = heatmap_min_overlap

        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def _get_instance_wise_sigmas(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get sigma values for each instance according to their size.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            np.ndarray: Array containing the sigma values for each instance.
        """
        sigmas = np.zeros((keypoints.shape[0], ), dtype=np.float32)

        for i in range(keypoints.shape[0]):
            # collect visible keypoints
            if keypoints_visible is not None:
                visible_keypoints = keypoints[i][keypoints_visible[i] > 0]
            else:
                visible_keypoints = keypoints[i]
            if visible_keypoints.size == 0:
                continue

            w, h = visible_keypoints.max(axis=0) - visible_keypoints.min(
                axis=0)

            # compute sigma for each instance
            # condition 1
            a1, b1 = 1, h + w
            c1 = w * h * (1 - self.heatmap_min_overlap) / (
                1 + self.heatmap_min_overlap)
            sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
            r1 = (b1 + sq1) / 2

            # condition 2
            a2 = 4
            b2 = 2 * (h + w)
            c2 = (1 - self.heatmap_min_overlap) * w * h
            sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
            r2 = (b2 + sq2) / 2

            # condition 3
            a3 = 4 * self.heatmap_min_overlap
            b3 = -2 * self.heatmap_min_overlap * (h + w)
            c3 = (self.heatmap_min_overlap - 1) * w * h
            sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
            r3 = (b3 + sq3) / 2

            sigmas[i] = max(0, min(r1, r2, r3) / 3)

        return sigmas

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The coupled heatmap in shape
                (1+K, H, W) where [W, H] is the `heatmap_size`.
            - instance_heatmaps (np.ndarray): The decoupled heatmap in shape
                (N*K, H, W) where M is the number of instances.
            - keypoint_weights (np.ndarray): The weight for heatmaps in shape
                (N*K).
            - instance_coords (np.ndarray): The coordinates of instance roots
                in shape (N, 2)
        """

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor

        # compute the root and scale of each instance
        roots, roots_visible = get_instance_root(_keypoints, keypoints_visible,
                                                 self.root_type,
                                                 self.heatmap_size)

        sigmas = self._get_instance_wise_sigmas(_keypoints, keypoints_visible)

        # generate global heatmaps
        heatmaps, _ = generate_gaussian_heatmaps(
            heatmap_size=self.heatmap_size,
            keypoints=np.concatenate((_keypoints, roots[:, None]), axis=1),
            keypoints_visible=np.concatenate(
                (keypoints_visible, roots_visible[:, None]), axis=1),
            sigma=sigmas)

        # select instances
        inst_roots, inst_indices = [], []
        diagonal_lengths = get_diagonal_lengths(_keypoints, keypoints_visible)
        for i in np.argsort(diagonal_lengths):
            if roots_visible[i] < 1:
                continue
            # rand root point in 3x3 grid
            x, y = roots[i] + np.random.randint(-1, 2, (2, ))
            x = max(0, min(x, self.heatmap_size[0] - 1))
            y = max(0, min(y, self.heatmap_size[1] - 1))
            if (x, y) not in inst_roots:
                inst_roots.append((x, y))
                inst_indices.append(i)
        if len(inst_indices) > self.encode_max_instances:
            rand_indices = random.sample(
                range(len(inst_indices)), self.encode_max_instances)
            inst_roots = [inst_roots[i] for i in rand_indices]
            inst_indices = [inst_indices[i] for i in rand_indices]

        # generate instance-wise heatmaps
        inst_heatmaps, inst_heatmap_weights = [], []
        for i in inst_indices:
            inst_heatmap, inst_heatmap_weight = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints[i:i + 1],
                keypoints_visible=keypoints_visible[i:i + 1],
                sigma=sigmas[i].item())
            inst_heatmaps.append(inst_heatmap)
            inst_heatmap_weights.append(inst_heatmap_weight)

        inst_heatmaps = np.concatenate(inst_heatmaps)
        inst_heatmap_weights = np.concatenate(inst_heatmap_weights)
        inst_roots = np.array(inst_roots, dtype=np.int32)

        encoded = dict(
            heatmaps=heatmaps,
            instance_heatmaps=inst_heatmaps,
            keypoint_weights=inst_heatmap_weights,
            instance_coords=inst_roots)

        return encoded

    def decode(self, instance_heatmaps: np.ndarray,
               instance_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from decoupled heatmaps. The decoded
        keypoint coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (N, K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        keypoints, keypoint_scores = [], []

        for i in range(instance_heatmaps.shape[0]):
            heatmaps = instance_heatmaps[i].copy()
            kpts, scores = get_heatmap_maximum(heatmaps)
            keypoints.append(refine_keypoints(kpts[None], heatmaps))
            keypoint_scores.append(scores[None])

        keypoints = np.concatenate(keypoints)
        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        keypoint_scores = np.concatenate(keypoint_scores)
        keypoint_scores *= instance_scores

        return keypoints, keypoint_scores
