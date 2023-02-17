# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_bbox, get_instance_root)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints


@KEYPOINT_CODECS.register_module()
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

        heatmap_min_overlap (float): Minimum overlap rate among instances.
            Used when calculating sigmas for instances. Defaults to 0.7
        background_weight (float): Loss weight of background pixels.
            Defaults to 0.1
        encode_max_instances (int): The maximum number of instances
            to encode for each sample. Defaults to 30

    .. _`CID`: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_
    Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_
    CVPR_2022_paper.html
    """

    # DecoupledHeatmap requires bounding boxes to determine the size of each
    # instance, so that it can assign varying sigmas based on their size
    auxiliary_encode_keys = {'bbox'}

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
        bbox: np.ndarray,
    ) -> np.ndarray:
        """Get sigma values for each instance according to their size.

        Args:
            bbox (np.ndarray): Bounding box in shape (N, 4, 2)

        Returns:
            np.ndarray: Array containing the sigma values for each instance.
        """
        sigmas = np.zeros((bbox.shape[0], ), dtype=np.float32)

        heights = np.sqrt(np.power(bbox[:, 0] - bbox[:, 1], 2).sum(axis=-1))
        widths = np.sqrt(np.power(bbox[:, 0] - bbox[:, 2], 2).sum(axis=-1))

        for i in range(bbox.shape[0]):
            h, w = heights[i], widths[i]

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

            sigmas[i] = min(r1, r2, r3) / 3

        return sigmas

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               bbox: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            bbox (np.ndarray): Bounding box in shape (N, 8) which includes
                coordinates of 4 corners.

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
        if bbox is None:
            # generate pseudo bbox via visible keypoints
            bbox = get_instance_bbox(keypoints, keypoints_visible)
            bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
            # corner order: left_top, left_bottom, right_top, right_bottom
            bbox[:, 1:3, 0] = bbox[:, 0:2, 0]

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor
        _bbox = bbox.reshape(-1, 4, 2) / self.scale_factor

        # compute the root and scale of each instance
        roots, roots_visible = get_instance_root(_keypoints, keypoints_visible,
                                                 self.root_type)

        sigmas = self._get_instance_wise_sigmas(_bbox)

        # generate global heatmaps
        heatmaps, keypoint_weights = generate_gaussian_heatmaps(
            heatmap_size=self.heatmap_size,
            keypoints=np.concatenate((_keypoints, roots[:, None]), axis=1),
            keypoints_visible=np.concatenate(
                (keypoints_visible, roots_visible[:, None]), axis=1),
            sigma=sigmas)
        roots_visible = keypoint_weights[:, -1]

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

        if len(inst_indices) > 0:
            inst_heatmaps = np.concatenate(inst_heatmaps)
            inst_heatmap_weights = np.concatenate(inst_heatmap_weights)
            inst_roots = np.array(inst_roots, dtype=np.int32)
        else:
            inst_heatmaps = np.empty((0, *self.heatmap_size[::-1]))
            inst_heatmap_weights = np.empty((0, ))
            inst_roots = np.empty((0, 2), dtype=np.int32)

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
            instance_heatmaps (np.ndarray): Heatmaps in shape (N, K, H, W)
            instance_scores (np.ndarray): Confidence of instance roots
                prediction in shape (N, 1)

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
