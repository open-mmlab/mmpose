# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (batch_heatmap_nms, generate_displacement_heatmap,
                    generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_root)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module(force=True)
class DecoupledHeatmap(BaseKeypointCodec):
    """Encode/decode keypoints with the method introduced in the paper SPM.

    See the paper `Single-stage multi-person pose machines`_
    by Nie et al (2017) for details

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (1, H, W)
            where [W, H] is the `heatmap_size`. If the keypoint heatmap is
            generated together, the output heatmap shape is (K+1, H, W)
        - heatmap_weights (np.ndarray): The target weights for heatmaps which
            has same shape with heatmaps.
        - displacements (np.ndarray): The dense keypoint displacement in
            shape (K*2, H, W).
        - displacement_weights (np.ndarray): The target weights for heatmaps
            which has same shape with displacements.

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float or tuple, optional): The sigma values of the Gaussian
            heatmaps. If sigma is a tuple, it includes both sigmas for root
            and keypoint heatmaps. ``None`` means the sigmas are computed
            automatically from the heatmap size. Defaults to ``None``
        generate_keypoint_heatmaps (bool): Whether to generate Gaussian
            heatmaps for each keypoint. Defaults to ``False``
        root_type (str): The method to generate the instance root. Options
            are:

            - ``'kpt_center'``: Average coordinate of all visible keypoints.
            - ``'bbox_center'``: Center point of bounding boxes outlined by
                all visible keypoints.

            Defaults to ``'kpt_center'``

        minimal_diagonal_length (int or float): The threshold of diagonal
            length of instance bounding box. Small instances will not be
            used in training. Defaults to 32
        background_weight (float): Loss weight of background pixels.
            Defaults to 0.1
        decode_thr (float): The threshold of keypoint response value in
            heatmaps. Defaults to 0.01
        decode_nms_kernel (int): The kernel size of the NMS during decoding,
            which should be an odd integer. Defaults to 5
        decode_max_instances (int): The maximum number of instances
            to decode. Defaults to 30

    .. _`Single-stage multi-person pose machines`:
        https://arxiv.org/abs/1908.09220
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        root_type: str = 'kpt_center',
        heatmap_min_overlap: float = 0.7,
        encode_max_instances: int = 30,
        decode_nms_kernel: int = 5,
        decode_max_instances: int = 30,
        decode_thr: float = 0.01,
    ):
        super().__init__()

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.root_type = root_type
        self.encode_max_instances = encode_max_instances
        self.heatmap_min_overlap = heatmap_min_overlap
        self.decode_nms_kernel = decode_nms_kernel
        self.decode_max_instances = decode_max_instances
        self.decode_thr = decode_thr

        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def _get_instance_wise_sigmas(self, keypoints, keypoints_visible):
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

    def _get_heatmap_weights(self,
                             heatmaps,
                             fg_weight: float = 1,
                             bg_weight: float = 0):
        """Generate weight array for heatmaps.

        Args:
            heatmaps (np.ndarray): Root and keypoint (optional) heatmaps
            fg_weight (float): Weight for foreground pixels. Defaults to 1.0
            bg_weight (float): Weight for background pixels. Defaults to 0.0

        Returns:
            np.ndarray: Heatmap weight array in the same shape with heatmaps
        """
        heatmap_weights = np.ones(heatmaps.shape) * bg_weight
        heatmap_weights[heatmaps > 0] = fg_weight
        return heatmap_weights

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into root heatmaps and keypoint displacement
        fields. Note that the original keypoint coordinates should be in the
        input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (1, H, W) where [W, H] is the `heatmap_size`. If keypoint
                heatmaps are generated together, the shape is (K+1, H, W)
            - heatmap_weights (np.ndarray): The pixel-wise weight for heatmaps
                 which has same shape with `heatmaps`
            - displacements (np.ndarray): The generated displacement fields in
                shape (K*D, H, W). The vector on each pixels represents the
                displacement of keypoints belong to the associated instance
                from this pixel.
            - displacement_weights (np.ndarray): The pixel-wise weight for
                displacements which has same shape with `displacements`
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
        keypoint_scores *= instance_scores[:, None]

        return keypoints, keypoint_scores
