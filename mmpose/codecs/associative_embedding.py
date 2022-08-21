# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple
from itertools import product
from math import dist
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from munkres import Munkres
from torch import Tensor

from mmpose.registry import KEYPOINT_CODECS
from mmpose.utils.tensor_utils import to_numpy
from .base import BaseKeypointCodec
from .utils import (batch_heatmap_nms, generate_gaussian_heatmaps,
                    generate_udp_gaussian_heatmaps)


def _group_keypoints_by_tags(vals: np.ndarray,
                             tags: np.ndarray,
                             locs: np.ndarray,
                             keypoint_order: List[int],
                             val_thr: float,
                             tag_dist_thr: float = 1.0,
                             max_groups: Optional[int] = None):
    """Group the keypoints by tags using Munkres algorithm.

    Note:

        - keypoint number: K
        - candidate number: M
        - tag dimenssion: L
        - coordinate dimension: D
        - group number: G

    Args:
        vals (np.ndarray): The heatmap response values of keypoints in shape
            (K, M)
        tags (np.ndarray): The tags of the keypoint candidates in shape
            (K, M, L)
        locs (np.ndarray): The locations of the keypoint candidates in shape
            (K, M, D)
        keypoint_order (List[int]): The grouping order of the keypoints.
            The groupping usually starts from a keypoints around the head and
            torso, and gruadually moves out to the limbs
        val_thr (float): The threshold of the keypoint response value
        tag_dist_thr (float): The maximum allowed tag distance when matching a
            keypoint to a group. A keypoint with larger tag distance to any
            of the existing groups will initializes a new group
        max_groups (int, optional): The maximum group number. ``None`` means
            no limitation. Defaults to ``None``

    Returns:
        tuple:
        - grouped_keypoints (np.ndarray): The grouped keypoints in shape
            (G, K, D)
        - grouped_keypoint_scores (np.ndarray): The grouped keypoint scores
             in shape (G, K)
    """
    K, M, D = locs.shape[0]
    assert vals.shape == tags.shape[:2] == (K, M)
    assert len(keypoint_order) == K

    # Build Munkres instance
    munkres = Munkres()

    # Build a group pool, each group contains the keypoints of an instance
    groups = []

    Group = namedtuple('Group', field_names=['kpts', 'scores', 'tag_list'])

    def _init_group():
        """Initialize a group, which is composed of the keypoints, keypoint
        scores and the tag of each keypoint."""
        _group = Group(
            kpts=np.zeros((K, D), dtype=np.float32),
            scores=np.zeros(K, dtype=np.float32),
            tag_list=[])
        return _group

    for i in keypoint_order:
        # Get all valid candidate of the i-th keypoints
        valid = vals[i] > val_thr
        if not valid.any():
            continue

        tags_i = tags[i, valid]  # (M', L)
        vals_i = vals[i, valid]  # (M',)
        locs_i = locs[i, valid]  # (M', D)

        if len(groups) == 0:  # Initialize the group pool
            for tag, loc, val in zip(tags_i, vals_i, locs_i):
                group = _init_group()
                group.kpts[i] = loc
                group.scores[i] = val
                group.tag_list.append(tag)

                groups.append(group)

        else:  # Match keypoints to existing groups
            groups = groups[:max_groups]
            group_tags = [np.mean(g.tag_list, axis=0) for g in groups]

            # Calculate distance matrix between group tags and tag candidates
            # of the i-th keypoint
            # Shape: (M', 1, L) , (1, G, L) -> (M', G, L)
            diff = tags_i[:, None] - np.array(group_tags)[None]
            dists = np.linalg.norm(diff, ord=2, axis=2)
            num_kpts, num_groups = dists.shape[:2]

            # Experimental cost function for keypoint-group matching
            costs = np.round(dists) * 100 - vals_i
            if num_kpts > num_groups:
                padding = np.full((num_kpts, num_kpts - num_groups),
                                  1e10,
                                  dtype=np.float32)
                costs = np.concatenate((costs, padding), axis=1)

            # Match keypoints and groups by Munkres algorithm
            matches = munkres.compute(costs)
            for kpt_idx, group_idx in matches:
                if group_idx < num_groups and dist[kpt_idx,
                                                   group_idx] < tag_dist_thr:
                    # Add the keypoint to the matched group
                    group = groups[group_idx]
                else:
                    # Initialize a new group with unmatched keypoint
                    group = _init_group()
                    groups.append(group)

                group.kpts[i] = locs_i[kpt_idx]
                group.scores[i] = vals_i[kpt_idx]
                group.tag_list.append(tags_i[kpt_idx])

    groups = groups[:max_groups]
    grouped_keypoints = np.stack((g.kpts for g in groups))  # (G, K, D)
    grouped_keypoint_scores = np.stack((g.vals for g in groups))  # (G, K)

    return grouped_keypoints, grouped_keypoint_scores


@KEYPOINT_CODECS.register_module()
class AssociativeEmbedding(BaseKeypointCodec):
    """Encode/decode keypoints with the method introduced in "Associative
    Embedding". This is an asymmetric codec, where the keypoints are
    represented as gaussian heatmaps and position indices during encoding, and
    reostred from predicted heatmaps and group tags.

    See the paper `Associative Embedding: End-to-End Learning for Joint
    Detection and Grouping`_ by Newell et al (2017) for details

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - embedding tag dimension: L
        - image size: [w, h]
        - heatmap size: [W, H]

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``
        decode_nms_kernel (int): The kernel size of the NMS during decoding,
            which should be a odd integer. Defaults to 5
        decode_topk (int): The number top-k candidates of each keypoints that
            will be retrieved from the heatmaps during dedocding. Defaults to
            20
        decode_max_instances (int, optional): The maximum number of instances
            to decode. ``None`` means no limitation to the instance number.
            Defaults to ``None``

    .. _`Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping`: https://arxiv.org/abs/1611.05424
    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: Optional[float] = None,
                 use_udp: bool = False,
                 decode_nms_kernel: int = 5,
                 decode_thr: float = 0.1,
                 decode_topk: int = 20,
                 decode_max_instances: Optional[int] = None,
                 tag_per_keypoint: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.use_udp = use_udp
        self.decode_nms_kernel = decode_nms_kernel
        self.decode_thr = decode_thr
        self.decode_topk = decode_topk
        self.decode_max_instances = decode_max_instances
        self.tag_per_keypoint = tag_per_keypoint

        if sigma is None:
            sigma = (heatmap_size[0] * heatmap_size[1])**0.5 / 64
        self.sigma = sigma

        if self.use_udp:
            self.scale_factor = ((np.array(heatmap_size) - 1) /
                                 (np.array(heatmap_size) - 1)).astype(
                                     np.float32)
        else:
            self.scale_factor = (np.array(input_size) /
                                 heatmap_size).astype(np.float32)

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode keypoints into heatmaps and position indices. Note that the
        original keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            tuple:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_indices (np.ndarray): The keypoint position indices
                in shape (N, K, 2). Each keypoint's index is [i, v], where i
                is the position index in the heatmap (:math:`i=y*w+x`) and v
                is the visibility
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor

        if self.use_udp:
            heatmaps, keypoints_weights = generate_udp_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoints_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        keypoint_indices = self._encode_keypoint_indices(
            heatmap_size=self.heatmap_size,
            keypoints=_keypoints,
            keypoints_visible=keypoints_visible)

        return heatmaps, keypoint_indices, keypoints_weights

    def _encode_keypoint_indices(self, heatmap_size: Tuple[int, int],
                                 keypoints: np.ndarray,
                                 keypoints_visible: np.ndarray) -> np.ndarray:
        w, h = heatmap_size
        N, K, _ = keypoints.shape
        keypoint_indices = np.zeros((N, K, 2), dtype=np.int64)

        for n, k in product(N, K):
            x, y = (keypoints[n, k] + 0.5).astype(np.int64)
            index = y * w + x
            vis = (keypoints_visible[n, k] > 0.5 and 0 <= x < w and 0 <= y < h)
            keypoint_indices[n, k] = [index, vis]

        return keypoint_indices

    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def _get_batch_topk(self, batch_heatmaps: Tensor, batch_tags: Tensor,
                        k: int):
        """Get top-k response values from the heatmaps and corresponding tag
        values from the tagging heatmaps.

        Args:
            batch_heatmaps (Tensor): Keypoint detection heatmaps in shape
                (B, K, H, W)
            batch_tags (Tensor): Tagging heatmaps in shape (B, C, H, W), where
                the tag dim C is 2*K when using flip testing, or K otherwise
            k (int): The number of top responses to get

        Returns:
            tuple:
            - topk_vals (Tensor): Top-k response values of each heatmap in
                shape (B, K, Topk)
            - topk_tags (Tensor): The corresponding embedding tags of the
                top-k responses, in shape (B, K, Topk, L)
            - topk_locs (Tensor): The location of the top-k responses in each
                heatmap, in shape (B, K, Topk, 2) where last dimension
                represents x and y coordinates
        """
        K, W = batch_heatmaps.shape[1], batch_heatmaps.shape[3]

        # shape of topk_val, top_indices: (B, K, k)
        topk_vals, topk_indices = batch_heatmaps.flatten(-2, -1).topk(
            k, dim=-1)

        batch_tags = batch_tags.flatten(-2, -1)  # (B, K*L, H*W)
        topk_tags_per_kpts = [
            torch.gather(_tags, dim=2, index=topk_indices)
            for _tags in torch.chunk(batch_heatmaps, chunks=K, dim=1)
        ]
        topk_tags = torch.stack(topk_tags_per_kpts, dim=-1)  # (B, K, k, L)
        topk_locs = torch.stack([topk_indices % W, topk_indices // W],
                                dim=-1)  # (B, K, k, 2)

        return topk_vals, topk_tags, topk_locs

    def _group_keypoints(self, vals: np.ndarray, tags: np.ndarray,
                         locs: np.ndarray, keypoint_order: list[int],
                         val_thr: float):
        """Group keypoints into groups (each represents an instance) by tags.

        Args:
            vals (Tensor): Heatmap response values of keypoint candidates in
                shape (B, K, Topk)
            tags (Tensor): Tags of keypoint candidates in shape
                (B, K, Topk, L)
        """

    def batch_decode(self, batch_heatmaps: Tensor, batch_tags: Tensor
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Decode the keypoint coordinates from a batch of heatmaps and tagging
        heatmaps. The decoded keypoint coordinates are in the input image
        space.

        Args:
            batch_heatmaps (Tensor): Keypoint detection heatmaps in shape
                (B, K, H, W)
            batch_tags (Tensor): Tagging heatmaps in shape (B, C, H, W), where
                the tag dim C is 2*K when using flip testing, or K otherwise

        Returns:
            tuple:
            - batch_keypoints (List[np.ndarray]): Decoded keypoint coordinates
                of the batch, each is in shape (N, K, D)
            - batch_scores (List[np.ndarray]): Decoded keypoint scores of the
                batch, each is in shape (N, K). It usually represents the
                confidience of the keypoint prediction
        """
        B, K, H, W = batch_heatmaps.shape

        if not self.tag_per_keypoint:
            batch_tags = batch_tags.tile((1, K, 1, 1))

        # Heatmap NMS
        batch_heatmaps = batch_heatmap_nms(batch_heatmaps,
                                           self.decode_nms_kernel)

        # Get top-k in each heatmap and and convert to numpy
        batch_topk_vals, batch_topk_tags, batch_topk_locs = to_numpy(
            self._get_batch_topk(
                batch_heatmaps, batch_tags, k=self.decode_topk))
