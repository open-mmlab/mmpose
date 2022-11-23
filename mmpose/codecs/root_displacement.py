# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (batch_heatmap_nms, generate_displacement_heatmap,
                    generate_gaussian_heatmaps)


@KEYPOINT_CODECS.register_module(force=True)
class RootDisplacement(BaseKeypointCodec):

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: Optional[Union[float, Tuple[float]]] = None,
        generate_keypoint_heatmaps: bool = False,
        root_type: str = 'kpt_center',
        minimal_diagonal_length=32,
        background_weight: float = 0.1,
        decode_nms_kernel: int = 5,
        decode_max_instances: int = 30,
        decode_score_threshold: float = 0.01,
        use_udp: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.generate_keypoint_heatmaps = generate_keypoint_heatmaps
        self.root_type = root_type
        self.minimal_diagonal_length = minimal_diagonal_length
        self.background_weight = background_weight
        self.decode_nms_kernel = decode_nms_kernel
        self.decode_max_instances = decode_max_instances
        self.decode_score_threshold = decode_score_threshold
        self.use_udp = use_udp

        if self.use_udp:
            self.scale_factor = ((np.array(input_size) - 1) /
                                 (np.array(heatmap_size) - 1)).astype(
                                     np.float32)
        else:
            self.scale_factor = (np.array(input_size) /
                                 heatmap_size).astype(np.float32)

        if sigma is None:
            sigma = (heatmap_size[0] * heatmap_size[1])**0.5 / 32
            if generate_keypoint_heatmaps:
                # sigma for root heatmap and keypoint heatmaps
                self.sigma = (sigma, sigma // 2)
            else:
                self.sigma = (sigma, )
        else:
            if isinstance(sigma, float):
                sigma = (sigma, )
            if generate_keypoint_heatmaps:
                assert len(sigma) == 2, 'sigma for keypoints must be given ' \
                                        'if `generate_keypoint_heatmaps` ' \
                                        'is True. e.g. sigma=(4, 2)'
            self.sigma = sigma

    def _get_diagonal_lengths(self,
                              keypoints: np.ndarray,
                              keypoints_visible: Optional[np.ndarray] = None
                              ) -> np.ndarray:
        # TODO: add docstring

        diagonal_length = np.zeros((keypoints.shape[0]), dtype=np.float32)
        for i in range(keypoints.shape[0]):
            if keypoints_visible is not None:
                visible_keypoints = keypoints[i][keypoints_visible[i] > 0]
            else:
                visible_keypoints = keypoints[i]
            if visible_keypoints.size == 0:
                continue

            h_w_diff = visible_keypoints.max(axis=0) - visible_keypoints.min(
                axis=0)
            diagonal_length[i] = np.sqrt(np.power(h_w_diff, 2).sum())
        return diagonal_length

    def _get_instance_root(self,
                           keypoints: np.ndarray,
                           keypoints_visible: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        # TODO: add docstring

        W, H = self.heatmap_size

        roots_coordinate = np.zeros((keypoints.shape[0], 2), dtype=np.float32)
        roots_visible = np.zeros((keypoints.shape[0]), dtype=np.float32)

        for i in range(keypoints.shape[0]):

            # collect visible keypoints
            if keypoints_visible is not None:
                visible_keypoints = keypoints[i][keypoints_visible[i] > 0]
            else:
                visible_keypoints = keypoints[i]
            if visible_keypoints.size == 0:
                continue

            # compute the instance root with visible keypoints
            if self.root_type == 'kpt_center':
                roots_coordinate[i] = visible_keypoints.mean(axis=0)
            elif self.root_type == 'bbox_center':
                roots_coordinate[i] = (visible_keypoints.max(axis=0) +
                                       visible_keypoints.min(axis=0)) / 2.0
            else:
                raise ValueError(
                    f'the value of `root_type` must be \'kpt_center\' or '
                    f'\'bbox_center\', but got \'{self.root_type}\'')

            # compute the visibility of roots
            if roots_coordinate[i][0] >= W or roots_coordinate[i][
                    1] >= H or roots_coordinate[i].min() < 0:
                roots_visible[i] = 0
            else:
                roots_visible[i] = 1

        return roots_coordinate, roots_visible

    def _get_heatmap_weights(self,
                             heatmaps,
                             fg_weight: float = 1,
                             bg_weight: float = 0):
        # TODO: add docstring
        heatmap_weights = np.ones(heatmaps.shape) * bg_weight
        heatmap_weights[heatmaps > 0] = fg_weight
        return heatmap_weights

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        # TODO: modify the docstring
        """Encode keypoints into heatmaps and position indices. Note that the
        original keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
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

        # compute the root and scale of each instance
        roots, roots_visible = self._get_instance_root(_keypoints,
                                                       keypoints_visible)
        diagonal_lengths = self._get_diagonal_lengths(_keypoints,
                                                      keypoints_visible)

        # discard the small instances
        roots_visible[diagonal_lengths < self.minimal_diagonal_length] = 0
        keypoints_visible[diagonal_lengths < self.minimal_diagonal_length] = 0

        if self.use_udp:
            raise NotImplementedError
        else:
            # generate heatmaps
            # WARNING: mask is absence here
            heatmaps, weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=roots[:, None],
                keypoints_visible=roots_visible[:, None],
                sigma=self.sigma[0])
            heatmap_weights = self._get_heatmap_weights(
                heatmaps, bg_weight=self.background_weight)

            if self.generate_keypoint_heatmaps:
                keypoint_heatmaps, _ = generate_gaussian_heatmaps(
                    heatmap_size=self.heatmap_size,
                    keypoints=_keypoints,
                    keypoints_visible=keypoints_visible,
                    sigma=self.sigma[1])

                keypoint_heatmaps_weights = self._get_heatmap_weights(
                    keypoint_heatmaps, bg_weight=self.background_weight)

                heatmaps = np.concatenate((keypoint_heatmaps, heatmaps),
                                          axis=0)
                heatmap_weights = np.concatenate(
                    (keypoint_heatmaps_weights, heatmap_weights), axis=0)

            # generate displacements
            displacements, displacement_weights = \
                generate_displacement_heatmap(
                    self.heatmap_size,
                    _keypoints,
                    keypoints_visible,
                    roots,
                    roots_visible,
                    diagonal_lengths,
                    self.sigma[0],
                )

        encoded = dict(
            heatmaps=heatmaps,
            heatmap_weights=heatmap_weights,
            displacements=displacements,
            displacement_weights=displacement_weights)

        return encoded

    def decode(self, heatmaps: Tensor,
               displacements: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints.

        Args:
            encoded (any): Encoded keypoint representation using the codec

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        # heatmaps, displacements = encoded
        _k, h, w = displacements.shape
        k = _k // 2
        displacements = displacements.view(k, 2, h, w)

        # convert displacements to a dense keypoint prediction
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        regular_grid = torch.stack([x, y], dim=0).to(displacements)
        posemaps = (regular_grid[None] - displacements).flatten(2)

        # find local maximum on root heatmap
        # optional: use a gaussian blur for heatmap
        root_heatmap_peaks = batch_heatmap_nms(heatmaps[None, -1:],
                                               self.decode_nms_kernel)
        root_scores, pos_idx = root_heatmap_peaks.flatten().topk(
            self.decode_max_instances)
        mask = root_scores > self.decode_score_threshold
        root_scores, pos_idx = root_scores[mask], pos_idx[mask]

        keypoints = posemaps[:, :, pos_idx].permute(2, 0, 1).contiguous()

        if self.generate_keypoint_heatmaps and heatmaps.shape[0] == 1 + k:
            keypoint_scores = self.get_keypoint_scores(heatmaps[:k], keypoints)
        else:
            keypoint_scores = None

        keypoints = torch.stack((keypoints[..., 0] * self.scale_factor[0],
                                 keypoints[..., 1] * self.scale_factor[1]),
                                dim=-1)
        return keypoints, root_scores, keypoint_scores

    def get_keypoint_scores(self, heatmaps: Tensor, keypoints: Tensor):
        k, h, w = heatmaps.shape
        keypoints = torch.stack((
            keypoints[..., 0] / (w - 1) * 2 - 1,
            keypoints[..., 1] / (h - 1) * 2 - 1,
        ),
                                dim=-1)
        keypoints = keypoints.transpose(0, 1).unsqueeze(1).contiguous()

        keypoint_scores = torch.nn.functional.grid_sample(
            heatmaps.unsqueeze(1), keypoints,
            padding_mode='border').view(k, -1).transpose(0, 1).contiguous()

        return keypoint_scores
