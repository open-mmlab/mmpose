# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (batch_heatmap_nms, generate_displacement_heatmap,
                    generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_root)


@KEYPOINT_CODECS.register_module()
class SPR(BaseKeypointCodec):
    """Encode/decode keypoints with Structured Pose Representation (SPR).

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
        sigma: Optional[Union[float, Tuple[float]]] = None,
        generate_keypoint_heatmaps: bool = False,
        root_type: str = 'kpt_center',
        minimal_diagonal_length: Union[int, float] = 5,
        background_weight: float = 0.1,
        decode_nms_kernel: int = 5,
        decode_max_instances: int = 30,
        decode_thr: float = 0.01,
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
        self.decode_thr = decode_thr

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
            if not isinstance(sigma, (tuple, list)):
                sigma = (sigma, )
            if generate_keypoint_heatmaps:
                assert len(sigma) == 2, 'sigma for keypoints must be given ' \
                                        'if `generate_keypoint_heatmaps` ' \
                                        'is True. e.g. sigma=(4, 2)'
            self.sigma = sigma

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
                                                 self.root_type)
        diagonal_lengths = get_diagonal_lengths(_keypoints, keypoints_visible)

        # discard the small instances
        roots_visible[diagonal_lengths < self.minimal_diagonal_length] = 0

        # generate heatmaps
        heatmaps, _ = generate_gaussian_heatmaps(
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

            heatmaps = np.concatenate((keypoint_heatmaps, heatmaps), axis=0)
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
        """Decode the keypoint coordinates from heatmaps and displacements. The
        decoded keypoint coordinates are in the input image space.

        Args:
            heatmaps (Tensor): Encoded root and keypoints (optional) heatmaps
                in shape (1, H, W) or (K+1, H, W)
            displacements (Tensor): Encoded keypoints displacement fields
                in shape (K*D, H, W)

        Returns:
            tuple:
            - keypoints (Tensor): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (tuple):
                - root_scores (Tensor): The root scores in shape (N, )
                - keypoint_scores (Tensor): The keypoint scores in
                    shape (N, K). If keypoint heatmaps are not generated,
                    `keypoint_scores` will be `None`
        """
        # heatmaps, displacements = encoded
        _k, h, w = displacements.shape
        k = _k // 2
        displacements = displacements.view(k, 2, h, w)

        # convert displacements to a dense keypoint prediction
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        regular_grid = torch.stack([x, y], dim=0).to(displacements)
        posemaps = (regular_grid[None] + displacements).flatten(2)

        # find local maximum on root heatmap
        root_heatmap_peaks = batch_heatmap_nms(heatmaps[None, -1:],
                                               self.decode_nms_kernel)
        root_scores, pos_idx = root_heatmap_peaks.flatten().topk(
            self.decode_max_instances)
        mask = root_scores > self.decode_thr
        root_scores, pos_idx = root_scores[mask], pos_idx[mask]

        keypoints = posemaps[:, :, pos_idx].permute(2, 0, 1).contiguous()

        if self.generate_keypoint_heatmaps and heatmaps.shape[0] == 1 + k:
            # compute scores for each keypoint
            keypoint_scores = self.get_keypoint_scores(heatmaps[:k], keypoints)
        else:
            keypoint_scores = None

        keypoints = torch.cat([
            kpt * self.scale_factor[i]
            for i, kpt in enumerate(keypoints.split(1, -1))
        ],
                              dim=-1)
        return keypoints, (root_scores, keypoint_scores)

    def get_keypoint_scores(self, heatmaps: Tensor, keypoints: Tensor):
        """Calculate the keypoint scores with keypoints heatmaps and
        coordinates.

        Args:
            heatmaps (Tensor): Keypoint heatmaps in shape (K, H, W)
            keypoints (Tensor): Keypoint coordinates in shape (N, K, D)

        Returns:
            Tensor: Keypoint scores in [N, K]
        """
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
