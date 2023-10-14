# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import Tensor

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import TASK_UTILS
from mmpose.structures.bbox import bbox_overlaps


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BBoxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


@TASK_UTILS.register_module()
class PoseOKS:
    """OKS score Calculator."""

    def __init__(self,
                 metainfo: Optional[str] = 'configs/_base_/datasets/coco.py'):

        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get('sigmas', None)
            if sigmas is not None:
                self.sigmas = torch.as_tensor(sigmas)

    @torch.no_grad()
    def __call__(self,
                 output: Tensor,
                 target: Tensor,
                 target_weights: Tensor,
                 areas: Tensor,
                 eps: float = 1e-8) -> Tensor:

        dist = torch.norm(output - target, dim=-1)
        areas = areas.reshape(*((1, ) * (dist.ndim - 2)), -1, 1)
        dist = dist / areas.pow(0.5).clip(min=eps)

        if hasattr(self, 'sigmas'):
            if self.sigmas.device != dist.device:
                self.sigmas = self.sigmas.to(dist.device)
            sigmas = self.sigmas.reshape(*((1, ) * (dist.ndim - 1)), -1)
            dist = dist / (sigmas * 2)

        target_weights = target_weights / target_weights.sum(
            dim=-1, keepdims=True).clip(min=eps)
        oks = (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1)
        return oks
