# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmyolo.registry import MODELS
from torch import Tensor

from mmpose.datasets.datasets.utils import parse_pose_metainfo


@MODELS.register_module()
class OksLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        loss_weight (float): Weight for the loss.
    """

    def __init__(self,
                 metainfo: Optional[str] = None,
                 loss_weight: float = 1.0):
        super().__init__()

        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get('sigmas', None)
            if sigmas is not None:
                self.register_buffer('sigmas', torch.as_tensor(sigmas))
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Tensor,
                bboxes: Optional[Tensor] = None) -> Tensor:
        oks = self.compute_oks(output, target, target_weights, bboxes)
        loss = 1 - oks
        return loss * self.loss_weight

    def compute_oks(self,
                    output: Tensor,
                    target: Tensor,
                    target_weights: Tensor,
                    bboxes: Optional[Tensor] = None) -> Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 2, where N
                is batch size, k is the number of keypoints, and 2 are the
                xy coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            target_weights (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 4,
                where 4 are the xyxy coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """

        dist = torch.norm(output - target, dim=-1)

        if hasattr(self, 'sigmas'):
            sigmas = self.sigmas.reshape(*((1, ) * (dist.ndim - 1)), -1)
            dist = dist / sigmas
        if bboxes is not None:
            area = torch.norm(bboxes[..., 2:] - bboxes[..., :2], dim=-1)
            dist = dist / area.clip(min=1e-8).unsqueeze(-1)

        return (torch.exp(-dist.pow(2) / 2) * target_weights).sum(
            dim=-1) / target_weights.sum(dim=-1).clip(min=1e-8)
