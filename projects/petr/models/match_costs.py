# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData
from torch import Tensor

from .losses import OksLoss


@TASK_UTILS.register_module()
class KptL1Cost(BaseMatchCost):
    """This class computes the L1 cost between predicted and ground truth
    keypoints.

    The cost is computed based on the normalized difference between the
    keypoints. The keypoints visibility is also taken into account while
    calculating the cost.
    """

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute the L1 cost between predicted and ground truth keypoints.

        Args:
            pred_instances (InstanceData): Predicted instances data.
            gt_instances (InstanceData): Ground truth instances data.
            img_meta (dict, optional): Meta data of the image. Defaults
                to None.

        Returns:
            Tensor: L1 cost between predicted and ground truth keypoints.
        """

        # Extract keypoints from predicted and ground truth instances
        pred_keypoints = pred_instances.keypoints
        gt_keypoints = gt_instances.keypoints

        # Get the visibility of keypoints and normalize it
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_keypoints_visible = gt_keypoints_visible / (
            2 * gt_keypoints_visible.sum(dim=1, keepdim=True) + 1e-8)

        # Normalize keypoints based on image shape
        img_h, img_w = img_meta['img_shape']
        factor = gt_keypoints.new_tensor([img_w, img_h]).reshape(1, 1, 2)
        gt_keypoints = (gt_keypoints / factor).unsqueeze(0)
        gt_keypoints_visible = gt_keypoints_visible.unsqueeze(0).unsqueeze(-1)
        pred_keypoints = (pred_keypoints / factor).unsqueeze(1)

        # Calculate L1 cost considering visibility of keypoints
        diff = (pred_keypoints - gt_keypoints) * gt_keypoints_visible
        kpt_cost = diff.flatten(2).norm(dim=2, p=1)

        return kpt_cost * self.weight


@TASK_UTILS.register_module()
class OksCost(BaseMatchCost, OksLoss):
    """This class computes the OKS (Object Keypoint Similarity) cost between
    predicted and ground truth keypoints.

    It normalizes keypoints based on image shape, then calculates the OKS using
    a method from the OksLoss class. It also includes visibility and bounding
    box information in the calculation.
    """

    def __init__(self, metainfo: Optional[str] = None, weight: float = 1.0):
        OksLoss.__init__(self, metainfo, weight)
        self.weight = self.loss_weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute the OKS cost between predicted and ground truth keypoints.

        Args:
            pred_instances (InstanceData): Predicted instances data.
            gt_instances (InstanceData): Ground truth instances data.
            img_meta (dict, optional): Meta data of the image. Defaults
                to None.

        Returns:
            Tensor: OKS cost between predicted and ground truth keypoints.
        """

        # Extract keypoints and bounding boxes
        pred_keypoints = pred_instances.keypoints
        gt_keypoints = gt_instances.keypoints
        gt_bboxes = gt_instances.bboxes
        gt_keypoints_visible = gt_instances.keypoints_visible

        # Normalize keypoints and bounding boxes based on image shape
        img_h, img_w = img_meta['img_shape']
        factor = gt_keypoints.new_tensor([img_w, img_h]).reshape(1, 1, 2)
        gt_keypoints = (gt_keypoints / factor).unsqueeze(0)
        pred_keypoints = (pred_keypoints / factor).unsqueeze(1)
        gt_bboxes = (gt_bboxes.reshape(-1, 2, 2) / factor).reshape(1, -1, 4)

        # Calculate OKS cost
        kpt_cost = self.compute_oks(pred_keypoints, gt_keypoints,
                                    gt_keypoints_visible, gt_bboxes)
        kpt_cost = -kpt_cost
        return kpt_cost * self.weight
