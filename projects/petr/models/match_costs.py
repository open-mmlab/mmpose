# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData
from torch import Tensor

from .losses import OksLoss


@TASK_UTILS.register_module()
class KptL1Cost(BaseMatchCost):

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:

        pred_keypoints = pred_instances.keypoints
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_keypoints_visible = gt_keypoints_visible / (2 * gt_keypoints_visible.sum(dim=1, keepdim=True) + 1e-8)
        
        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_keypoints.new_tensor([img_w, img_h]).reshape(1, 1, 2)
        gt_keypoints = (gt_keypoints / factor).unsqueeze(0)
        gt_keypoints_visible = gt_keypoints_visible.unsqueeze(0).unsqueeze(-1)
        pred_keypoints = (pred_keypoints / factor).unsqueeze(1)
        
        diff = (pred_keypoints - gt_keypoints) * gt_keypoints_visible
        kpt_cost = diff.flatten(2).norm(dim=2, p=1)
        
        return kpt_cost * self.weight


@TASK_UTILS.register_module()
class OksCost(BaseMatchCost, OksLoss):

    def __init__(self, metainfo: Optional[str] = None, weight: float = 1.0):
        OksLoss.__init__(self, metainfo, weight)
        self.weight = self.loss_weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:

        pred_keypoints = pred_instances.keypoints
        gt_keypoints = gt_instances.keypoints
        gt_bboxes = gt_instances.bboxes
        gt_keypoints_visible = gt_instances.keypoints_visible

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_keypoints.new_tensor([img_w, img_h]).reshape(1, 1, 2)
        gt_keypoints = (gt_keypoints / factor).unsqueeze(0)
        pred_keypoints = (pred_keypoints / factor).unsqueeze(1)
        gt_bboxes = (gt_bboxes.reshape(-1, 2, 2) / factor).reshape(1, -1, 4)

        kpt_cost = self.compute_oks(pred_keypoints, gt_keypoints,
                                    gt_keypoints_visible, gt_bboxes)
        kpt_cost = -kpt_cost
        return kpt_cost * self.weight
