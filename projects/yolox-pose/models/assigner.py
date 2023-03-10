# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, SimOTAAssigner
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData
from mmyolo.registry import MODELS, TASK_UTILS

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class PoseSimOTAAssigner(SimOTAAssigner):

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 oks_weight: float = 20.0,
                 vis_weight: float = 1.0,
                 pose_ratio: float = 0.0,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 oks_calculator: ConfigType = dict(type='OksLoss')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight
        self.pose_ratio = max(min(pose_ratio, 1), 0)

        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.oks_calculator = MODELS.build(oks_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes[..., :4]
        pred_kpts = pred_instances.bboxes[..., 4:]
        pred_kpts = pred_kpts.reshape(*pred_kpts.shape[:-1], -1, 3)
        pred_kpts_vis = pred_kpts[..., -1]
        pred_kpts = pred_kpts[..., :2]
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = pred_kpts[valid_mask]
        valid_pred_kpts_vis = pred_kpts_vis[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # calculate iou
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS)

        # calculate oks
        pairwise_oks = self.oks_calculator.compute_oks(
            valid_pred_kpts.unsqueeze(1),  # [num_valid, -1, k, 2]
            gt_keypoints.unsqueeze(0),  # [1, num_gt, k, 2]
            gt_keypoints_visible.unsqueeze(0),  # [1, num_gt, k]
            bboxes=gt_bboxes.unsqueeze(0),  # [1, num_gt, 4]
        )  # -> [num_valid, num_gt]
        oks_cost = -torch.log(pairwise_oks + EPS)

        # calculate cls
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # disable AMP autocast and calculate BCE with FP32 to avoid overflow
        with torch.cuda.amp.autocast(enabled=False):
            cls_cost = (
                F.binary_cross_entropy(
                    valid_pred_scores.to(dtype=torch.float32),
                    gt_onehot_label,
                    reduction='none',
                ).sum(-1).to(dtype=valid_pred_scores.dtype))

        # calculate vis
        valid_pred_kpts_vis = valid_pred_kpts_vis.sigmoid().unsqueeze(
            1).repeat(1, num_gt, 1)  # [num_valid, 1, k]
        gt_kpt_vis = gt_keypoints_visible.unsqueeze(
            0).float()  # [1, num_gt, k]
        with torch.cuda.amp.autocast(enabled=False):
            vis_cost = (
                F.binary_cross_entropy(
                    valid_pred_kpts_vis.to(dtype=torch.float32),
                    gt_kpt_vis.repeat(num_valid, 1, 1),
                    reduction='none',
                ).sum(-1).to(dtype=valid_pred_kpts_vis.dtype))

        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight +
            oks_cost * self.oks_weight + vis_cost * self.vis_weight +
            (~is_in_boxes_and_center) * INF)

        # mixed metric
        pairwise_ious = pairwise_ious.pow(1 -
                                          self.pose_ratio) * pairwise_oks.pow(
                                              self.pose_ratio)
        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
