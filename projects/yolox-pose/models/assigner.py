# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, SimOTAAssigner
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData
from mmyolo.registry import MODELS, TASK_UTILS
from torch import Tensor

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class PoseSimOTAAssigner(SimOTAAssigner):

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 oks_weight: float = 0.0,
                 vis_weight: float = 0.0,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 oks_calculator: ConfigType = dict(type='OksLoss')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight

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

        cost_matrix = (~is_in_boxes_and_center) * INF

        # calculate iou
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        if self.iou_weight > 0:
            iou_cost = -torch.log(pairwise_ious + EPS)
            cost_matrix = cost_matrix + iou_cost * self.iou_weight

        # calculate oks
        pairwise_oks = self.oks_calculator.compute_oks(
            valid_pred_kpts.unsqueeze(1),  # [num_valid, -1, k, 2]
            gt_keypoints.unsqueeze(0),  # [1, num_gt, k, 2]
            gt_keypoints_visible.unsqueeze(0),  # [1, num_gt, k]
            bboxes=gt_bboxes.unsqueeze(0),  # [1, num_gt, 4]
        )  # -> [num_valid, num_gt]
        if self.oks_weight > 0:
            oks_cost = -torch.log(pairwise_oks + EPS)
            cost_matrix = cost_matrix + oks_cost * self.oks_weight

        # calculate cls
        if self.cls_weight > 0:
            gt_onehot_label = (
                F.one_hot(gt_labels.to(torch.int64),
                          pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                              num_valid, 1, 1))

            valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(
                1, num_gt, 1)
            # disable AMP autocast to avoid overflow
            with torch.cuda.amp.autocast(enabled=False):
                cls_cost = (
                    F.binary_cross_entropy(
                        valid_pred_scores.to(dtype=torch.float32),
                        gt_onehot_label,
                        reduction='none',
                    ).sum(-1).to(dtype=valid_pred_scores.dtype))
            cost_matrix = cost_matrix + cls_cost * self.cls_weight

        # calculate vis
        if self.vis_weight > 0:
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
            cost_matrix = cost_matrix + vis_cost * self.vis_weight

        # mixed metric
        pairwise_oks = pairwise_oks.pow(0.5)
        matched_pred_oks, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, pairwise_oks, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_oks
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           pairwise_oks: Tensor, num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_oks = (matching_matrix *
                            pairwise_oks).sum(1)[fg_mask_inboxes]
        return matched_pred_oks, matched_gt_inds
