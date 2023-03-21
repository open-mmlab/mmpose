# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.models.utils import filter_scores_and_topk
from mmdet.utils import ConfigType, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.model import ModuleList, bias_init_with_prob
from mmengine.structures import InstanceData
from mmyolo.models.dense_heads import YOLOXHead, YOLOXHeadModule
from mmyolo.registry import MODELS
from torch import Tensor

from .utils import OutputSaveFunctionWrapper, OutputSaveObjectWrapper


@MODELS.register_module()
class YOLOXPoseHeadModule(YOLOXHeadModule):
    """YOLOXPoseHeadModule serves as a head module for `YOLOX-Pose`.

    In comparison to `YOLOXHeadModule`, this module introduces branches for
    keypoint prediction.
    """

    def __init__(self, num_keypoints: int, *args, **kwargs):
        self.num_keypoints = num_keypoints
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initializes the layers in the head module."""
        super()._init_layers()

        # The pose branch requires additional layers for precise regression
        self.stacked_convs *= 2

        # Create separate layers for each level of feature maps
        pose_convs, offsets_preds, vis_preds = [], [], []
        for _ in self.featmap_strides:
            pose_convs.append(self._build_stacked_convs())
            offsets_preds.append(
                nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            vis_preds.append(
                nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

        self.multi_level_pose_convs = ModuleList(pose_convs)
        self.multi_level_conv_offsets = ModuleList(offsets_preds)
        self.multi_level_conv_vis = ModuleList(vis_preds)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()

        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_vis in self.multi_level_conv_vis:
            conv_vis.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network."""
        offsets_pred, vis_pred = [], []
        for i in range(len(x)):
            pose_feat = self.multi_level_pose_convs[i](x[i])
            offsets_pred.append(self.multi_level_conv_offsets[i](pose_feat))
            vis_pred.append(self.multi_level_conv_vis[i](pose_feat))

        return (*super().forward(x), offsets_pred, vis_pred)


@MODELS.register_module()
class YOLOXPoseHead(YOLOXHead):
    """YOLOXPoseHead head used in `YOLO-Pose.

    <https://arxiv.org/abs/2204.06806>`_.

    Args:
        loss_pose (ConfigDict, optional): Config of keypoint OKS loss.
    """

    def __init__(
        self,
        loss_pose: Optional[ConfigType] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_pose = MODELS.build(loss_pose)
        self.num_keypoints = self.head_module.num_keypoints

        # set up buffers to save variables generated in methods of
        # the class's base class.
        self._log = defaultdict(list)
        self.sampler = OutputSaveObjectWrapper(self.sampler)

        # ensure that the `sigmas` in self.assigner.oks_calculator
        # is on the same device as the model
        if hasattr(self.assigner, 'oks_calculator'):
            self.add_module('assigner_oks_calculator',
                            self.assigner.oks_calculator)

    def _clear(self):
        """Clear variable buffers."""
        self.sampler.clear()
        self._log.clear()

    def loss_by_feat(self,
                     cls_scores: Sequence[Tensor],
                     bbox_preds: Sequence[Tensor],
                     objectnesses: Sequence[Tensor],
                     kpt_preds: Sequence[Tensor],
                     vis_preds: Sequence[Tensor],
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None
                     ) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        In addition to the base class method, keypoint losses are also
        calculated in this method.
        """

        self._clear()

        # collect keypoints coordinates and visibility from model predictions
        kpt_preds = torch.cat([
            kpt_pred.flatten(2).permute(0, 2, 1).contiguous()
            for kpt_pred in kpt_preds
        ],
                              dim=1)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        grid_priors = torch.cat(mlvl_priors)

        flatten_kpts = self.decode_pose(grid_priors[..., :2], kpt_preds,
                                        grid_priors[..., 2])

        vis_preds = torch.cat([
            vis_pred.flatten(2).permute(0, 2, 1).contiguous()
            for vis_pred in vis_preds
        ],
                              dim=1)

        # compute detection losses and collect targets for keypoints
        # predictions simultaneously
        self._log['pred_keypoints'] = list(flatten_kpts.detach().split(
            1, dim=0))
        self._log['pred_keypoints_vis'] = list(vis_preds.detach().split(
            1, dim=0))

        losses = super().loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                      batch_gt_instances, batch_img_metas,
                                      batch_gt_instances_ignore)

        kpt_targets, vis_targets = [], []
        sampling_results = self.sampler.log['sample']
        sampling_result_idx = 0
        for gt_instances in batch_gt_instances:
            if len(gt_instances) > 0:
                sampling_result = sampling_results[sampling_result_idx]
                kpt_target = gt_instances['keypoints'][
                    sampling_result.pos_assigned_gt_inds]
                vis_target = gt_instances['keypoints_visible'][
                    sampling_result.pos_assigned_gt_inds]
                sampling_result_idx += 1
                kpt_targets.append(kpt_target)
                vis_targets.append(vis_target)

        if len(kpt_targets) > 0:
            kpt_targets = torch.cat(kpt_targets, 0)
            vis_targets = torch.cat(vis_targets, 0)

        # compute keypoint losses
        if len(kpt_targets) > 0:
            vis_targets = (vis_targets > 0).float()
            pos_masks = torch.cat(self._log['foreground_mask'], 0)
            bbox_targets = torch.cat(self._log['bbox_target'], 0)
            loss_kpt = self.loss_pose(
                flatten_kpts.view(-1, self.num_keypoints, 2)[pos_masks],
                kpt_targets, vis_targets, bbox_targets)
            loss_vis = self.loss_cls(
                vis_preds.view(-1, self.num_keypoints)[pos_masks],
                vis_targets) / vis_targets.sum()
        else:
            loss_kpt = kpt_preds.sum() * 0
            loss_vis = vis_preds.sum() * 0

        losses.update(dict(loss_kpt=loss_kpt, loss_vis=loss_vis))

        self._clear()
        return losses

    @torch.no_grad()
    def _get_targets_single(self,
                            priors: Tensor,
                            cls_preds: Tensor,
                            decoded_bboxes: Tensor,
                            objectness: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None
                            ) -> tuple:
        """Calculates targets for a single image, and saves them to the log.

        This method is similar to the _get_targets_single method in the base
        class, but additionally saves the foreground mask and bbox targets to
        the log.
        """

        # Construct a combined representation of bboxes and keypoints to
        # ensure keypoints are also involved in the positive sample
        # assignment process
        kpt = self._log['pred_keypoints'].pop(0).squeeze(0)
        kpt_vis = self._log['pred_keypoints_vis'].pop(0).squeeze(0)
        kpt = torch.cat((kpt, kpt_vis.unsqueeze(-1)), dim=-1)
        decoded_bboxes = torch.cat((decoded_bboxes, kpt.flatten(1)), dim=1)

        targets = super()._get_targets_single(priors, cls_preds,
                                              decoded_bboxes, objectness,
                                              gt_instances, img_meta,
                                              gt_instances_ignore)
        self._log['foreground_mask'].append(targets[0])
        self._log['bbox_target'].append(targets[3])
        return targets

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        kpt_preds: Optional[List[Tensor]] = None,
                        vis_preds: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into bbox
        and keypoint results.

        In addition to the base class method, keypoint predictions are also
        calculated in this method.
        """

        # calculate predicted bboxes and get the kept instances indices
        with OutputSaveFunctionWrapper(
                filter_scores_and_topk,
                super().predict_by_feat.__globals__) as outputs_1:
            with OutputSaveFunctionWrapper(
                    batched_nms,
                    super()._bbox_post_process.__globals__) as outputs_2:
                results_list = super().predict_by_feat(cls_scores, bbox_preds,
                                                       objectnesses,
                                                       batch_img_metas, cfg,
                                                       rescale, with_nms)
                keep_indices_topk = [out[2] for out in outputs_1]
                keep_indices_nms = [out[1] for out in outputs_2]

        num_imgs = len(batch_img_metas)

        # recover keypoints coordinates from model predictions
        featmap_sizes = [vis_pred.shape[2:] for vis_pred in vis_preds]
        priors = torch.cat(self.mlvl_priors)
        strides = [
            priors.new_full((featmap_size.numel() * self.num_base_priors, ),
                            stride) for featmap_size, stride in zip(
                                featmap_sizes, self.featmap_strides)
        ]
        strides = torch.cat(strides)
        kpt_preds = torch.cat([
            kpt_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_keypoints * 2) for kpt_pred in kpt_preds
        ],
                              dim=1)
        flatten_decoded_kpts = self.decode_pose(priors, kpt_preds, strides)

        vis_preds = torch.cat([
            vis_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_keypoints) for vis_pred in vis_preds
        ],
                              dim=1).sigmoid()

        # select keypoints predictions according to bbox scores and nms result
        keep_indices_nms_idx = 0
        for pred_instances, kpts, kpts_vis, img_meta, keep_idxs \
            in zip(
                results_list, flatten_decoded_kpts, vis_preds,
                batch_img_metas, keep_indices_topk):

            pred_instances.bbox_scores = pred_instances.scores

            if len(pred_instances) == 0:
                pred_instances.keypoints = kpts[:0]
                pred_instances.keypoint_scores = kpts_vis[:0]
                continue

            kpts = kpts[keep_idxs]
            kpts_vis = kpts_vis[keep_idxs]

            if rescale:
                pad_param = img_meta.get('img_meta', None)
                scale_factor = img_meta['scale_factor']
                if pad_param is not None:
                    kpts -= kpts.new_tensor([pad_param[2], pad_param[0]])
                kpts /= kpts.new_tensor(scale_factor).repeat(
                    (1, self.num_keypoints, 1))

            keep_idxs_nms = keep_indices_nms[keep_indices_nms_idx]
            kpts = kpts[keep_idxs_nms]
            kpts_vis = kpts_vis[keep_idxs_nms]
            keep_indices_nms_idx += 1

            pred_instances.keypoints = kpts
            pred_instances.keypoint_scores = kpts_vis

        return results_list

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples,
                rescale: bool = False):
        predictions = [
            pred_instances.numpy() for pred_instances in super().predict(
                x, batch_data_samples, rescale)
        ]
        return predictions

    def decode_pose(self, grids: torch.Tensor, offsets: torch.Tensor,
                    strides: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression offsets to keypoints.

        Args:
            grids (torch.Tensor): The coordinates of the feature map grids.
            offsets (torch.Tensor): The predicted offset of each keypoint
                relative to its corresponding grid.
            strides (torch.Tensor | int): The stride of the feature map for
                each instance.

        Returns:
            torch.Tensor: The decoded keypoints coordinates.
        """

        if isinstance(strides, int):
            strides = torch.tensor([strides]).to(offsets)

        strides = strides.reshape(1, -1, 1, 1)
        offsets = offsets.reshape(*offsets.shape[:2], -1, 2)
        xy_coordinates = (offsets[..., :2] * strides) + grids.unsqueeze(1)
        return xy_coordinates

    @staticmethod
    def gt_instances_preprocess(batch_gt_instances: List[InstanceData], *args,
                                **kwargs) -> List[InstanceData]:
        return batch_gt_instances
