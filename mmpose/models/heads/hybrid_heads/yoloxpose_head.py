# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.evaluation.functional import nms_torch
from mmpose.models.utils import filter_scores_and_topk
from mmpose.registry import MODELS, TASK_UTILS
from mmpose.structures import PoseDataSample
from mmpose.utils import reduce_mean
from mmpose.utils.typing import (ConfigType, Features, OptSampleList,
                                 Predictions, SampleList)


class YOLOXPoseHeadModule(BaseModule):
    """YOLOXPose head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_keypoints (int): Number of keypoints defined for one instance.
         in_channels (Union[int, Sequence]): Number of channels in the input
             feature map.
        feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
         widen_factor (float): Width multiplier, multiply number of
             channels in each layer by this amount. Defaults to 1.0.
        num_groups (int): Group number of group convolution layers in keypoint
            regression branch. Defaults to 8.
        channels_per_group (int): Number of channels for each group of group
            convolution layers in keypoint regression branch. Defaults to 32.
        featmap_strides (Sequence[int]): Downsample factor of each feature
            map. Defaults to [8, 16, 32].
        conv_bias (bool or str): If specified as `auto`, it will be decided
            by the norm_cfg. Bias of conv will be set as True if `norm_cfg`
            is None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: Optional[ConfigType] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_reg_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(
                nn.Conv2d(self.feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_reg = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_reg.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def _init_pose_branch(self):
        self.conv_pose = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                in_chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        in_chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_kpt = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_kpt.append(
                nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(
                nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.out_cls, self.out_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            objectnesses (List[Tensor]): Objectness scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
        """

        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []

        for i in range(len(x)):

            cls_feat = self.conv_cls[i](x[i])
            reg_feat = self.conv_reg[i](x[i])
            pose_feat = self.conv_pose[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](reg_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            kpt_offsets.append(self.out_kpt[i](pose_feat))
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis


@MODELS.register_module()
class YOLOXPoseHead(BaseModule):

    def __init__(
        self,
        num_keypoints: int,
        head_module_cfg: Optional[ConfigType] = None,
        featmap_strides: Sequence[int] = [8, 16, 32],
        num_classes: int = 1,
        use_aux_loss: bool = False,
        assigner: ConfigType = None,
        prior_generator: ConfigType = None,
        loss_cls: Optional[ConfigType] = None,
        loss_obj: Optional[ConfigType] = None,
        loss_bbox: Optional[ConfigType] = None,
        loss_oks: Optional[ConfigType] = None,
        loss_vis: Optional[ConfigType] = None,
        loss_bbox_aux: Optional[ConfigType] = None,
        loss_kpt_aux: Optional[ConfigType] = None,
        overlaps_power: float = 1.0,
    ):
        super().__init__()

        self.featmap_sizes = None
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.use_aux_loss = use_aux_loss
        self.num_keypoints = num_keypoints
        self.overlaps_power = overlaps_power

        self.prior_generator = TASK_UTILS.build(prior_generator)
        if head_module_cfg is not None:
            head_module_cfg['featmap_strides'] = featmap_strides
            head_module_cfg['num_keypoints'] = num_keypoints
            self.head_module = YOLOXPoseHeadModule(**head_module_cfg)
        self.assigner = TASK_UTILS.build(assigner)

        # build losses
        self.loss_cls = MODELS.build(loss_cls)
        if loss_obj is not None:
            self.loss_obj = MODELS.build(loss_obj)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_oks = MODELS.build(loss_oks)
        self.loss_vis = MODELS.build(loss_vis)
        if loss_bbox_aux is not None:
            self.loss_bbox_aux = MODELS.build(loss_bbox_aux)
        if loss_kpt_aux is not None:
            self.loss_kpt_aux = MODELS.build(loss_kpt_aux)

    def forward(self, feats: Features):
        assert isinstance(feats, (tuple, list))
        return self.head_module(feats)

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect & reform predictions
        cls_scores, objectnesses, bbox_preds, kpt_offsets, \
            kpt_vis = self.forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses)
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_bbox_decoded = self.decode_bbox(flatten_bbox_preds,
                                                flatten_priors[..., :2],
                                                flatten_priors[..., -1])
        flatten_kpt_decoded = self.decode_kpt_reg(flatten_kpt_offsets,
                                                  flatten_priors[..., :2],
                                                  flatten_priors[..., -1])

        # 2. generate targets
        targets = self._get_targets(flatten_priors,
                                    flatten_cls_scores.detach(),
                                    flatten_objectness.detach(),
                                    flatten_bbox_decoded.detach(),
                                    flatten_kpt_decoded.detach(),
                                    flatten_kpt_vis.detach(),
                                    batch_data_samples)
        pos_masks, cls_targets, obj_targets, obj_weights, \
            bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets, \
            vis_targets, vis_weights, pos_areas, pos_priors, group_indices, \
            num_fg_imgs = targets

        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # 3. calculate loss
        # 3.1 objectness loss
        losses = dict()

        obj_preds = flatten_objectness.view(-1, 1)
        losses['loss_obj'] = self.loss_obj(obj_preds, obj_targets,
                                           obj_weights) / num_total_samples

        if num_pos > 0:
            # 3.2 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses['loss_bbox'] = self.loss_bbox(
                bbox_preds, bbox_targets) / num_total_samples

            # 3.3 keypoint loss
            kpt_preds = flatten_kpt_decoded.view(-1, self.num_keypoints,
                                                 2)[pos_masks]
            losses['loss_kpt'] = self.loss_oks(kpt_preds, kpt_targets,
                                               vis_targets, pos_areas)

            # 3.4 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1,
                                                 self.num_keypoints)[pos_masks]
            losses['loss_vis'] = self.loss_vis(kpt_vis_preds, vis_targets,
                                               vis_weights)

            # 3.5 classification loss
            cls_preds = flatten_cls_scores.view(-1,
                                                self.num_classes)[pos_masks]
            losses['overlaps'] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            losses['loss_cls'] = self.loss_cls(cls_preds,
                                               cls_targets) / num_total_samples

            if self.use_aux_loss:
                if hasattr(self, 'loss_bbox_aux'):
                    # 3.6 auxiliary bbox regression loss
                    bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                    losses['loss_bbox_aux'] = self.loss_bbox_aux(
                        bbox_preds_raw, bbox_aux_targets) / num_total_samples

                if hasattr(self, 'loss_kpt_aux'):
                    # 3.7 auxiliary keypoint regression loss
                    kpt_preds_raw = flatten_kpt_offsets.view(
                        -1, self.num_keypoints, 2)[pos_masks]
                    kpt_weights = vis_targets / vis_targets.size(-1)
                    losses['loss_kpt_aux'] = self.loss_kpt_aux(
                        kpt_preds_raw, kpt_aux_targets, kpt_weights)

        return losses

    @torch.no_grad()
    def _get_targets(
        self,
        priors: Tensor,
        batch_cls_scores: Tensor,
        batch_objectness: Tensor,
        batch_decoded_bboxes: Tensor,
        batch_decoded_kpts: Tensor,
        batch_kpt_vis: Tensor,
        batch_data_samples: SampleList,
    ):
        num_imgs = len(batch_data_samples)

        # use clip to avoid nan
        batch_cls_scores = batch_cls_scores.clip(min=-1e4, max=1e4).sigmoid()
        batch_objectness = batch_objectness.clip(min=-1e4, max=1e4).sigmoid()
        batch_kpt_vis = batch_kpt_vis.clip(min=-1e4, max=1e4).sigmoid()
        batch_cls_scores[torch.isnan(batch_cls_scores)] = 0
        batch_objectness[torch.isnan(batch_objectness)] = 0

        targets_each = []
        for i in range(num_imgs):
            target = self._get_targets_single(priors, batch_cls_scores[i],
                                              batch_objectness[i],
                                              batch_decoded_bboxes[i],
                                              batch_decoded_kpts[i],
                                              batch_kpt_vis[i],
                                              batch_data_samples[i])
            targets_each.append(target)

        targets = list(zip(*targets_each))
        for i, target in enumerate(targets):
            if torch.is_tensor(target[0]):
                target = tuple(filter(lambda x: x.size(0) > 0, target))
                if len(target) > 0:
                    targets[i] = torch.cat(target)

        foreground_masks, cls_targets, obj_targets, obj_weights, \
            bbox_targets, kpt_targets, vis_targets, vis_weights, pos_areas, \
            pos_priors, group_indices, num_pos_per_img = targets

        # post-processing for targets
        if self.use_aux_loss:
            bbox_cxcy = (bbox_targets[:, :2] + bbox_targets[:, 2:]) / 2.0
            bbox_wh = bbox_targets[:, 2:] - bbox_targets[:, :2]
            bbox_aux_targets = torch.cat([
                (bbox_cxcy - pos_priors[:, :2]) / pos_priors[:, 2:],
                torch.log(bbox_wh / pos_priors[:, 2:] + 1e-8)
            ],
                                         dim=-1)

            kpt_aux_targets = (kpt_targets - pos_priors[:, None, :2]) \
                / pos_priors[:, None, 2:]
        else:
            bbox_aux_targets, kpt_aux_targets = None, None

        return (foreground_masks, cls_targets, obj_targets, obj_weights,
                bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets,
                vis_targets, vis_weights, pos_areas, pos_priors, group_indices,
                num_pos_per_img)

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        objectness: Tensor,
        decoded_bboxes: Tensor,
        decoded_kpts: Tensor,
        kpt_vis: Tensor,
        data_sample: PoseDataSample,
    ) -> tuple:
        """Compute classification, bbox, keypoints and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_scores (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in xyxy format.
            decoded_kpts (Tensor): Decoded keypoints predictions of one image,
                a 3D-Tensor with shape [num_priors, num_keypoints, 2].
            kpt_vis (Tensor): Keypoints visibility predictions of one image,
                a 2D-Tensor with shape [num_priors, num_keypoints].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            data_sample (PoseDataSample): Data sample that contains the ground
                truth annotations for current image.

        Returns:
            tuple: A tuple containing various target tensors for training:
                - foreground_mask (Tensor): Binary mask indicating foreground
                    priors.
                - cls_target (Tensor): Classification targets.
                - obj_target (Tensor): Objectness targets.
                - obj_weight (Tensor): Weights for objectness targets.
                - bbox_target (Tensor): BBox targets.
                - kpt_target (Tensor): Keypoints targets.
                - vis_target (Tensor): Visibility targets for keypoints.
                - vis_weight (Tensor): Weights for keypoints visibility
                    targets.
                - pos_areas (Tensor): Areas of positive samples.
                - pos_priors (Tensor): Priors corresponding to positive
                    samples.
                - group_index (List[Tensor]): Indices of groups for positive
                    samples.
                - num_pos_per_img (int): Number of positive samples.
        """
        # TODO: change the shape of objectness to [num_priors]
        num_priors = priors.size(0)
        gt_instances = data_sample.gt_instance_labels
        gt_fields = data_sample.get('gt_fields', dict())
        num_gts = len(gt_instances)

        # No target
        if num_gts == 0:
            cls_target = cls_scores.new_zeros((0, self.num_classes))
            bbox_target = cls_scores.new_zeros((0, 4))
            obj_target = cls_scores.new_zeros((num_priors, 1))
            obj_weight = cls_scores.new_ones((num_priors, 1))
            kpt_target = cls_scores.new_zeros((0, self.num_keypoints, 2))
            vis_target = cls_scores.new_zeros((0, self.num_keypoints))
            vis_weight = cls_scores.new_zeros((0, self.num_keypoints))
            pos_areas = cls_scores.new_zeros((0, ))
            pos_priors = priors[:0]
            foreground_mask = cls_scores.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, obj_weight,
                    bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                    pos_priors, [], 0)

        # assign positive samples
        scores = cls_scores * objectness
        pred_instances = InstanceData(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=priors,
            keypoints=decoded_kpts,
            keypoints_visible=kpt_vis,
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        # sampling
        pos_inds = torch.nonzero(
            assign_result['gt_inds'] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result['labels'][pos_inds]
        pos_assigned_gt_inds = assign_result['gt_inds'][pos_inds] - 1

        # bbox target
        bbox_target = gt_instances.bboxes[pos_assigned_gt_inds.long()]

        # cls target
        max_overlaps = assign_result['max_overlaps'][pos_inds]
        cls_target = F.one_hot(pos_gt_labels,
                               self.num_classes) * max_overlaps.unsqueeze(-1)

        # pose targets
        kpt_target = gt_instances.keypoints[pos_assigned_gt_inds]
        vis_target = gt_instances.keypoints_visible[pos_assigned_gt_inds]
        if 'keypoints_visible_weights' in gt_instances:
            vis_weight = gt_instances.keypoints_visible_weights[
                pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)
        pos_areas = gt_instances.areas[pos_assigned_gt_inds]

        # obj target
        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1

        invalid_mask = gt_fields.get('heatmap_mask', None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            # ignore the tokens that predict the unlabled instances
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(
                dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(
                mask, grids, mode='bilinear', padding_mode='zeros')
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        # misc
        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        group_index = [
            torch.where(pos_assigned_gt_inds == num)[0]
            for num in torch.unique(pos_assigned_gt_inds)
        ]

        return (foreground_mask, cls_target, obj_target, obj_weight,
                bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                pos_priors, group_index, num_pos_per_img)

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-scale features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (1, h, w)
                    or (K+1, h, w) if keypoint heatmaps are predicted
                - displacements (Tensor): The predicted displacement fields
                    in shape (K*2, h, w)
        """

        cls_scores, objectnesses, bbox_preds, kpt_offsets, \
            kpt_vis = self.forward(feats)

        cfg = copy.deepcopy(test_cfg)

        batch_img_metas = [d.metainfo for d in batch_data_samples]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(), ),
                                    stride) for featmap_size, stride in zip(
                                        featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores).sigmoid()
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses).sigmoid()
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        flatten_bbox_preds = self.decode_bbox(flatten_bbox_preds,
                                              flatten_priors, flatten_stride)
        flatten_kpt_reg = self.decode_kpt_reg(flatten_kpt_offsets,
                                              flatten_priors, flatten_stride)

        results_list = []
        for (bboxes, scores, objectness, kpt_reg, kpt_vis,
             img_meta) in zip(flatten_bbox_preds, flatten_cls_scores,
                              flatten_objectness, flatten_kpt_reg,
                              flatten_kpt_vis, batch_img_metas):

            score_thr = cfg.get('score_thr', 0.01)
            scores *= objectness

            nms_pre = cfg.get('nms_pre', 100000)
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs_score, results = filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
            labels = results['labels']

            bboxes = bboxes[keep_idxs_score]
            kpt_vis = kpt_vis[keep_idxs_score]
            stride = flatten_stride[keep_idxs_score]
            keypoints = kpt_reg[keep_idxs_score]

            if bboxes.numel() > 0:
                nms_thr = cfg.get('nms_thr', 1.0)
                if nms_thr < 1.0:
                    keep_idxs_nms = nms_torch(bboxes, scores, nms_thr)
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    keypoints = keypoints[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes,
                bbox_scores=scores,
                keypoints=keypoints,
                keypoint_scores=kpt_vis,
                keypoints_visible=kpt_vis)

            input_size = img_meta['input_size']
            results.bboxes[:, 0::2].clamp_(0, input_size[0])
            results.bboxes[:, 1::2].clamp_(0, input_size[1])

            results_list.append(results.numpy())

        return results_list

    def decode_bbox(self, pred_bboxes: torch.Tensor, priors: torch.Tensor,
                    stride: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression results (delta_x, delta_y, log_w, log_h) to
        bounding boxes (tl_x, tl_y, br_x, br_y).

        Note:
            - batch size: B
            - token number: N

        Args:
            pred_bboxes (torch.Tensor): Encoded boxes with shape (B, N, 4),
                representing (delta_x, delta_y, log_w, log_h) for each box.
            priors (torch.Tensor): Anchors coordinates, with shape (N, 2).
            stride (torch.Tensor | int): Strides of the bboxes. It can be a
                single value if the same stride applies to all boxes, or it
                can be a tensor of shape (N, ) if different strides are used
                for each box.

        Returns:
            torch.Tensor: Decoded bounding boxes with shape (N, 4),
                representing (tl_x, tl_y, br_x, br_y) for each box.
        """
        stride = stride.view(1, stride.size(0), 1)
        priors = priors.view(1, priors.size(0), 2)

        xys = (pred_bboxes[..., :2] * stride) + priors
        whs = pred_bboxes[..., 2:].exp() * stride

        # Calculate bounding box corners
        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def decode_kpt_reg(self, pred_kpt_offsets: torch.Tensor,
                       priors: torch.Tensor,
                       stride: torch.Tensor) -> torch.Tensor:
        """Decode regression results (delta_x, delta_y) to keypoints
        coordinates (x, y).

        Args:
            pred_kpt_offsets (torch.Tensor): Encoded keypoints offsets with
                shape (batch_size, num_anchors, num_keypoints, 2).
            priors (torch.Tensor): Anchors coordinates with shape
                (num_anchors, 2).
            stride (torch.Tensor): Strides of the anchors.

        Returns:
            torch.Tensor: Decoded keypoints coordinates with shape
                (batch_size, num_boxes, num_keypoints, 2).
        """
        stride = stride.view(1, stride.size(0), 1, 1)
        priors = priors.view(1, priors.size(0), 1, 2)
        pred_kpt_offsets = pred_kpt_offsets.reshape(
            *pred_kpt_offsets.shape[:-1], self.num_keypoints, 2)

        decoded_kpts = pred_kpt_offsets * stride + priors
        return decoded_kpts

    def _flatten_predictions(self, preds: List[Tensor]):
        """Flattens the predictions from a list of tensors to a single
        tensor."""
        if len(preds) == 0:
            return None

        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)
