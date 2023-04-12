# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor


@MODELS.register_module()
class PETRHead(DeformableDETRHead):

    def __init__(self,
                 num_keypoints: int = 17,
                 num_pred_kpt_layer: int = 2,
                 loss_reg: dict = None,
                 loss_reg_aux: dict = None,
                 loss_oks: dict = None,
                 loss_oks_aux: dict = None,
                 loss_hm: dict = None,
                 *args,
                 **kwargs):
        self.num_keypoints = num_keypoints
        self.num_pred_kpt_layer = num_pred_kpt_layer
        super().__init__(*args, **kwargs)

        self._target_buffer = dict()

        self.loss_reg = MODELS.build(loss_reg)
        self.loss_reg_aux = MODELS.build(loss_reg_aux)
        self.loss_oks = MODELS.build(loss_oks)
        self.loss_oks_aux = MODELS.build(loss_oks_aux)
        self.loss_hm = MODELS.build(loss_hm)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [Linear(self.embed_dims, self.embed_dims * 2), nn.ReLU()]
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims * 2, self.num_keypoints * 2))
        reg_branch = nn.Sequential(*reg_branch)

        kpt_branch = []
        for _ in range(self.num_reg_fcs):
            kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(self.embed_dims, 2))
        kpt_branch = nn.Sequential(*kpt_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
            self.kpt_branches = nn.ModuleList(
                [kpt_branch for _ in range(self.num_pred_kpt_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
            self.kpt_branches = nn.ModuleList([
                copy.deepcopy(kpt_branch)
                for _ in range(self.num_pred_kpt_layer)
            ])

        self.heatmap_fc = Linear(self.embed_dims, self.num_keypoints)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == self.num_keypoints * 2:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                dec_outputs_coord: Tensor,
                det_labels: Tensor,
                det_scores: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        assert len(batch_img_metas) == 1, f'PETR only support test with ' \
            f'batch size 1, but got {len(batch_img_metas)}.'
        img_meta = batch_img_metas[0]

        if dec_outputs_coord.ndim == 4:
            dec_outputs_coord = dec_outputs_coord[-1]

        # filter instance
        if self.test_cfg.get('score_thr', .0) > 0:
            kept_inds = det_scores > self.test_cfg['score_thr']
            det_labels = det_labels[kept_inds]
            det_scores = det_scores[kept_inds]
            dec_outputs_coord = dec_outputs_coord[kept_inds]

        if len(dec_outputs_coord) > 0:
            # decode keypoints
            h, w = img_meta['img_shape']
            if rescale:
                h = h / img_meta['scale_factor'][0]
                w = w / img_meta['scale_factor'][1]
            keypoints = torch.stack([
                dec_outputs_coord[..., 0] * w,
                dec_outputs_coord[..., 1] * h,
            ],
                                    dim=2)
            keypoint_scores = torch.ones(keypoints.shape[:-1])

            # generate bounding boxes by outlining the detected poses
            bboxes = torch.stack([
                keypoints[..., 0].min(dim=1).values.clamp(0, w),
                keypoints[..., 1].min(dim=1).values.clamp(0, h),
                keypoints[..., 0].max(dim=1).values.clamp(0, w),
                keypoints[..., 1].max(dim=1).values.clamp(0, h),
            ],
                                 dim=1)
        else:
            keypoints = torch.empty(0, *dec_outputs_coord.shape[1:])
            keypoint_scores = torch.ones(keypoints.shape[:-1])
            bboxes = torch.empty(0, 4)

        results = InstanceData()
        results.set_metainfo(img_meta)
        results.bboxes = bboxes
        results.scores = det_scores
        results.bbox_scores = det_scores
        results.labels = det_labels
        results.keypoints = keypoints
        results.keypoint_scores = keypoint_scores
        results = results.numpy()

        return [results]

    def loss(self, enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             all_layers_classes: Tensor, all_layers_coords: Tensor,
             hm_memory: Tensor, hm_mask: Tensor, dec_outputs_coord: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # enc_outputs_class:  [bs, mlv_shape, 1]
        # enc_outputs_coord:  [bs, mlv_shape, 2*num_keypoints]
        # all_layers_classes  [3, bs, num_queries, 1]
        # all_layers_coords   [3, bs, num_queries, 2*num_keypoints]
        # hm_memory:          [bs, lv0_h, lv0_w, 256]
        # hm_mask:            [bs, lv0_h, lv0_w]
        # dec_outputs_coord:  [2, max_inst, num_keypoints, 2]

        batch_gt_instances = []
        batch_img_metas = []
        batch_gt_fields = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_fields.append(data_sample.gt_fields)

        loss_dict = dict()

        # calculate loss for decoder output
        losses_cls, losses_kpt, losses_oks = multi_apply(
            self.loss_by_feat_single,
            all_layers_classes,
            all_layers_coords,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            cache_targets=True)
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        loss_dict['loss_oks'] = losses_oks[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_kpt_i, loss_oks_i in zip(losses_cls[:-1],
                                                      losses_kpt[:-1],
                                                      losses_oks[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            num_dec_layer += 1

        # calculate loss for encoder output
        losses_cls, losses_kpt, losses_oks = self.loss_by_feat_single(
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            cache_targets=False,
            compute_oks_loss=False)
        loss_dict['enc_loss_cls'] = losses_cls
        loss_dict['enc_loss_kpt'] = losses_kpt

        # calculate heatmap loss
        loss_hm = self.loss_heatmap(
            hm_memory, hm_mask, batch_gt_fields=batch_gt_fields)
        loss_dict['loss_hm'] = loss_hm

        # calculate loss for kpt_decoder output
        losses_kpt, losses_oks = multi_apply(
            self.loss_refined_kpts,
            dec_outputs_coord,
            batch_img_metas=batch_img_metas,
        )

        num_dec_layer = 0
        for loss_kpt_i, loss_oks_i in zip(losses_kpt, losses_oks):
            loss_dict[f'd{num_dec_layer}.loss_kpt_refine'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_oks_refine'] = loss_oks_i
            num_dec_layer += 1

        self._target_buffer.clear()
        return loss_dict

    # TODO: rename this method
    def loss_by_feat_single(self,
                            cls_scores: Tensor,
                            kpt_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict],
                            cache_targets: bool = False,
                            compute_oks_loss: bool = True) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        # cls_scores    [bs, num_queries, 1]
        # kpt_preds     [bs, num_queries, 2*num_keypoitns]

        num_imgs, num_queries = cls_scores.shape[:2]

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [
            kpt_preds[i].reshape(-1, self.num_keypoints, 2)
            for i in range(num_imgs)
        ]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, kpt_targets_list,
         kpt_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)  # [bs*300]
        label_weights = torch.cat(label_weights_list, 0)  # [bs*300] (all 1)
        bbox_targets = torch.cat(bbox_targets_list,
                                 0)  # [bs*300, 4] (normalized)
        kpt_targets = torch.cat(kpt_targets_list,
                                0)  # [bs*300, 17, 2] (normalized)
        kpt_weights = torch.cat(kpt_weights_list, 0)  # [bs*300, 17]

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, self.num_keypoints, 2)
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_reg_aux(
            kpt_preds,
            kpt_targets,
            kpt_weights.unsqueeze(-1),
            avg_factor=num_valid_kpt)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        pos_mask = kpt_weights.sum(-1) > 0
        if compute_oks_loss and pos_mask.any().item():
            pos_inds = (pos_mask.nonzero()).div(
                num_queries, rounding_mode='trunc').squeeze(-1)

            # construct factors used for rescale keypoints
            factors = []
            for img_meta, kpt_pred in zip(batch_img_metas, kpt_preds):
                img_h, img_w, = img_meta['img_shape']
                factor = kpt_pred.new_tensor([img_w, img_h]).reshape(1, 1, 2)
                factors.append(factor)
            factors = torch.cat(factors, 0)
            factors = factors[pos_inds]

            # keypoint oks loss
            pos_kpt_preds = kpt_preds[pos_mask] * factors
            pos_kpt_targets = kpt_targets[pos_mask] * factors
            pos_kpt_weights = kpt_weights[pos_mask]
            pos_bbox_targets = (bbox_targets[pos_mask].reshape(-1, 2, 2) *
                                factor).reshape(-1, 4)

            loss_oks = self.loss_oks_aux(pos_kpt_preds, pos_kpt_targets,
                                         pos_kpt_weights, pos_bbox_targets)
        else:
            loss_oks = torch.zeros_like(loss_kpt)

        return loss_cls, loss_kpt, loss_oks

    def loss_heatmap(self, hm_memory, hm_mask, batch_gt_fields):

        # compute heatmap predition
        pred_heatmaps = self.heatmap_fc(hm_memory)
        pred_heatmaps = torch.clamp(
            pred_heatmaps.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pred_heatmaps = pred_heatmaps.permute(0, 3, 1, 2).contiguous()

        # construct heatmap target
        gt_heatmaps = torch.zeros_like(pred_heatmaps)
        for i, gf in enumerate(batch_gt_fields):
            gt_heatmap = gf.gt_heatmaps
            h = min(gt_heatmap.size(1), gt_heatmaps.size(2))
            w = min(gt_heatmap.size(2), gt_heatmaps.size(3))
            gt_heatmaps[i, :, :h, :w] = gt_heatmap[:, :h, :w]

        loss_hm = self.loss_hm(pred_heatmaps, gt_heatmaps, None,
                               1 - hm_mask.unsqueeze(1).float())
        return loss_hm

    def loss_refined_kpts(self, kpt_preds: Tensor,
                          batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        # kpt_preds     [num_selected, num_keypoints, 2]
        bbox_targets_list = self._target_buffer['bbox_targets_list']
        kpt_targets_list = self._target_buffer['kpt_targets_list']
        kpt_weights_list = self._target_buffer['kpt_weights_list']
        num_queries = len(kpt_targets_list[0])
        bbox_targets = torch.cat(bbox_targets_list,
                                 0).contiguous()  # [bs*300, 4] (normalized)
        kpt_targets = torch.cat(kpt_targets_list,
                                0).contiguous()  # [bs*300, 17, 2] (normalized)
        kpt_weights = torch.cat(kpt_weights_list,
                                0).contiguous()  # [bs*300, 17]

        pos_mask = (kpt_weights.sum(-1) > 0).contiguous()
        pos_inds = (pos_mask.nonzero()).div(
            num_queries, rounding_mode='trunc').squeeze(-1)

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, self.num_keypoints, 2)
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_reg(
            kpt_preds,
            kpt_targets[pos_mask],
            kpt_weights[pos_mask].unsqueeze(-1),
            avg_factor=num_valid_kpt)

        if pos_mask.any().item():
            # construct factors used for rescale keypoints
            factors = []
            for img_meta in batch_img_metas:
                img_h, img_w, = img_meta['img_shape']
                factor = kpt_preds.new_tensor([img_w, img_h]).reshape(1, 1, 2)
                factors.append(factor)
            factors = torch.cat(factors, 0)

            factors = factors[pos_inds]

            # keypoint oks loss
            pos_kpt_preds = kpt_preds * factors
            pos_kpt_targets = kpt_targets[pos_mask]

            pos_kpt_targets = pos_kpt_targets * factors
            pos_kpt_weights = kpt_weights[pos_mask]
            pos_bbox_targets = (bbox_targets[pos_mask].reshape(-1, 2, 2) *
                                factor).reshape(-1, 4)

            loss_oks = self.loss_oks_aux(pos_kpt_preds, pos_kpt_targets,
                                         pos_kpt_weights, pos_bbox_targets)
        else:
            loss_oks = torch.zeros_like(loss_kpt)

        return loss_kpt, loss_oks

    @torch.no_grad()
    def get_targets(self,
                    cls_scores_list: List[Tensor],
                    kpt_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    cache_targets: bool = False) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, kpt_targets_list,
         kpt_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, kpt_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        if cache_targets:
            self._target_buffer['labels_list'] = labels_list
            self._target_buffer['label_weights_list'] = label_weights_list
            self._target_buffer['bbox_targets_list'] = bbox_targets_list
            self._target_buffer['kpt_targets_list'] = kpt_targets_list
            self._target_buffer['kpt_weights_list'] = kpt_weights_list
            self._target_buffer['num_total_pos'] = num_total_pos
            self._target_buffer['num_total_neg'] = num_total_neg

        return (labels_list, label_weights_list, bbox_targets_list,
                kpt_targets_list, kpt_weights_list, num_total_pos,
                num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, kpt_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        num_insts = kpt_pred.size(0)
        factor = kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0).unsqueeze(1)
        kpt_pred = kpt_pred * factor

        pred_instances = InstanceData(scores=cls_score, keypoints=kpt_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_labels = gt_instances.labels
        gt_bboxes = gt_instances.bboxes
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # label targets
        labels = gt_labels.new_full((num_insts, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_insts)

        # kpt targets
        kpt_targets = torch.zeros_like(kpt_pred)
        pos_gt_keypoints = gt_keypoints[pos_assigned_gt_inds.long(), :]
        kpt_targets[pos_inds] = pos_gt_keypoints / factor
        kpt_weights = torch.zeros_like(kpt_pred).narrow(-1, 0, 1).squeeze(-1)
        pos_gt_keypoints_visible = gt_keypoints_visible[
            pos_assigned_gt_inds.long()]
        kpt_weights[pos_inds] = (pos_gt_keypoints_visible > 0).float()

        # bbox_targets, which is used to compute oks loss
        bbox_targets = torch.zeros_like(kpt_pred).narrow(-2, 0, 2)
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long()]
        bbox_targets[pos_inds] = pos_gt_bboxes.reshape(
            *pos_gt_bboxes.shape[:-1], 2, 2) / factor
        bbox_targets = bbox_targets.flatten(-2)

        return (labels, label_weights, bbox_targets, kpt_targets, kpt_weights,
                pos_inds, neg_inds)
