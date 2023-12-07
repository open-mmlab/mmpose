# Copyright (c) OpenMMLab. All rights reserved.
import copy
import types
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmdet.utils import ConfigType, reduce_mean
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.evaluation.functional import nms_torch
from mmpose.models.utils import (GAUEncoder, SinePositionalEncoding,
                                 filter_scores_and_topk)
from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_xyxy2cs
from mmpose.utils.typing import Features, OptSampleList, Predictions
from .yoloxpose_head import YOLOXPoseHead

EPS = 1e-8


class RTMOHeadModule(BaseModule):
    """RTMO head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_keypoints (int): Number of keypoints defined for one instance.
         in_channels (int): Number of channels in the input feature maps.
        cls_feat_channels (int): Number of channels in the classification score
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
        in_channels: int,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        cls_feat_channels: int = 256,
        stacked_convs: int = 2,
        num_groups=8,
        channels_per_group=36,
        pose_vec_channels=-1,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: Optional[ConfigType] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_feat_channels = int(cls_feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        self.in_channels = int(in_channels * widen_factor)
        self.num_keypoints = num_keypoints

        self.num_groups = num_groups
        self.channels_per_group = int(widen_factor * channels_per_group)
        self.pose_vec_channels = pose_vec_channels

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_reg_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.cls_feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.cls_feat_channels,
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
        for _ in self.featmap_strides:
            self.out_cls.append(
                nn.Conv2d(self.cls_feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_pose = nn.ModuleList()
        out_chn = self.num_groups * self.channels_per_group
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                chn = self.in_channels if i == 0 else out_chn
                groups = 1 if i == 0 else self.num_groups
                stacked_convs.append(
                    ConvModule(
                        chn,
                        out_chn,
                        3,
                        stride=1,
                        padding=1,
                        groups=groups,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_kpt_reg = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(out_chn, 4, 1))
            self.out_kpt_reg.append(
                nn.Conv2d(out_chn, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(out_chn, self.num_keypoints, 1))

        if self.pose_vec_channels > 0:
            self.out_pose = nn.ModuleList()
            for _ in self.featmap_strides:
                self.out_pose.append(
                    nn.Conv2d(out_chn, self.pose_vec_channels, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.out_cls:
            conv_cls.bias.data.fill_(bias_init)

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

        cls_scores, bbox_preds = [], []
        kpt_offsets, kpt_vis = [], []
        pose_feats = []

        for i in range(len(x)):

            cls_feat, reg_feat = x[i].split(x[i].size(1) // 2, 1)

            cls_feat = self.conv_cls[i](cls_feat)
            reg_feat = self.conv_pose[i](reg_feat)

            cls_scores.append(self.out_cls[i](cls_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            if self.training:
                kpt_offsets.append(self.out_kpt_reg[i](reg_feat))
            kpt_vis.append(self.out_kpt_vis[i](reg_feat))

            if self.pose_vec_channels > 0:
                pose_feats.append(self.out_pose[i](reg_feat))
            else:
                pose_feats.append(reg_feat)

        return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats


class DCC(BaseModule):
    """Dynamic Coordinate Classifier for One-stage Pose Estimation."""

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        feat_channels: int,
        num_bins: Tuple[int, int],
        spe_cfg: Optional[dict] = dict(out_channels=128, temperature=300),
        gau_cfg: Optional[dict] = dict(
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc='add'),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.feat_channels = feat_channels
        self.num_bins = num_bins
        self.gau_cfg = gau_cfg

        self.spe = SinePositionalEncoding(**spe_cfg)
        self.spe_feat_channels = spe_cfg['out_channels']

        self._build_layers()
        self._build_basic_bins()

    def _build_layers(self):

        # GAU encoder
        if self.gau_cfg is not None:
            gau_cfg = self.gau_cfg.copy()
            gau_cfg['in_token_dims'] = self.feat_channels
            gau_cfg['out_token_dims'] = self.feat_channels
            self.gau = GAUEncoder(**gau_cfg)
            if gau_cfg.get('pos_enc', 'none') in ('add', 'rope'):
                self.pos_enc = nn.Parameter(
                    torch.randn(self.num_keypoints, gau_cfg['s']))

        # fully-connected layers to convert pose feats to keypoint feats
        pose_vec_fc = [
            nn.Linear(self.in_channels,
                      self.feat_channels * self.num_keypoints),
            nn.BatchNorm1d(self.feat_channels * self.num_keypoints)
        ]
        self.pose_vec_fc = nn.Sequential(*pose_vec_fc)

        # adapter fully-connected layers for dynamic encodings
        self.x_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)
        self.y_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)

        # fully-connected layers to predict sigma
        self.sigma_fc = nn.Sequential(
            nn.Linear(self.in_channels, self.num_keypoints), nn.Sigmoid(),
            Scale(0.1))

    def _build_basic_bins(self):
        # build basic grid coordinates for x, y
        self.register_buffer('y_bins',
                             torch.linspace(-0.5, 0.5, self.num_bins[1]))
        self.register_buffer('x_bins',
                             torch.linspace(-0.5, 0.5, self.num_bins[0]))

    def _apply_softmax(self, x_hms, y_hms):
        """Apply softmax on 1d heatmaps."""
        x_hms = x_hms.clamp(min=-5e4, max=5e4)
        y_hms = y_hms.clamp(min=-5e4, max=5e4)
        pred_x = x_hms - x_hms.max(dim=-1, keepdims=True).values.detach()
        pred_y = y_hms - y_hms.max(dim=-1, keepdims=True).values.detach()

        exp_x, exp_y = pred_x.exp(), pred_y.exp()
        prob_x = exp_x / (exp_x.sum(dim=-1, keepdims=True) + EPS)
        prob_y = exp_y / (exp_y.sum(dim=-1, keepdims=True) + EPS)

        return prob_x, prob_y

    def _calculate_spe(self, bbox_cs, grids, stride):
        center, scale = bbox_cs.split(2, dim=-1)
        center = center - grids

        x_bins, y_bins = self.x_bins, self.y_bins

        x_bins = x_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 0:1] + center[..., 0:1]  # [B, ?opt, 1]
        y_bins = y_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 1:2] + center[..., 1:2]

        spe_x = self.x_fc(self.spe(
            position=x_bins))  # [B, ?opt, bins, feats_channels]
        spe_y = self.y_fc(self.spe(
            position=y_bins))  # [B, ?opt, bins, feats_channels]

        return spe_x, spe_y

    def _pose_vec_to_heatmaps(self, pose_vecs, spe_x, spe_y):

        pose_feats = self.pose_vec_fc(pose_vecs)

        pose_feats = pose_feats.reshape(*pose_feats.shape[:-1],
                                        self.num_keypoints, self.feat_channels)

        pose_feats = self.gau(
            pose_feats, pos_enc=getattr(self, 'pos_enc', None))

        x_hms = torch.matmul(pose_feats, spe_x.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(pose_feats, spe_y.transpose(-1, -2).contiguous())

        return x_hms, y_hms

    def _decode_xy_heatmaps(self, x_hms, y_hms, bbox_cs, ipr_decode=False):

        if ipr_decode:
            x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
            center, scale = bbox_cs.split(2, dim=-1)
            # center = center

            x_bins, y_bins = self.x_bins, self.y_bins

            x_bins = x_bins.view(*((1,) * (scale.ndim-1)), -1) \
                * scale[..., 0:1] + center[..., 0:1]  # [B, ?opt, 1]
            y_bins = y_bins.view(*((1,) * (scale.ndim-1)), -1) \
                * scale[..., 1:2] + center[..., 1:2]

            x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
            y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)

        else:

            x = x_hms.argmax(dim=-1).float() / (x_hms.size(-1) - 1
                                                )  # [n, num_keypoints]
            y = y_hms.argmax(dim=-1).float() / (y_hms.size(-1) - 1
                                                )  # [n, num_keypoints]

            x = bbox_cs.narrow(-1, 0, 1) + bbox_cs.narrow(-1, 2, 1) * (x - 0.5)
            y = bbox_cs.narrow(-1, 1, 1) + bbox_cs.narrow(-1, 3, 1) * (y - 0.5)

        return torch.stack((x, y), dim=-1)

    def generate_target_heatmap(self, kpt_targets, bbox_cs, sigmas, areas):
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim-1)), -1) \
            * scale[..., 1:2] + center[..., 1:2]

        dist_x = torch.abs(kpt_targets.narrow(2, 0, 1) -
                           x_bins.unsqueeze(1))  # [n, num_keypoints, bins]
        dist_y = torch.abs(kpt_targets.narrow(2, 1, 1) -
                           y_bins.unsqueeze(1))  # [n, num_keypoints, bins]

        # normalize
        areas = areas.pow(0.5).clip(min=1).reshape(-1, 1, 1)
        sigmas = sigmas.clip(min=1e-3).unsqueeze(2)
        dist_x = dist_x / areas / sigmas
        dist_y = dist_y / areas / sigmas

        hm_x = torch.exp(-dist_x / 2) / sigmas
        hm_y = torch.exp(-dist_y / 2) / sigmas

        return hm_x, hm_y

    def forward_train(self, pose_vecs, bbox_cs, grids, stride):
        spe_x, spe_y = self._calculate_spe(bbox_cs, grids, stride)
        x_hms, y_hms = self._pose_vec_to_heatmaps(pose_vecs, spe_x, spe_y)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        pred_hm_x, pred_hm_y = self._apply_softmax(x_hms, y_hms)
        sigmas = self.sigma_fc(pose_vecs)
        return pose_preds, (pred_hm_x, pred_hm_y), sigmas

    @torch.no_grad()
    def forward_test(self,
                     pose_vecs,
                     bbox_cs,
                     grids,
                     stride=None,
                     ipr_decode=False):
        spe_x, spe_y = self._calculate_spe(bbox_cs, grids, stride)
        x_hms, y_hms = self._pose_vec_to_heatmaps(pose_vecs, spe_x, spe_y)
        pose_preds = self._decode_xy_heatmaps(
            x_hms, y_hms, bbox_cs, ipr_decode=ipr_decode)

        return pose_preds

    def switch_to_deploy(self):
        if getattr(self, 'deploy', False):
            return

        # 1. pose_vec_fc
        fc, bn = self.pose_vec_fc
        self._eps = bn.eps
        std = (bn.running_var + bn.eps).sqrt()
        weight = fc.weight * (bn.weight / std).unsqueeze(1)
        bias = bn.bias + (fc.bias - bn.running_mean) * bn.weight / std
        fc.weight.data = weight.detach()
        fc.bias.data = bias.detach()

        self.pose_vec_fc = fc

        # 2. gau
        with torch.no_grad():
            pos_enc = self.pos_enc.reshape(1, 1, *self.pos_enc.shape).detach()
            gamma_q = self.gau.gamma[0].view(1, 1, 1, self.gau.gamma.size(-1))
            gamma_k = self.gau.gamma[1].view(1, 1, 1, self.gau.gamma.size(-1))
            beta_q = self.gau.beta[0].view(1, 1, 1,
                                           self.gau.beta.size(-1)) + pos_enc
            beta_k = self.gau.beta[1].view(1, 1, 1,
                                           self.gau.beta.size(-1)) + pos_enc
            gamma_q = gamma_q.detach().cpu()
            gamma_k = gamma_k.detach().cpu()
            beta_q = beta_q.detach().cpu()
            beta_k = beta_k.detach().cpu()

        @torch.no_grad()
        def _forward(self, x, *args, **kwargs):
            norm = torch.norm(x, dim=-1, keepdim=True) * self.ln.scale
            x = x / norm.clamp(min=self.ln.eps) * self.ln.g

            uv = self.uv(x)
            uv = self.act_fn(uv)

            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)

            # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
            # dim = base.ndim - self.gamma.ndim + 1
            # gamma = self.gamma.view(*((1,) * dim), *self.gamma.size())
            # beta = self.beta.view(*((1,) * dim), *self.beta.size())
            # base = base.unsqueeze(-2) * gamma + beta
            # print(base.shape, base.device)

            q = base * gamma_q + beta_q
            k = base * gamma_k + beta_k

            # [B, K, 2, s] -> [B, K, s], [B, K, s]
            # q, k = torch.unbind(base, dim=-2)
            qk = torch.matmul(q, k.transpose(-1, -2))

            kernel = torch.square(torch.nn.functional.relu(qk / self.sqrt_s))
            x = u * torch.matmul(kernel, v)
            x = self.o(x)
            return x

        self.gau._forward = types.MethodType(_forward, self.gau)

        # 3. forward_test

        x_bins_ = self.x_bins.view(1, 1, -1).detach().cpu()
        y_bins_ = self.y_bins.view(1, 1, -1).detach().cpu()
        dim_t = self.spe.dim_t.view(1, 1, 1, -1).detach().cpu()
        # freq = position.unsqueeze(-1) / dim_t
        # pos_enc = torch.cat((freq.cos(), freq.sin()), dim=-1)

        @torch.no_grad()
        def _forward_test(self, pose_vecs, bbox_cs, grids, ipr_decode=False):
            center, scale = bbox_cs.split(2, dim=-1)
            center = center - grids

            # x_bins, y_bins = self.x_bins, self.y_bins

            x_bins = x_bins_ * scale[..., 0:1] + center[..., 0:1]
            y_bins = y_bins_ * scale[..., 1:2] + center[..., 1:2]

            freq_x = x_bins.unsqueeze(-1) / dim_t
            spe_x = torch.cat((freq_x.cos(), freq_x.sin()), dim=-1)
            freq_y = y_bins.unsqueeze(-1) / dim_t
            spe_y = torch.cat((freq_y.cos(), freq_y.sin()), dim=-1)

            spe_x = self.x_fc(spe_x)  # [B, ?opt, bins, feats_channels]
            spe_y = self.y_fc(spe_y)  # [B, ?opt, bins, feats_channels]

            pose_feats = self.pose_vec_fc(pose_vecs)
            pose_feats = pose_feats.reshape(*pose_feats.shape[:-1],
                                            self.num_keypoints,
                                            self.feat_channels)

            pose_feats = self.gau(pose_feats)

            x_hms = torch.matmul(pose_feats,
                                 spe_x.transpose(-1, -2).contiguous())
            y_hms = torch.matmul(pose_feats,
                                 spe_y.transpose(-1, -2).contiguous())

            if ipr_decode:
                x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
                center, scale = bbox_cs.split(2, dim=-1)
                # center = center

                x_bins, y_bins = self.x_bins, self.y_bins

                x_bins = x_bins.view(*((1,) * (scale.ndim-1)), -1) \
                    * scale[..., 0:1] + center[..., 0:1]  # [B, ?opt, 1]
                y_bins = y_bins.view(*((1,) * (scale.ndim-1)), -1) \
                    * scale[..., 1:2] + center[..., 1:2]

                x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
                y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)

            else:

                x = x_hms.argmax(dim=-1).float() / (x_hms.size(-1) - 1)
                y = y_hms.argmax(dim=-1).float() / (y_hms.size(-1) - 1)

                x = bbox_cs.narrow(-1, 0, 1) + bbox_cs.narrow(-1, 2, 1) * (
                    x - 0.5)
                y = bbox_cs.narrow(-1, 1, 1) + bbox_cs.narrow(-1, 3, 1) * (
                    y - 0.5)

            return torch.stack((x, y), dim=-1)

        self.forward_test = types.MethodType(_forward_test, self)

        self.deploy = True


@MODELS.register_module()
class RTMOHead(YOLOXPoseHead):

    def __init__(
        self,
        num_keypoints: int,
        head_module_cfg: ConfigType,
        featmap_strides: Sequence[int] = [8, 16, 32],
        num_classes: int = 1,
        use_aux_loss: bool = False,
        use_kpt_reg: bool = False,
        assigner: ConfigType = None,
        prior_generator: ConfigType = None,
        dcc_cfg: Optional[ConfigType] = None,
        loss_cls: Optional[ConfigType] = None,
        loss_obj: Optional[ConfigType] = None,
        loss_bbox: Optional[ConfigType] = None,
        loss_oks: Optional[ConfigType] = None,
        loss_vis: Optional[ConfigType] = None,
        loss_mle: Optional[ConfigType] = None,
        loss_bbox_aux: Optional[ConfigType] = None,
        loss_kpt_aux: Optional[ConfigType] = None,
        overlaps_power: float = 1.0,
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            head_module_cfg=None,
            featmap_strides=featmap_strides,
            num_classes=num_classes,
            use_aux_loss=use_aux_loss,
            assigner=assigner,
            prior_generator=prior_generator,
            loss_cls=loss_cls,
            loss_obj=loss_obj,
            loss_bbox=loss_bbox,
            loss_oks=loss_oks,
            loss_vis=loss_vis,
            loss_bbox_aux=loss_bbox_aux,
            loss_kpt_aux=loss_kpt_aux,
            overlaps_power=overlaps_power)

        # override to ensure consistency
        head_module_cfg['featmap_strides'] = featmap_strides
        head_module_cfg['num_keypoints'] = num_keypoints

        # build modules
        self.head_module = RTMOHeadModule(**head_module_cfg)

        self.use_kpt_reg = use_kpt_reg
        if dcc_cfg is not None:
            dcc_cfg['num_keypoints'] = num_keypoints
            self.dcc = DCC(**dcc_cfg)

        # build losses
        if loss_mle is not None:
            self.loss_mle = MODELS.build(loss_mle)

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
        cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_vecs = self.forward(
            feats)

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
        flatten_objectness = torch.ones_like(
            flatten_cls_scores).detach().narrow(-1, 0, 1) * 1e4
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
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
        extra_info = dict(num_samples=num_total_samples)
        losses = dict()
        cls_preds_all = flatten_cls_scores.view(-1, self.num_classes)

        if num_pos > 0:
            # cls_preds = cls_preds_all[pos_masks]

            # 3.2 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses['loss_bbox'] = self.loss_bbox(
                bbox_preds, bbox_targets) / num_total_samples

            # 3.3 keypoint loss
            kpt_reg_preds = flatten_kpt_decoded.view(-1, self.num_keypoints,
                                                     2)[pos_masks]

            if hasattr(self, 'loss_mle') and self.loss_mle.loss_weight > 0:
                pose_vecs = flatten_pose_vecs.view(
                    -1, flatten_pose_vecs.size(-1))[pos_masks]
                bbox_cs = torch.cat(bbox_xyxy2cs(bbox_preds, 1.25), dim=1)
                # 'cc' refers to 'cordinate classification'
                kpt_cc_preds, pred_hms, sigmas = \
                    self.dcc.forward_train(pose_vecs,
                                           bbox_cs,
                                           pos_priors[..., :2],
                                           pos_priors[..., -1])
                target_hms = self.dcc.generate_target_heatmap(
                    kpt_targets,
                    bbox_cs,
                    sigmas,
                    pos_areas,
                    grids=pos_priors[..., :2],
                    stride=pos_priors[..., -1])
                losses['loss_mle'] = self.loss_mle(pred_hms, target_hms,
                                                   vis_targets)

            if self.use_kpt_reg:
                losses['loss_oks'] = self.loss_oks(kpt_reg_preds, kpt_targets,
                                                   vis_targets, pos_areas)
            else:
                with torch.no_grad():
                    diff_cc = torch.norm(kpt_cc_preds - kpt_targets, dim=-1)
                    diff_reg = torch.norm(kpt_reg_preds - kpt_targets, dim=-1)
                    mask = (diff_reg > diff_cc).float()
                    kpt_weights_reg = vis_targets * mask
                    # kpt_weights_cc = vis_targets * (2 - mask)
                    oks = self.assigner.oks_calculator(kpt_cc_preds,
                                                       kpt_targets,
                                                       vis_targets, pos_areas)

                losses['loss_oks'] = self.loss_oks(kpt_reg_preds,
                                                   kpt_cc_preds.detach(),
                                                   kpt_weights_reg, pos_areas)
                cls_targets = oks.unsqueeze(1).detach()

            # 3.4 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1,
                                                 self.num_keypoints)[pos_masks]
            losses['loss_vis'] = self.loss_vis(kpt_vis_preds, vis_targets,
                                               vis_weights)

            # 3.5 classification loss
            extra_info['overlaps'] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            obj_targets[pos_masks] = cls_targets.to(obj_targets)

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
                    kpt_weights = vis_targets
                    losses['loss_kpt_aux'] = self.loss_kpt_aux(
                        kpt_preds_raw, kpt_aux_targets,
                        kpt_weights) / num_total_samples

        losses['loss_cls'] = self.loss_cls(cls_preds_all, obj_targets,
                                           obj_weights) / num_total_samples
        losses.update(extra_info)

        return losses

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

        cls_scores, bbox_preds, _, kpt_vis, pose_vecs = self.forward(feats)

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
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
        if flatten_pose_vecs is None:
            flatten_pose_vecs = [None] * len(batch_img_metas)
        flatten_bbox_preds = self.decode_bbox(flatten_bbox_preds,
                                              flatten_priors, flatten_stride)

        results_list = []
        for (bboxes, scores, kpt_vis, pose_vecs,
             img_meta) in zip(flatten_bbox_preds, flatten_cls_scores,
                              flatten_kpt_vis, flatten_pose_vecs,
                              batch_img_metas):

            score_thr = cfg.get('score_thr', 0.01)

            nms_pre = cfg.get('nms_pre', 100000)
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs_score, results = filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
            labels = results['labels']

            bboxes = bboxes[keep_idxs_score]
            kpt_vis = kpt_vis[keep_idxs_score]
            grids = flatten_priors[keep_idxs_score]
            stride = flatten_stride[keep_idxs_score]

            if bboxes.numel() > 0:
                nms_thr = cfg.get('nms_thr', 1.0)
                if nms_thr < 1.0:

                    keep_idxs_nms = nms_torch(bboxes, scores, nms_thr)
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]

                pose_vecs = pose_vecs[keep_idxs_score][keep_idxs_nms]
                bbox_cs = torch.cat(bbox_xyxy2cs(bboxes, 1.25), dim=1)
                grids = grids[keep_idxs_nms]
                ipr_decode = cfg.get('ipr_decode', False)
                keypoints = self.dcc.forward_test(
                    pose_vecs, bbox_cs, grids, stride, ipr_decode=ipr_decode)

            else:
                # empty prediction
                keypoints = bboxes.new_zeros((0, self.num_keypoints, 2))

            results = InstanceData(
                scores=scores,
                labels=labels + 1,
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

    def switch_to_deploy(self):
        if getattr(self, 'deploy', False):
            return

        self.deploy = True

        # grid generator
        cls_scores = [torch.rand(1, 1, 40, 40), torch.rand(1, 1, 20, 20)]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device='cpu')
        self.flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            self.flatten_priors.new_full((featmap_size.numel(), ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        self.flatten_stride = torch.cat(mlvl_strides)
