# ----------------------------------------------------------------------------
# Adapted from https://github.com/IDEA-Research/ED-Pose/ \
#              tree/master/models/edpose
# Original licence: IDEA License 1.0
# ----------------------------------------------------------------------------

import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.models.utils import inverse_sigmoid
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from .base_transformer_head import TransformerHead
from .transformers.deformable_detr_layers import (
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoder)
from .transformers.utils import FFN, PositionEmbeddingSineHW


class EDPoseDecoder(BaseModule):
    """Transformer decoder of EDPose: `Explicit Box Detection Unifies End-to-
    End Multi-Person Pose Estimation.

    Args:
        layer_cfg (ConfigDict): the config of each encoder
            layer. All the layers will share the same config.
        num_layers (int): Number of decoder layers.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`.
        embed_dims (int): Dims of embed.
        query_dim (int): Dims of queries.
        num_feature_levels (int): Number of feature levels.
        num_box_decoder_layers (int): Number of box decoder layers.
        num_keypoints (int): Number of datasets' body keypoints.
        num_dn (int): Number of denosing points.
        num_group (int): Number of decoder layers.
    """

    def __init__(self,
                 layer_cfg,
                 num_layers,
                 return_intermediate,
                 embed_dims: int = 256,
                 query_dim=4,
                 num_feature_levels=1,
                 num_box_decoder_layers=2,
                 num_keypoints=17,
                 num_dn=100,
                 num_group=100):
        super().__init__()

        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.embed_dims = embed_dims

        assert return_intermediate, 'support return_intermediate only'
        self.return_intermediate = return_intermediate

        assert query_dim in [
            2, 4
        ], 'query_dim should be 2/4 but {}'.format(query_dim)
        self.query_dim = query_dim

        self.num_feature_levels = num_feature_levels

        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.embed_dims)

        self.ref_point_head = FFN(self.query_dim // 2 * self.embed_dims,
                                  self.embed_dims, self.embed_dims, 2)

        self.num_keypoints = num_keypoints
        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.pose_embed = None
        self.pose_hw_embed = None
        self.num_box_decoder_layers = num_box_decoder_layers
        self.box_pred_damping = None
        self.num_group = num_group
        self.rm_detach = None
        self.num_dn = num_dn
        self.hw = nn.Embedding(self.num_keypoints, 2)
        self.keypoint_embed = nn.Embedding(self.num_keypoints, embed_dims)
        self.kpt_index = [
            x for x in range(self.num_group * (self.num_keypoints + 1))
            if x % (self.num_keypoints + 1) != 0
        ]

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                reference_points: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                humandet_attn_mask: Tensor, human2pose_attn_mask: Tensor,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of decoder
        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results.

        Returns:
            Tuple[Tuple[Tensor]]: Outputs of Deformable Transformer Decoder.

            - output (Tuple[Tensor]): Output embeddings of the last decoder,
              each has shape (num_decoder_layers, num_queries, bs, embed_dims)
            - reference_points (Tensor): The reference of the last decoder
              layer, each has shape (num_decoder_layers, bs, num_queries, 4).
              The coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        attn_mask = humandet_attn_mask
        intermediate = []
        intermediate_reference_points = [reference_points]
        effect_num_dn = self.num_dn if self.training else 0
        inter_select_number = self.num_group
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[None, :]

            query_sine_embed = self.get_proposal_pos_embed(
                reference_points_input[:, :, 0, :])  # nq, bs, 256*2
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

            output = layer(
                output.transpose(0, 1),
                query_pos=query_pos.transpose(0, 1),
                value=value.transpose(0, 1),
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input.transpose(
                    0, 1).contiguous(),
                self_attn_mask=attn_mask,
                **kwargs)
            output = output.transpose(0, 1)
            intermediate.append(self.norm(output))

            # human update
            if layer_id < self.num_box_decoder_layers:
                delta_unsig = self.bbox_embed[layer_id](output)
                new_reference_points = delta_unsig + inverse_sigmoid(
                    reference_points)
                new_reference_points = new_reference_points.sigmoid()

            # query expansion
            if layer_id == self.num_box_decoder_layers - 1:
                dn_output = output[:effect_num_dn]
                dn_new_reference_points = new_reference_points[:effect_num_dn]
                class_unselected = self.class_embed[layer_id](
                    output)[effect_num_dn:]
                topk_proposals = torch.topk(
                    class_unselected.max(-1)[0], inter_select_number, dim=0)[1]
                new_reference_points_for_box = torch.gather(
                    new_reference_points[effect_num_dn:], 0,
                    topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                new_output_for_box = torch.gather(
                    output[effect_num_dn:], 0,
                    topk_proposals.unsqueeze(-1).repeat(1, 1, self.embed_dims))
                bs = new_output_for_box.shape[1]
                new_output_for_keypoint = new_output_for_box[:, None, :, :] \
                    + self.keypoint_embed.weight[None, :, None, :]
                if self.num_keypoints == 17:
                    delta_xy = self.pose_embed[-1](new_output_for_keypoint)[
                        ..., :2]
                else:
                    delta_xy = self.pose_embed[0](new_output_for_keypoint)[
                        ..., :2]
                keypoint_xy = (inverse_sigmoid(
                    new_reference_points_for_box[..., :2][:, None]) +
                               delta_xy).sigmoid()
                num_queries, _, bs, _ = keypoint_xy.shape
                keypoint_wh_weight = self.hw.weight.unsqueeze(0).unsqueeze(
                    -2).repeat(num_queries, 1, bs, 1).sigmoid()
                keypoint_wh = keypoint_wh_weight * \
                    new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat(
                    (keypoint_xy, keypoint_wh), dim=-1)
                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1),
                     new_reference_points_for_keypoint),
                    dim=1).flatten(0, 1)
                output = torch.cat(
                    (new_output_for_box.unsqueeze(1), new_output_for_keypoint),
                    dim=1).flatten(0, 1)
                new_reference_points = torch.cat(
                    (dn_new_reference_points, new_reference_points), dim=0)
                output = torch.cat((dn_output, output), dim=0)
                attn_mask = human2pose_attn_mask

            # human-to-keypoints update
            if layer_id >= self.num_box_decoder_layers:
                effect_num_dn = self.num_dn if self.training else 0
                inter_select_number = self.num_group
                ref_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_dn = output[:effect_num_dn]
                output_bbox_norm = output[effect_num_dn:][0::(
                    self.num_keypoints + 1)]
                ref_before_sigmoid_bbox_dn = \
                    ref_before_sigmoid[:effect_num_dn]
                ref_before_sigmoid_bbox_norm = \
                    ref_before_sigmoid[effect_num_dn:][0::(
                        self.num_keypoints + 1)]
                delta_unsig_dn = self.bbox_embed[layer_id](output_bbox_dn)
                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_dn = delta_unsig_dn + ref_before_sigmoid_bbox_dn
                outputs_unsig_norm = delta_unsig_norm + \
                    ref_before_sigmoid_bbox_norm
                new_reference_points_for_box_dn = outputs_unsig_dn.sigmoid()
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid(
                )
                output_kpt = output[effect_num_dn:].index_select(
                    0, torch.tensor(self.kpt_index, device=output.device))
                delta_xy_unsig = self.pose_embed[layer_id -
                                                 self.num_box_decoder_layers](
                                                     output_kpt)
                outputs_unsig = ref_before_sigmoid[
                    effect_num_dn:].index_select(
                        0, torch.tensor(self.kpt_index,
                                        device=output.device)).clone()
                delta_hw_unsig = self.pose_hw_embed[
                    layer_id - self.num_box_decoder_layers](
                        output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid()
                bs = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat(
                    (new_reference_points_for_box_norm.unsqueeze(1),
                     new_reference_points_for_keypoint.view(
                         -1, self.num_keypoints, bs, 4)),
                    dim=1).flatten(0, 1)
                new_reference_points = torch.cat(
                    (new_reference_points_for_box_dn,
                     new_reference_points_norm),
                    dim=0)

            reference_points = new_reference_points.detach()
            intermediate_reference_points.append(reference_points)

        decoder_outputs = [itm_out.transpose(0, 1) for itm_out in intermediate]
        reference_points = [
            itm_refpoint.transpose(0, 1)
            for itm_refpoint in intermediate_reference_points
        ]

        return decoder_outputs, reference_points

    @staticmethod
    def get_proposal_pos_embed(pos_tensor: Tensor,
                               temperature: int = 10000,
                               num_pos_feats: int = 128) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            pos_tensor (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """

        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos


class EDPoseOutHead(BaseModule):
    """Final Head of EDPose: `Explicit Box Detection Unifies End-to-End Multi-
    Person Pose Estimation.

    Args:
        num_classes (int): The number of classes.
        num_keypoints (int): The number of datasets' body keypoints.
        num_queries (int): The number of queries.
        cls_no_bias (bool): Weather add the bias to class embed.
        embed_dims (int): The dims of embed.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        refine_queries_num (int): The number of refines queries after
            decoders.
        num_box_decoder_layers (int): The number of bbox decoder layer.
        num_group (int): The number of groups.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        dec_pred_class_embed_share (bool): Whether to share parameters
            for all the class prediction layers. Defaults to `False`.
        dec_pred_bbox_embed_share (bool): Whether to share parameters
            for all the bbox prediction layers. Defaults to `False`.
        dec_pred_pose_embed_share (bool): Whether to share parameters
            for all the pose prediction layers. Defaults to `False`.
    """

    def __init__(self,
                 num_classes,
                 num_keypoints: int = 17,
                 num_queries: int = 900,
                 cls_no_bias: bool = False,
                 embed_dims: int = 256,
                 as_two_stage: bool = False,
                 refine_queries_num: int = 100,
                 num_box_decoder_layers: int = 2,
                 num_group: int = 100,
                 num_pred_layer: int = 6,
                 dec_pred_class_embed_share: bool = False,
                 dec_pred_bbox_embed_share: bool = False,
                 dec_pred_pose_embed_share: bool = False,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.as_two_stage = as_two_stage
        self.num_classes = num_classes
        self.refine_queries_num = refine_queries_num
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_keypoints = num_keypoints
        self.num_queries = num_queries

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.dec_pred_pose_embed_share = dec_pred_pose_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(
            self.embed_dims, self.num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _bbox_embed = FFN(self.embed_dims, self.embed_dims, 4, 3)
        _pose_embed = FFN(self.embed_dims, self.embed_dims, 2, 3)
        _pose_hw_embed = FFN(self.embed_dims, self.embed_dims, 2, 3)

        self.num_group = num_group
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(num_pred_layer)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(num_pred_layer)
            ]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(num_pred_layer)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed) for i in range(num_pred_layer)
            ]

        if num_keypoints == 17:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [
                    _pose_embed
                    for i in range(num_pred_layer - num_box_decoder_layers + 1)
                ]
            else:
                pose_embed_layerlist = [
                    copy.deepcopy(_pose_embed)
                    for i in range(num_pred_layer - num_box_decoder_layers + 1)
                ]
        else:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [
                    _pose_embed
                    for i in range(num_pred_layer - num_box_decoder_layers)
                ]
            else:
                pose_embed_layerlist = [
                    copy.deepcopy(_pose_embed)
                    for i in range(num_pred_layer - num_box_decoder_layers)
                ]

        pose_hw_embed_layerlist = [
            _pose_hw_embed
            for i in range(num_pred_layer - num_box_decoder_layers)
        ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed = nn.ModuleList(pose_hw_embed_layerlist)

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""

        for m in self.bbox_embed:
            constant_init(m[-1], 0, bias=0)
        for m in self.pose_embed:
            constant_init(m[-1], 0, bias=0)

    def forward(self, hidden_states: List[Tensor], references: List[Tensor],
                mask_dict: Dict, hidden_states_enc: Tensor,
                referens_enc: Tensor, batch_data_samples) -> Dict:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - pred_logits (Tensor): Outputs from the
              classification head, the socres of every bboxes.
            - pred_boxes (Tensor): The output boxes.
            - pred_keypoints (Tensor): The output keypoints.
        """
        # update human boxes
        effec_dn_num = self.refine_queries_num if self.training else 0
        outputs_coord_list = []
        outputs_class = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed,
                      layer_hs) in enumerate(
                          zip(references[:-1], self.bbox_embed,
                              self.class_embed, hidden_states)):
            if dec_lid < self.num_box_decoder_layers:
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                    layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs)
                outputs_coord_list.append(layer_outputs_unsig)
                outputs_class.append(layer_cls)
            else:
                layer_hs_bbox_dn = layer_hs[:, :effec_dn_num, :]
                layer_hs_bbox_norm = \
                    layer_hs[:, effec_dn_num:, :][:, 0::(
                        self.num_keypoints + 1), :]
                bs = layer_ref_sig.shape[0]
                ref_before_sigmoid_bbox_dn = \
                    layer_ref_sig[:, : effec_dn_num, :]
                ref_before_sigmoid_bbox_norm = \
                    layer_ref_sig[:, effec_dn_num:, :][:, 0::(
                        self.num_keypoints + 1), :]
                layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                layer_outputs_unsig_dn = layer_delta_unsig_dn + \
                    inverse_sigmoid(ref_before_sigmoid_bbox_dn)
                layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                layer_outputs_unsig_norm = layer_delta_unsig_norm + \
                    inverse_sigmoid(ref_before_sigmoid_bbox_norm)
                layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                layer_outputs_unsig = torch.cat(
                    (layer_outputs_unsig_dn, layer_outputs_unsig_norm), dim=1)
                layer_cls_dn = layer_cls_embed(layer_hs_bbox_dn)
                layer_cls_norm = layer_cls_embed(layer_hs_bbox_norm)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)
                outputs_class.append(layer_cls)
                outputs_coord_list.append(layer_outputs_unsig)

        # update keypoints boxes
        outputs_keypoints_list = []
        kpt_index = [
            x for x in range(self.num_group * (self.num_keypoints + 1))
            if x % (self.num_keypoints + 1) != 0
        ]
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(
                zip(references[:-1], hidden_states)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_res = layer_hs.new_zeros(
                    (bs, self.num_queries, self.num_keypoints * 3))
                outputs_keypoints_list.append(layer_res)
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_kpt = \
                    layer_hs[:, effec_dn_num:, :].index_select(
                        1, torch.tensor(kpt_index, device=layer_hs.device))
                delta_xy_unsig = self.pose_embed[dec_lid -
                                                 self.num_box_decoder_layers](
                                                     layer_hs_kpt)
                layer_ref_sig_kpt = \
                    layer_ref_sig[:, effec_dn_num:, :].index_select(
                        1, torch.tensor(kpt_index, device=layer_hs.device))
                layer_outputs_unsig_keypoints = delta_xy_unsig + \
                    inverse_sigmoid(layer_ref_sig_kpt[..., :2])
                vis_xy_unsig = torch.ones_like(
                    layer_outputs_unsig_keypoints,
                    device=layer_outputs_unsig_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_keypoints,
                                 vis_xy_unsig[:, :, 0].unsqueeze(-1)),
                                dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape(
                    (bs, self.num_group, self.num_keypoints, 3)).flatten(2, 3)
                layer_res = self.keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_keypoints_list.append(layer_res)

        dn_mask_dict = mask_dict
        if self.refine_queries_num > 0 and dn_mask_dict is not None:
            outputs_class, outputs_coord_list, outputs_keypoints_list = \
                self.dn_post_process2(
                    outputs_class, outputs_coord_list,
                    outputs_keypoints_list, dn_mask_dict
                )

        for _out_class, _out_bbox, _out_keypoint in zip(
                outputs_class, outputs_coord_list, outputs_keypoints_list):
            assert _out_class.shape[1] == \
                _out_bbox.shape[1] == _out_keypoint.shape[1]

        return outputs_class[-1], outputs_coord_list[
            -1], outputs_keypoints_list[-1]

    def keypoint_xyzxyz_to_xyxyzz(self, keypoints: torch.Tensor):
        """
        Args:
            keypoints (torch.Tensor): ..., 51
        """
        res = torch.zeros_like(keypoints)
        num_points = keypoints.shape[-1] // 3
        res[..., 0:2 * num_points:2] = keypoints[..., 0::3]
        res[..., 1:2 * num_points:2] = keypoints[..., 1::3]
        res[..., 2 * num_points:] = keypoints[..., 2::3]
        return res


@MODELS.register_module()
class EDPoseHead(TransformerHead):
    """Head introduced in `Explicit Box Detection Unifies End-to-End Multi-
    Person Pose Estimation`_ by J Yang1 et al (2023). The head is composed of
    Encoder, Decoder and Out_head.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/ED-Pose>`_.

    More details can be found in the `paper
    <https://arxiv.org/pdf/2302.01593.pdf>`_ .

    Args:
        num_queries (int): Number of query in Transformer.
        num_feature_levels (int): Number of feature levels. Defaults to 4.
        num_keypoints (int): Number of keypoints. Defaults to 4.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        out_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding final out head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer position encoding. Defaults None.
        denosing_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            human query denoising training strategy.
        data_decoder (:obj:`ConfigDict` or dict, optional): Config of the
            data decoder which transform the results from output space to
            input space.
        dec_pred_class_embed_share (bool): Whether to share the class embed
            layer. Default False.
        dec_pred_bbox_embed_share (bool): Whether to share the bbox embed
            layer. Default False.
        refine_queries_num (int): Number of refined human content queries
            and their position queries .
        two_stage_keep_all_tokens (bool): Whether to keep all tokens.
    """

    def __init__(self,
                 num_queries: int = 100,
                 num_feature_levels: int = 4,
                 num_keypoints: int = 17,
                 as_two_stage: bool = False,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 out_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 data_decoder: OptConfigType = None,
                 denosing_cfg: OptConfigType = None,
                 dec_pred_class_embed_share: bool = False,
                 dec_pred_bbox_embed_share: bool = False,
                 refine_queries_num: int = 100,
                 two_stage_keep_all_tokens: bool = False) -> None:

        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.refine_queries_num = refine_queries_num
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_heads = decoder['layer_cfg']['self_attn_cfg']['num_heads']
        self.num_group = decoder['num_group']
        self.num_keypoints = num_keypoints
        self.denosing_cfg = denosing_cfg
        if data_decoder is not None:
            self.data_decoder = KEYPOINT_CODECS.build(data_decoder)
        else:
            self.data_decoder = None

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            out_head=out_head,
            positional_encoding=positional_encoding,
            num_queries=num_queries)

        self.positional_encoding = PositionEmbeddingSineHW(
            **self.positional_encoding_cfg)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = EDPoseDecoder(
            num_keypoints=num_keypoints, **self.decoder_cfg)
        self.out_head = EDPoseOutHead(
            num_keypoints=num_keypoints,
            as_two_stage=as_two_stage,
            refine_queries_num=refine_queries_num,
            **self.out_head_cfg,
            **self.decoder_cfg)

        self.embed_dims = self.encoder.embed_dims
        self.label_enc = nn.Embedding(
            self.denosing_cfg['dn_labelbook_size'] + 1, self.embed_dims)

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims)
            self.refpoint_embedding = nn.Embedding(self.num_queries, 4)

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.decoder.bbox_embed = self.out_head.bbox_embed
        self.decoder.pose_embed = self.out_head.pose_embed
        self.decoder.pose_hw_embed = self.out_head.pose_hw_embed
        self.decoder.class_embed = self.out_head.class_embed

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            if dec_pred_class_embed_share and dec_pred_bbox_embed_share:
                self.enc_out_bbox_embed = self.out_head.bbox_embed[0]
            else:
                self.enc_out_bbox_embed = copy.deepcopy(
                    self.out_head.bbox_embed[0])

            if dec_pred_class_embed_share and dec_pred_bbox_embed_share:
                self.enc_out_class_embed = self.out_head.class_embed[0]
            else:
                self.enc_out_class_embed = copy.deepcopy(
                    self.out_head.class_embed[0])

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)

        nn.init.normal_(self.level_embed)

    def pre_transformer(self,
                        img_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None
                        ) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        Args:
            img_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.encoder()`.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = img_feats[0].size(0)
        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = img_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in img_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(img_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        if self.refine_queries_num > 0 or batch_data_samples is not None:
            input_query_label, input_query_bbox, humandet_attn_mask, \
                human2pose_attn_mask, mask_dict =\
                self.prepare_for_denosing(
                    batch_data_samples,
                    device=img_feats[0].device)
        else:
            assert batch_data_samples is None
            input_query_bbox = input_query_label = \
                humandet_attn_mask = human2pose_attn_mask = mask_dict = None

        encoder_inputs_dict = dict(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            humandet_attn_mask=humandet_attn_mask,
            human2pose_attn_mask=human2pose_attn_mask,
            input_query_bbox=input_query_bbox,
            input_query_label=input_query_label,
            mask_dict=mask_dict)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self,
                        img_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure is defined as:
        'pre_transformer' -> 'encoder'

        Args:
            img_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        memory = self.encoder(**encoder_inputs_dict)
        encoder_outputs_dict = dict(memory=memory, **decoder_inputs_dict)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor, input_query_bbox: Tensor,
                    input_query_label: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query` and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.
            input_query_bbox (Tensor): Denosing bbox query for training.
            input_query_label (Tensor): Denosing label query for training.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.decoder()`.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions.
        """
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unact = self.enc_out_bbox_embed(
                output_memory) + output_proposals

            topk_proposals = torch.topk(
                enc_outputs_class.max(-1)[0], self.num_queries, dim=1)[1]
            topk_coords_undetach = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_undetach.detach()
            reference_points = topk_coords_unact.sigmoid()

            query_undetach = torch.gather(
                output_memory, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.embed_dims))
            query = query_undetach.detach()

            if input_query_bbox is not None:
                reference_points = torch.cat(
                    [input_query_bbox, topk_coords_unact], dim=1).sigmoid()
                query = torch.cat([input_query_label, query], dim=1)
            if self.two_stage_keep_all_tokens:
                hidden_states_enc = output_memory.unsqueeze(0)
                referens_enc = enc_outputs_coord_unact.unsqueeze(0)
            else:
                hidden_states_enc = query_undetach.unsqueeze(0)
                referens_enc = topk_coords_undetach.sigmoid().unsqueeze(0)
        else:
            hidden_states_enc, referens_enc = None, None
            query = self.query_embedding.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)
            reference_points = \
                self.refpoint_embedding.weight[:, None, :].repeat(1, bs, 1)

            if input_query_bbox is not None:
                reference_points = torch.cat(
                    [input_query_bbox, reference_points], dim=1)
                query = torch.cat([input_query_label, query], dim=1)
            reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query, reference_points=reference_points)
        head_inputs_dict = dict(
            hidden_states_enc=hidden_states_enc, referens_enc=referens_enc)
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, memory: Tensor, memory_mask: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor, humandet_attn_mask: Tensor,
                        human2pose_attn_mask: Tensor, input_query_bbox: Tensor,
                        input_query_label: Tensor, mask_dict: Dict) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure is defined as:
        'pre_decoder' -> 'decoder'

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            humandet_attn_mask (Tensor): Human attention mask.
            human2pose_attn_mask (Tensor): Human to pose attention mask.
            input_query_bbox (Tensor): Denosing bbox query for training.
            input_query_label (Tensor): Denosing label query for training.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        decoder_in, head_in = self.pre_decoder(memory, memory_mask,
                                               spatial_shapes,
                                               input_query_bbox,
                                               input_query_label)

        inter_states, inter_references = self.decoder(
            query=decoder_in['query'].transpose(0, 1),
            value=memory.transpose(0, 1),
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=decoder_in['reference_points'].transpose(0, 1),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            humandet_attn_mask=humandet_attn_mask,
            human2pose_attn_mask=human2pose_attn_mask)
        references = inter_references
        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=references,
            mask_dict=mask_dict)
        decoder_outputs_dict.update(head_in)
        return decoder_outputs_dict

    def forward_out_head(self, batch_data_samples: OptSampleList,
                         hidden_states: List[Tensor], references: List[Tensor],
                         mask_dict: Dict, hidden_states_enc: Tensor,
                         referens_enc: Tensor) -> Tuple[Tensor]:
        """Forward function."""
        out = self.out_head(hidden_states, references, mask_dict,
                            hidden_states_enc, referens_enc,
                            batch_data_samples)
        return out

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features."""
        input_shapes = np.array(
            [d.metainfo['input_size'] for d in batch_data_samples])

        if test_cfg.get('flip_test', False):
            assert NotImplementedError(
                'flip_test is currently not supported '
                'for EDPose. Please set `model.test_cfg.flip_test=False`')
        else:
            pred_logits, pred_boxes, pred_keypoints = self.forward(
                feats, batch_data_samples)  # (B, K, D)

            pred = self.decode(
                input_shapes,
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                pred_keypoints=pred_keypoints)
        return pred

    def decode(self, input_shapes: np.ndarray, pred_logits: Tensor,
               pred_boxes: Tensor, pred_keypoints: Tensor):
        """Select the final top-k keypoints, and decode the results from
        normalize size to origin input size.

        Args:
            input_shapes (Tensor): The size of input image.
            pred_logits (Tensor): The result of score.
            pred_boxes (Tensor): The result of bbox.
            pred_keypoints (Tensor): The result of keypoints.

        Returns:
        """

        if self.data_decoder is None:
            raise RuntimeError(f'The data decoder has not been set in \
                {self.__class__.__name__}. '
                               'Please set the data decoder configs in \
                    the init parameters to '
                               'enable head methods `head.predict()` and \
                     `head.decode()`')

        preds = []

        pred_logits = pred_logits.sigmoid()
        pred_logits, pred_boxes, pred_keypoints = to_numpy(
            [pred_logits, pred_boxes, pred_keypoints])

        for input_shape, pred_logit, pred_bbox, pred_kpts in zip(
                input_shapes, pred_logits, pred_boxes, pred_keypoints):

            bboxes, keypoints, keypoint_scores = self.data_decoder.decode(
                input_shape, pred_logit, pred_bbox, pred_kpts)

            # pack outputs
            preds.append(
                InstanceData(
                    keypoints=keypoints,
                    keypoint_scores=keypoint_scores,
                    bboxes=bboxes))

        return preds

    def gen_encoder_output_proposals(self, memory: Tensor, memory_mask: Tensor,
                                     spatial_shapes: Tensor
                                     ) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """
        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_mask[:,
                                        _cur:(_cur + H * W)].view(bs, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)

        output_proposals = inverse_sigmoid(output_proposals)
        output_proposals = output_proposals.masked_fill(
            memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg

    def prepare_for_denosing(self, targets: OptSampleList, device):
        """prepare for dn components in forward function."""
        if not self.training:
            bs = len(targets)
            attn_mask_infere = torch.zeros(
                bs,
                self.num_heads,
                self.num_group * (self.num_keypoints + 1),
                self.num_group * (self.num_keypoints + 1),
                device=device,
                dtype=torch.bool)
            group_bbox_kpt = (self.num_keypoints + 1)
            kpt_index = [
                x for x in range(self.num_group * (self.num_keypoints + 1))
                if x % (self.num_keypoints + 1) == 0
            ]
            for matchj in range(self.num_group * (self.num_keypoints + 1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1) * group_bbox_kpt
                if sj > 0:
                    attn_mask_infere[:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_keypoints + 1):
                    attn_mask_infere[:, :, matchj, ej:] = True
            for match_x in range(self.num_group * (self.num_keypoints + 1)):
                if match_x % group_bbox_kpt == 0:
                    attn_mask_infere[:, :, match_x, kpt_index] = False

            attn_mask_infere = attn_mask_infere.flatten(0, 1)
            return None, None, None, attn_mask_infere, None

        # targets, dn_scalar, noise_scale = dn_args
        device = targets[0]['boxes'].device
        bs = len(targets)
        refine_queries_num = self.refine_queries_num

        # gather gt boxes and labels
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_keypoints = [t['keypoints'] for t in targets]

        # repeat them
        def get_indices_for_repeat(now_num, target_num, device='cuda'):
            """
            Input:
                - now_num: int
                - target_num: int
            Output:
                - indices: tensor[target_num]
            """
            out_indice = []
            base_indice = torch.arange(now_num).to(device)
            multiplier = target_num // now_num
            out_indice.append(base_indice.repeat(multiplier))
            residue = target_num % now_num
            out_indice.append(base_indice[torch.randint(
                0, now_num, (residue, ), device=device)])
            return torch.cat(out_indice)

        gt_boxes_expand = []
        gt_labels_expand = []
        gt_keypoints_expand = []
        for idx, (gt_boxes_i, gt_labels_i, gt_keypoint_i) in enumerate(
                zip(gt_boxes, gt_labels, gt_keypoints)):
            num_gt_i = gt_boxes_i.shape[0]
            if num_gt_i > 0:
                indices = get_indices_for_repeat(num_gt_i, refine_queries_num,
                                                 device)
                gt_boxes_expand_i = gt_boxes_i[indices]  # num_dn, 4
                gt_labels_expand_i = gt_labels_i[indices]
                gt_keypoints_expand_i = gt_keypoint_i[indices]
            else:
                # all negative samples when no gt boxes
                gt_boxes_expand_i = torch.rand(
                    refine_queries_num, 4, device=device)
                gt_labels_expand_i = torch.ones(
                    refine_queries_num, dtype=torch.int64,
                    device=device) * int(self.num_classes)
                gt_keypoints_expand_i = torch.rand(
                    refine_queries_num, self.num_keypoints * 3, device=device)
            gt_boxes_expand.append(gt_boxes_expand_i)
            gt_labels_expand.append(gt_labels_expand_i)
            gt_keypoints_expand.append(gt_keypoints_expand_i)
        gt_boxes_expand = torch.stack(gt_boxes_expand)
        gt_labels_expand = torch.stack(gt_labels_expand)
        gt_keypoints_expand = torch.stack(gt_keypoints_expand)
        knwon_boxes_expand = gt_boxes_expand.clone()
        knwon_labels_expand = gt_labels_expand.clone()

        # add noise
        if self.denosing_cfg['dn_label_noise_ratio'] > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < self.denosing_cfg['dn_label_noise_ratio']
            new_label = torch.randint_like(
                knwon_labels_expand[chosen_indice], 0,
                self.dn_labelbook_size)  # randomly put a new one here
            knwon_labels_expand[chosen_indice] = new_label

        if self.denosing_cfg['dn_box_noise_scale'] > 0:
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            knwon_boxes_expand += torch.mul(
                (torch.rand_like(knwon_boxes_expand) * 2 - 1.0),
                diff) * self.denosing_cfg['dn_box_noise_scale']
            knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        input_query_label = self.label_enc(knwon_labels_expand)
        input_query_bbox = inverse_sigmoid(knwon_boxes_expand)

        # prepare mask
        if 'group2group' in self.denosing_cfg['dn_attn_mask_type_list']:
            attn_mask = torch.zeros(
                bs,
                self.num_heads,
                refine_queries_num + self.num_queries,
                refine_queries_num + self.num_queries,
                device=device,
                dtype=torch.bool)
            attn_mask[:, :, refine_queries_num:, :refine_queries_num] = True
            for idx, (gt_boxes_i,
                      gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(refine_queries_num):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask[idx, :, matchi, :si] = True
                    if ei < refine_queries_num:
                        attn_mask[idx, :, matchi, ei:refine_queries_num] = True
            attn_mask = attn_mask.flatten(0, 1)

        if 'group2group' in self.denosing_cfg['dn_attn_mask_type_list']:
            attn_mask2 = torch.zeros(
                bs,
                self.num_heads,
                refine_queries_num + self.num_group * (self.num_keypoints + 1),
                refine_queries_num + self.num_group * (self.num_keypoints + 1),
                device=device,
                dtype=torch.bool)
            attn_mask2[:, :, refine_queries_num:, :refine_queries_num] = True
            group_bbox_kpt = (self.num_keypoints + 1)
            kpt_index = [
                x for x in range(self.num_group * (self.num_keypoints + 1))
                if x % (self.num_keypoints + 1) == 0
            ]
            for matchj in range(self.num_group * (self.num_keypoints + 1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1) * group_bbox_kpt
                if sj > 0:
                    attn_mask2[:, :, refine_queries_num:,
                               refine_queries_num:][:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_keypoints + 1):
                    attn_mask2[:, :, refine_queries_num:,
                               refine_queries_num:][:, :, matchj, ej:] = True

            for match_x in range(self.num_group * (self.num_keypoints + 1)):
                if match_x % group_bbox_kpt == 0:
                    attn_mask2[:, :, refine_queries_num:,
                               refine_queries_num:][:, :, match_x,
                                                    kpt_index] = False

            for idx, (gt_boxes_i,
                      gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(refine_queries_num):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask2[idx, :, matchi, :si] = True
                    if ei < refine_queries_num:
                        attn_mask2[idx, :, matchi,
                                   ei:refine_queries_num] = True
            attn_mask2 = attn_mask2.flatten(0, 1)

        mask_dict = {
            'pad_size': refine_queries_num,
            'known_bboxs': gt_boxes_expand,
            'known_labels': gt_labels_expand,
            'known_keypoints': gt_keypoints_expand
        }

        return input_query_label, input_query_bbox, \
            attn_mask, attn_mask2, mask_dict

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        assert NotImplementedError(
            'the training of EDPose has not been '
            'supported. Please stay tuned for further update.')
