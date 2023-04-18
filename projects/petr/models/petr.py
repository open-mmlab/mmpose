# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.models.detectors import DeformableDETR
from mmdet.models.layers import (DeformableDetrTransformerDecoder,
                                 DeformableDetrTransformerEncoder,
                                 SinePositionalEncoding)
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from .transformers import PetrTransformerDecoder


@MODELS.register_module()
class PETR(DeformableDETR):

    def __init__(self,
                 num_keypoints: int = 17,
                 hm_encoder: dict = None,
                 kpt_decoder: dict = None,
                 *args,
                 **kwargs):
        self.num_keypoints = num_keypoints
        self.hm_encoder = hm_encoder
        self.kpt_decoder = kpt_decoder
        super().__init__(*args, **kwargs)

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.encoder._register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook)
        self.decoder._register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = PetrTransformerDecoder(**self.decoder)
        self.hm_encoder = DeformableDetrTransformerEncoder(**self.hm_encoder)
        self.kpt_decoder = DeformableDetrTransformerDecoder(**self.kpt_decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims * 2)
        self.kpt_query_embedding = nn.Embedding(self.num_keypoints,
                                                self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None,
                            test_mode: bool = True) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        # encoder_inputs_dict
        #   |-  feat:               [bs, mlv_shape, 256]
        #   |-  feat_mask:          [bs, mlv_shape]
        #   |-  feat_pos:           [bs, mlv_shape, 256]
        #   |-  spatial_shapes:     [4, 2]
        #   |-  level_start_index:  [4]
        #   |-  valid_ratios:       [bs, 4, 2]
        # decoder_inputs_dict
        #   |-  memory_mask:        [bs, mlv_shape]
        #   |-  spatial_shapes      [4, 2]
        #   |-  level_start_index   [4]
        #   |-  valid_ratios        [bs, 4, 2]

        encoder_outputs_dict, heatmap_dict = self.forward_encoder(
            **encoder_inputs_dict, test_mode=test_mode)
        # encoder_outputs_dict
        #   |-  memory:             [bs, mlv_shape, 256]
        #   |-  memory_mask:        [bs, mlv_shape] (feat_mask)
        #   |-  spatial_shapes:     [4, 2]
        # heatmap_dict
        #   |-  hm_memory:          [bs, lv0_h, lv0_w, 256]
        #   |-  hm_mask:            [bs, lv0_h, lv0_w]

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        # tmp_dec_in
        #   |-  query:              [bs, num_queries, 256]
        #   |-  query_pos:          [bs, num_queries, 256]
        #   |-  memory:             [bs, mlv_shape, 256] (memory)
        #   |-  reference_points:   [bs, num_queries, 2*num_keypoints]
        # head_inputs_dict (train only)
        #   |-  enc_outputs_class:  [bs, mlv_shape, 1]
        #   |-  enc_outputs_coord:  [bs, mlv_shape, 34]

        decoder_inputs_dict.update(tmp_dec_in)
        # decoder_inputs_dict
        #   |-  query:              [bs, num_queries, 256]
        #   |-  query_pos:          [bs, num_queries, 256]
        #   |-  memory:             [bs, mlv_shape, 256] (memory)
        #   |-  memory_mask:        [bs, mlv_shape]
        #   |-  spatial_shapes      [4, 2]
        #   |-  level_start_index   [4]
        #   |-  valid_ratios        [bs, 4, 2]

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        # decoder_outputs_dict
        #   |-  hidden_states       [3, bs, num_queries, 256]
        #   |-  references          [1, 300, 34] * 4
        #   |-  all_layers_classes  [3, bs, num_queries, 1]
        #   |-  all_layers_coords   [3, bs, num_queries, 2*num_keypoints]

        kpt_decoder_inputs_dict = self.pre_kpt_decoder(
            **decoder_outputs_dict,
            batch_data_samples=batch_data_samples,
            test_mode=test_mode)
        # kpt_decoder_inputs_dict
        #   |-  pos_kpt_coords:     [max_inst, 2*num_keypoints]
        #   |-  pos_img_inds:       [max_inst]
        #   |-  det_labels:         [max_inst] (test only)
        #   |-  det_scores:         [max_inst] (test only)

        kpt_decoder_inputs_dict.update(decoder_inputs_dict)
        # kpt_decoder_inputs_dict
        #   |-  pos_kpt_coords:     [max_inst, 2*num_keypoints]
        #   |-  pos_img_inds:       [max_inst]
        #   |-  det_labels:         [max_inst]
        #   |-  query:              [bs, num_queries, 256]
        #   |-  query_pos:          [bs, num_queries, 256]
        #   |-  memory:             [bs, mlv_shape, 256] (memory)
        #   |-  memory_mask:        [bs, mlv_shape]
        #   |-  spatial_shapes      [4, 2]
        #   |-  level_start_index   [4]
        #   |-  valid_ratios        [bs, 4, 2]

        kpt_decoder_outputs_dict = self.forward_kpt_decoder(
            **kpt_decoder_inputs_dict)
        # kpt_decoder_outputs_dict (test)
        #   |-  inter_states:       [2, max_inst, num_keypoints, 256]
        #   |-  reference_points:   [max_inst, num_keypoints, 2]
        #   |-  inter_references:   [2, max_inst, num_keypoints, 2]

        dec_outputs_coord = self.forward_kpt_head(**kpt_decoder_outputs_dict)
        # dec_outputs_coord:        [2, max_inst, num_keypoints, 2]

        head_inputs_dict['dec_outputs_coord'] = dec_outputs_coord
        if test_mode:
            head_inputs_dict['det_labels'] = kpt_decoder_inputs_dict[
                'det_labels']
            head_inputs_dict['det_scores'] = kpt_decoder_inputs_dict[
                'det_scores']
        else:
            head_inputs_dict.update(heatmap_dict)
            head_inputs_dict['all_layers_classes'] = decoder_outputs_dict[
                'all_layers_classes']
            head_inputs_dict['all_layers_coords'] = decoder_outputs_dict[
                'all_layers_coords']
        # head_inputs_dict
        #   |-  enc_outputs_class:  [bs, mlv_shape, 1] (train only)
        #   |-  enc_outputs_coord:  [bs, mlv_shape, 34] (train only)
        #   |-  all_layers_classes  [3, bs, num_queries, 1] (train only)
        #   |-  all_layers_coords   [3, bs, num_queries, 2*num_keypoints] (to)
        #   |-  hm_memory:          [bs, lv0_h, lv0_w, 256] (train only)
        #   |-  hm_mask:            [bs, lv0_h, lv0_w] (train only)
        #   |-  dec_outputs_coord:  [2, max_inst, num_keypoints, 2]
        #   |-  det_labels:         [max_inst] (test only)
        #   |-  det_scores:         [max_inst] (test only)

        return head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # torch.save(dict(
        #     batch_inputs=batch_inputs.cpu(),
        #     batch_data_samples=batch_data_samples
        # ), 'notebooks/train_proc_tensors/img+ds.pth')
        # exit(0)

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples, test_mode=False)
        # head_inputs_dict
        #   |-  enc_outputs_class:  [bs, mlv_shape, 1]
        #   |-  enc_outputs_coord:  [bs, mlv_shape, 34]
        #   |-  all_layers_classes  [3, bs, num_queries, 1]
        #   |-  all_layers_coords   [3, bs, num_queries, 2*num_keypoints]
        #   |-  hm_memory:          [bs, lv0_h, lv0_w, 256]
        #   |-  hm_mask:            [bs, lv0_h, lv0_w]
        #   |-  dec_outputs_coord:  [2, max_inst, num_keypoints, 2]

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples, test_mode=True)
        # head_inputs_dict
        #   |-  dec_outputs_coord:  [2, max_inst, num_keypoints, 2]
        #   |-  det_labels:         [max_inst]
        #   |-  det_scores:         [max_inst]

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward_encoder(self,
                        feat: Tensor,
                        feat_mask: Tensor,
                        feat_pos: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        test_mode: bool = True) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)

        # only for training
        heatmap_dict = dict()
        if not test_mode:
            batch_size = memory.size(0)
            hm_memory = memory[:, level_start_index[0]:level_start_index[1], :]
            hm_pos_embed = feat_pos[:, level_start_index[0]:
                                    level_start_index[1], :]
            hm_mask = feat_mask[:, level_start_index[0]:level_start_index[1]]
            hm_memory = self.hm_encoder(
                query=hm_memory,
                query_pos=hm_pos_embed,
                key_padding_mask=hm_mask,
                spatial_shapes=spatial_shapes.narrow(0, 0, 1),
                level_start_index=level_start_index[0],
                valid_ratios=valid_ratios.narrow(1, 0, 1))
            hm_memory = hm_memory.reshape(batch_size, spatial_shapes[0, 0],
                                          spatial_shapes[0, 1], -1)
            hm_mask = hm_mask.reshape(batch_size, spatial_shapes[0, 0],
                                      spatial_shapes[0, 1])

            heatmap_dict['hm_memory'] = hm_memory
            heatmap_dict['hm_mask'] = hm_mask

        return encoder_outputs_dict, heatmap_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_coord`. They are both `None` when 'as_two_stage'
              is `False`. The dict is empty when `self.training` is `False`.
        """
        batch_size, _, c = memory.shape

        query_embed = self.query_embedding.weight
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)

        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact[..., 0::2] += output_proposals[..., 0:1]
            enc_outputs_coord_unact[..., 1::2] += output_proposals[..., 1:2]
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_coord_unact.size(-1)))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged as
                (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        references = [reference_points, *inter_references]

        all_layers_classes, all_layers_coords = self.bbox_head(
            hidden_states=inter_states, references=references)

        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=references,
            all_layers_classes=all_layers_classes,
            all_layers_coords=all_layers_coords)

        return decoder_outputs_dict

    def pre_kpt_decoder(self,
                        all_layers_classes,
                        all_layers_coords,
                        batch_data_samples,
                        test_mode=False,
                        **kwargs):

        cls_scores = all_layers_classes[-1]
        kpt_coords = all_layers_coords[-1]

        if test_mode:
            assert cls_scores.size(0) == 1, \
                f'only `batch_size=1` is supported in testing, but got ' \
                f'{cls_scores.size(0)}'

            cls_scores = cls_scores[0]
            kpt_coords = kpt_coords[0]

            max_per_img = self.test_cfg['max_per_img']
            if self.bbox_head.loss_cls.use_sigmoid:
                cls_scores = cls_scores.sigmoid()
                scores, indices = cls_scores.view(-1).topk(max_per_img)
                det_labels = indices % self.bbox_head.num_classes
                bbox_index = indices // self.bbox_head.num_classes
                kpt_coords = kpt_coords[bbox_index]
            else:
                scores, det_labels = F.softmax(
                    cls_scores, dim=-1)[..., :-1].max(-1)
                scores, bbox_index = scores.topk(max_per_img)
                kpt_coords = kpt_coords[bbox_index]
                det_labels = det_labels[bbox_index]

            kpt_weights = torch.ones_like(kpt_coords)

            kpt_decoder_inputs_dict = dict(
                det_labels=det_labels,
                det_scores=scores,
            )

        else:

            batch_gt_instances = [ds.gt_instances for ds in batch_data_samples]
            batch_img_metas = [ds.metainfo for ds in batch_data_samples]

            num_imgs = cls_scores.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            kpt_coords_list = [
                kpt_coords[i].reshape(-1, self.bbox_head.num_keypoints, 2)
                for i in range(num_imgs)
            ]

            cls_reg_targets = self.bbox_head.get_targets(
                cls_scores_list,
                kpt_coords_list,
                batch_gt_instances,
                batch_img_metas,
                cache_targets=True)
            kpt_weights = torch.cat(cls_reg_targets[4])
            kpt_coords = kpt_coords.flatten(0, 1)
            kpt_decoder_inputs_dict = {}

        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_coords = torch.zeros_like(kpt_coords[:1])
            pos_img_inds = kpt_coords.new_zeros([1], dtype=torch.int64)
        else:
            pos_kpt_coords = kpt_coords[pos_inds]
            pos_img_inds = (pos_inds.nonzero() /
                            self.num_queries).squeeze(1).to(torch.int64)

        kpt_decoder_inputs_dict.update(
            dict(
                pos_kpt_coords=pos_kpt_coords,
                pos_img_inds=pos_img_inds,
            ))
        return kpt_decoder_inputs_dict

    def forward_kpt_decoder(self, memory, memory_mask, pos_kpt_coords,
                            pos_img_inds, spatial_shapes, level_start_index,
                            valid_ratios, **kwargs):

        kpt_query_embedding = self.kpt_query_embedding.weight
        query_pos, query = torch.split(
            kpt_query_embedding, kpt_query_embedding.size(1) // 2, dim=1)
        pos_num = pos_kpt_coords.size(0)
        query_pos = query_pos.unsqueeze(0).expand(pos_num, -1, -1)
        query = query.unsqueeze(0).expand(pos_num, -1, -1)
        reference_points = pos_kpt_coords.reshape(pos_num,
                                                  pos_kpt_coords.size(1) // 2,
                                                  2).detach()
        pos_memory = memory[pos_img_inds, :, :]
        memory_mask = memory_mask[pos_img_inds, :]
        valid_ratios = valid_ratios[pos_img_inds, ...]

        # forward_kpt_decoder
        inter_states, inter_references = self.kpt_decoder(
            query=query,
            key=None,
            value=pos_memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.kpt_branches)

        kpt_decoder_outputs_dict = dict(
            inter_states=inter_states,
            reference_points=reference_points,
            inter_references=inter_references,
        )

        return kpt_decoder_outputs_dict

    def forward_kpt_head(self, inter_states, reference_points,
                         inter_references):
        outputs_kpts = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = reference_points
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp_kpt = self.bbox_head.kpt_branches[lvl](inter_states[lvl])
            assert reference.shape[-1] == 2
            tmp_kpt += reference
            outputs_kpt = tmp_kpt.sigmoid()
            outputs_kpts.append(outputs_kpt)

        outputs_kpts = torch.stack(outputs_kpts)
        return outputs_kpts

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert the state dict from official repo to a
        compatible format :class:`PETR`.

        The hook will be automatically registered during initialization.
        """
        return
        
        if 'mmengine_version' in local_meta:
            return

        if local_meta.get('mmdet_verion', '0') > '3':
            return

        mappings = OrderedDict()
        mappings['bbox_head.transformer.'] = ''
        mappings['level_embeds'] = 'level_embed'
        mappings['bbox_head.query_embedding'] = 'query_embedding'
        mappings['refine_query_embedding'] = 'kpt_query_embedding'
        mappings['attentions.0'] = 'self_attn'
        mappings['attentions.1'] = 'cross_attn'
        mappings['ffns.0'] = 'ffn'
        mappings['bbox_head.kpt_branches'] = 'bbox_head.reg_branches'
        mappings['bbox_head.refine_kpt_branches'] = 'bbox_head.kpt_branches'
        mappings['refine_decoder'] = 'kpt_decoder'
        mappings['bbox_head.fc_hm'] = 'bbox_head.heatmap_fc'
        mappings['enc_output_norm'] = 'memory_trans_norm'
        mappings['enc_output'] = 'memory_trans_fc'

        # convert old-version state dict
        for old_key, new_key in mappings.items():
            keys = list(state_dict.keys())
            for k in keys:
                if old_key in k:
                    v = state_dict.pop(k)
                    k = k.replace(old_key, new_key)
                    state_dict[k] = v
