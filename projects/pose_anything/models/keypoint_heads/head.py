from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmengine.model import xavier_init
from models.utils import build_positional_encoding, build_transformer

# from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmpose.evaluation import keypoint_pck_accuracy
from mmpose.models import HEADS
from mmpose.models.utils.ops import resize


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = coords.copy()
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class TokenDecodeMLP(nn.Module):
    """The MLP used to predict coordinates from the support keypoints
    tokens."""

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels=2,
                 num_layers=3):
        super(TokenDecodeMLP, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


@HEADS.register_module()
class PoseHead(nn.Module):
    """In two stage regression A3, the proposal generator are moved into
    transformer.

    All valid proposals will be added with an positional embedding to better
    regress the location
    """

    def __init__(self,
                 in_channels,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 encoder_positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=512,
                     normalize=True),
                 share_kpt_branch=False,
                 num_decoder_layer=3,
                 with_heatmap_loss=False,
                 with_bb_loss=False,
                 bb_temperature=0.2,
                 heatmap_loss_weight=2.0,
                 support_order_dropout=-1,
                 extra=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.encoder_positional_encoding = build_positional_encoding(
            encoder_positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.d_model
        self.with_heatmap_loss = with_heatmap_loss
        self.with_bb_loss = with_bb_loss
        self.bb_temperature = bb_temperature
        self.heatmap_loss_weight = heatmap_loss_weight
        self.support_order_dropout = support_order_dropout

        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            (f' be exactly 2 times of '
             f'num_feats. Found'
             f' {self.embed_dims} '
             f'and {num_feats}.')
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.query_proj = Linear(self.in_channels, self.embed_dims)
        # Instantiate the proposal generator and subsequent keypoint branch.
        kpt_branch = TokenDecodeMLP(
            in_channels=self.embed_dims, hidden_channels=self.embed_dims)
        if share_kpt_branch:
            self.kpt_branch = nn.ModuleList(
                [kpt_branch for i in range(num_decoder_layer)])
        else:
            self.kpt_branch = nn.ModuleList(
                [deepcopy(kpt_branch) for i in range(num_decoder_layer)])

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        # initialization for input_proj & prediction head
        for mlp in self.kpt_branch:
            nn.init.constant_(mlp.mlp[-1].weight.data, 0)
            nn.init.constant_(mlp.mlp[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)

        nn.init.xavier_uniform_(self.query_proj.weight, gain=1)
        nn.init.constant_(self.query_proj.bias, 0)

    def forward(self, x, feature_s, target_s, mask_s, skeleton):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        # process query image feature
        x = self.input_proj(x)
        bs, dim, h, w = x.shape

        # Disable the support keypoint positional embedding
        support_order_embedding = x.new_zeros(
            (bs, self.embed_dims, 1, target_s[0].shape[1])).to(torch.bool)

        # Feature map pos embedding
        masks = x.new_zeros(
            (x.shape[0], x.shape[2], x.shape[3])).to(torch.bool)
        pos_embed = self.positional_encoding(masks)

        # process keypoint token feature
        query_embed_list = []
        for i, (feature, target) in enumerate(zip(feature_s, target_s)):
            # resize the support feature back to the heatmap sizes.
            resized_feature = resize(
                input=feature,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False)
            target = target / (
                target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
            support_keypoints = target.flatten(2) @ resized_feature.flatten(
                2).permute(0, 2, 1)
            query_embed_list.append(support_keypoints)

        support_keypoints = torch.mean(torch.stack(query_embed_list, dim=0), 0)
        support_keypoints = support_keypoints * mask_s
        support_keypoints = self.query_proj(support_keypoints)
        masks_query = (~mask_s.to(torch.bool)).squeeze(
            -1)  # True indicating this query matched no actual joints.

        # outs_dec: [nb_dec, bs, num_query, c]
        # memory: [bs, c, h, w]
        # x = Query image feature,
        # support_keypoints = Support keypoint feature
        outs_dec, initial_proposals, out_points, similarity_map = (
            self.transformer(x, masks, support_keypoints, pos_embed,
                             support_order_embedding, masks_query,
                             self.positional_encoding, self.kpt_branch,
                             skeleton))

        output_kpts = []
        for idx in range(outs_dec.shape[0]):
            layer_delta_unsig = self.kpt_branch[idx](outs_dec[idx])
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                out_points[idx])
            output_kpts.append(layer_outputs_unsig.sigmoid())

        return torch.stack(
            output_kpts, dim=0), initial_proposals, similarity_map

    def get_loss(self, output, initial_proposals, similarity_map, target,
                 target_heatmap, target_weight, target_sizes):
        # Calculate top-down keypoint loss.
        losses = dict()
        # denormalize the predicted coordinates.
        num_dec_layer, bs, nq = output.shape[:3]
        target_sizes = target_sizes.to(output.device)  # [bs, 1, 2]
        target = target / target_sizes
        target = target[None, :, :, :].repeat(num_dec_layer, 1, 1, 1)

        # set the weight for unset query point to be zero
        normalizer = target_weight.squeeze(dim=-1).sum(dim=-1)  # [bs, ]
        normalizer[normalizer == 0] = 1

        # compute the heatmap loss
        if self.with_heatmap_loss:
            losses['heatmap_loss'] = self.heatmap_loss(
                similarity_map, target_heatmap, target_weight,
                normalizer) * self.heatmap_loss_weight

        # compute l1 loss for initial_proposals
        proposal_l1_loss = F.l1_loss(
            initial_proposals, target[0], reduction='none')
        proposal_l1_loss = proposal_l1_loss.sum(
            dim=-1, keepdim=False) * target_weight.squeeze(dim=-1)
        proposal_l1_loss = proposal_l1_loss.sum(
            dim=-1, keepdim=False) / normalizer  # [bs, ]
        losses['proposal_loss'] = proposal_l1_loss.sum() / bs

        # compute l1 loss for each layer
        for idx in range(num_dec_layer):
            layer_output, layer_target = output[idx], target[idx]
            l1_loss = F.l1_loss(
                layer_output, layer_target, reduction='none')  # [bs, query, 2]
            l1_loss = l1_loss.sum(
                dim=-1, keepdim=False) * target_weight.squeeze(
                    dim=-1)  # [bs, query]
            # normalize the loss for each sample with the number of visible
            # joints
            l1_loss = l1_loss.sum(dim=-1, keepdim=False) / normalizer  # [bs, ]
            losses['l1_loss' + '_layer' + str(idx)] = l1_loss.sum() / bs

        return losses

    def get_max_coords(self, heatmap, heatmap_size=64):
        B, C, H, W = heatmap.shape
        heatmap = heatmap.view(B, C, -1)
        max_cor = heatmap.argmax(dim=2)
        row, col = torch.floor(max_cor / heatmap_size), max_cor % heatmap_size
        support_joints = torch.cat((row.unsqueeze(-1), col.unsqueeze(-1)),
                                   dim=-1)
        return support_joints

    def heatmap_loss(self, similarity_map, target_heatmap, target_weight,
                     normalizer):
        # similarity_map: [bs, num_query, h, w]
        # target_heatmap: [bs, num_query, sh, sw]
        # target_weight: [bs, num_query, 1]

        # preprocess the similarity_map
        h, w = similarity_map.shape[-2:]
        # similarity_map = torch.clamp(similarity_map, 0.0, None)
        similarity_map = similarity_map.sigmoid()

        target_heatmap = F.interpolate(
            target_heatmap, size=(h, w), mode='bilinear')
        target_heatmap = (target_heatmap /
                          (target_heatmap.max(dim=-1)[0].max(dim=-1)[0] +
                           1e-10)[:, :, None, None]
                          )  # make sure sum of each query is 1

        l2_loss = F.mse_loss(
            similarity_map, target_heatmap, reduction='none')  # bs, nq, h, w
        l2_loss = l2_loss * target_weight[:, :, :, None]  # bs, nq, h, w
        l2_loss = l2_loss.flatten(2, 3).sum(-1) / (h * w)  # bs, nq
        l2_loss = l2_loss.sum(-1) / normalizer  # bs,

        return l2_loss.mean()

    def get_accuracy(self,
                     output,
                     target,
                     target_weight,
                     target_sizes,
                     height=256):
        """Calculate accuracy for top-down keypoint loss.

        Args:
            output (torch.Tensor[NxKx2]): estimated keypoints in ABSOLUTE
            coordinates.
            target (torch.Tensor[NxKx2]): gt keypoints in ABSOLUTE coordinates.
            target_weight (torch.Tensor[NxKx1]): Weights across different
            joint types.
            target_sizes (torch.Tensor[Nx2): shapes of the image.
        """
        # NOTE: In POMNet, PCK is estimated on 1/8 resolution, which is
        # slightly different here.

        accuracy = dict()
        output = output * float(height)
        output, target, target_weight, target_sizes = (
            output.detach().cpu().numpy(), target.detach().cpu().numpy(),
            target_weight.squeeze(-1).long().detach().cpu().numpy(),
            target_sizes.squeeze(1).detach().cpu().numpy())

        _, avg_acc, _ = keypoint_pck_accuracy(
            output,
            target,
            target_weight.astype(np.bool8),
            thr=0.2,
            normalize=target_sizes)
        accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def decode(self, img_metas, output, img_size, **kwargs):
        """Decode the predicted keypoints from prediction.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)
        W, H = img_size
        output = output * np.array([
            W, H
        ])[None, None, :]  # [bs, query, 2], coordinates with recovered shapes.

        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])

            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(
                    img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])

        preds = np.zeros(output.shape)
        for idx in range(output.shape[0]):
            preds[i] = transform_preds(
                output[i],
                c[i],
                s[i], [W, H],
                use_udp=self.test_cfg.get('use_udp', False))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = 1.0
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result
