import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import xavier_init
from models.utils.builder import TRANSFORMER
from torch import Tensor


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ProposalGenerator(nn.Module):

    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, dynamic_proj_dim), nn.ReLU(),
            nn.Linear(dynamic_proj_dim, hidden_dim))
        self.dynamic_act = nn.Tanh()

    def forward(self, query_feat, support_feat, spatial_shape):
        """
        Args:
            support_feat: [query, bs, c]
            query_feat: [hw, bs, c]
            spatial_shape: h, w
        """
        device = query_feat.device
        _, bs, c = query_feat.shape
        h, w = spatial_shape
        side_normalizer = torch.tensor([w, h]).to(query_feat.device)[
            None, None, :]  # [bs, query, 2], Normalize the coord to [0,1]

        query_feat = query_feat.transpose(0, 1)
        support_feat = support_feat.transpose(0, 1)
        nq = support_feat.shape[1]

        fs_proj = self.support_proj(support_feat)  # [bs, query, c]
        fq_proj = self.query_proj(query_feat)  # [bs, hw, c]
        pattern_attention = self.dynamic_act(
            self.dynamic_proj(fs_proj))  # [bs, query, c]

        fs_feat = (pattern_attention + 1) * fs_proj  # [bs, query, c]
        similarity = torch.bmm(fq_proj,
                               fs_feat.transpose(1, 2))  # [bs, hw, query]
        similarity = similarity.transpose(1, 2).reshape(bs, nq, h, w)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0.5, h - 0.5, h, dtype=torch.float32, device=device),  # (h, w)
            torch.linspace(
                0.5, w - 0.5, w, dtype=torch.float32, device=device))

        # compute softmax and sum up
        coord_grid = torch.stack([grid_x, grid_y],
                                 dim=0).unsqueeze(0).unsqueeze(0).repeat(
                                     bs, nq, 1, 1, 1)  # [bs, query, 2, h, w]
        coord_grid = coord_grid.permute(0, 1, 3, 4, 2)  # [bs, query, h, w, 2]
        similarity_softmax = similarity.flatten(2, 3).softmax(
            dim=-1)  # [bs, query, hw]
        similarity_coord_grid = similarity_softmax[:, :, :,
                                                   None] * coord_grid.flatten(
                                                       2, 3)
        proposal_for_loss = similarity_coord_grid.sum(
            dim=2, keepdim=False)  # [bs, query, 2]
        proposal_for_loss = proposal_for_loss / side_normalizer

        max_pos = torch.argmax(
            similarity.reshape(bs, nq, -1), dim=-1,
            keepdim=True)  # (bs, nq, 1)
        max_mask = F.one_hot(max_pos, num_classes=w * h)  # (bs, nq, 1, w*h)
        max_mask = max_mask.reshape(bs, nq, w,
                                    h).type(torch.float)  # (bs, nq, w, h)
        local_max_mask = F.max_pool2d(
            input=max_mask, kernel_size=3, stride=1,
            padding=1).reshape(bs, nq, w * h, 1)  # (bs, nq, w*h, 1)
        '''
        proposal = (similarity_coord_grid * local_max_mask).sum(
            dim=2, keepdim=False) / torch.count_nonzero(
                local_max_mask, dim=2)
        '''
        # first, extract the local probability map with the mask
        local_similarity_softmax = similarity_softmax[:, :, :,
                                                      None] * local_max_mask

        # then, re-normalize the local probability map
        local_similarity_softmax = local_similarity_softmax / (
            local_similarity_softmax.sum(dim=-2, keepdim=True) + 1e-10
        )  # [bs, nq, w*h, 1]

        # point-wise mulplication of local probability map and coord grid
        proposals = local_similarity_softmax * coord_grid.flatten(
            2, 3)  # [bs, nq, w*h, 2]

        # sum the mulplication to obtain the final coord proposals
        proposals = proposals.sum(dim=2) / side_normalizer  # [bs, nq, 2]

        return proposal_for_loss, similarity, proposals


@TRANSFORMER.register_module()
class EncoderDecoder(nn.Module):

    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 graph_decoder=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 similarity_proj_dim=256,
                 dynamic_proj_dim=128,
                 return_intermediate_dec=True,
                 look_twice=False,
                 detach_support_feat=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = GraphTransformerDecoderLayer(d_model, nhead,
                                                     dim_feedforward, dropout,
                                                     activation,
                                                     normalize_before,
                                                     graph_decoder)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = GraphTransformerDecoder(
            d_model,
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            look_twice=look_twice,
            detach_support_feat=detach_support_feat)

        self.proposal_generator = ProposalGenerator(
            hidden_dim=d_model,
            proj_dim=similarity_proj_dim,
            dynamic_proj_dim=dynamic_proj_dim)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

    def forward(self,
                src,
                mask,
                support_embed,
                pos_embed,
                support_order_embed,
                query_padding_mask,
                position_embedding,
                kpt_branch,
                skeleton,
                return_attn_map=False):

        bs, c, h, w = src.shape

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        support_order_embed = support_order_embed.flatten(2).permute(2, 0, 1)
        pos_embed = torch.cat((pos_embed, support_order_embed))
        query_embed = support_embed.transpose(0, 1)
        mask = mask.flatten(1)

        query_embed, refined_support_embed = self.encoder(
            src,
            query_embed,
            src_key_padding_mask=mask,
            query_key_padding_mask=query_padding_mask,
            pos=pos_embed)

        # Generate initial proposals and corresponding positional embedding.
        initial_proposals_for_loss, similarity_map, initial_proposals = (
            self.proposal_generator(
                query_embed, refined_support_embed, spatial_shape=[h, w]))
        initial_position_embedding = position_embedding.forward_coordinates(
            initial_proposals)

        outs_dec, out_points, attn_maps = self.decoder(
            refined_support_embed,
            query_embed,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=initial_position_embedding,
            tgt_key_padding_mask=query_padding_mask,
            position_embedding=position_embedding,
            initial_proposals=initial_proposals,
            kpt_branch=kpt_branch,
            skeleton=skeleton,
            return_attn_map=return_attn_map)

        return outs_dec.transpose(
            1, 2), initial_proposals_for_loss, out_points, similarity_map


class GraphTransformerDecoder(nn.Module):

    def __init__(self,
                 d_model,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False,
                 look_twice=False,
                 detach_support_feat=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(d_model, d_model, d_model, num_layers=2)
        self.look_twice = look_twice
        self.detach_support_feat = detach_support_feat

    def forward(self,
                support_feat,
                query_feat,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
                position_embedding=None,
                initial_proposals=None,
                kpt_branch=None,
                skeleton=None,
                return_attn_map=False):
        """
        position_embedding: Class used to compute positional embedding
        initial_proposals: [bs, nq, 2], normalized coordinates of initial
        proposals kpt_branch: MLP used to predict the offsets for each query.
        """

        refined_support_feat = support_feat
        intermediate = []
        attn_maps = []
        bi = initial_proposals.detach()
        bi_tag = initial_proposals.detach()
        query_points = [initial_proposals.detach()]

        tgt_key_padding_mask_remove_all_true = tgt_key_padding_mask.clone().to(
            tgt_key_padding_mask.device)
        tgt_key_padding_mask_remove_all_true[
            tgt_key_padding_mask.logical_not().sum(dim=-1) == 0, 0] = False

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:  # use positional embedding form initial
                # proposals
                query_pos_embed = query_pos.transpose(0, 1)
            else:
                # recalculate the positional embedding
                query_pos_embed = position_embedding.forward_coordinates(bi)
                query_pos_embed = query_pos_embed.transpose(0, 1)
            query_pos_embed = self.ref_point_head(query_pos_embed)

            if self.detach_support_feat:
                refined_support_feat = refined_support_feat.detach()

            refined_support_feat, attn_map = layer(
                refined_support_feat,
                query_feat,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask_remove_all_true,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos_embed,
                skeleton=skeleton)

            if self.return_intermediate:
                intermediate.append(self.norm(refined_support_feat))

            if return_attn_map:
                attn_maps.append(attn_map)

            # update the query coordinates
            delta_bi = kpt_branch[layer_idx](
                refined_support_feat.transpose(0, 1))

            # Prediction loss
            if self.look_twice:
                bi_pred = self.update(bi_tag, delta_bi)
                bi_tag = self.update(bi, delta_bi)
            else:
                bi_tag = self.update(bi, delta_bi)
                bi_pred = bi_tag

            bi = bi_tag.detach()
            query_points.append(bi_pred)

        if self.norm is not None:
            refined_support_feat = self.norm(refined_support_feat)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(refined_support_feat)

        if self.return_intermediate:
            return torch.stack(intermediate), query_points, attn_maps

        return refined_support_feat.unsqueeze(0), query_points, attn_maps

    def update(self, query_coordinates, delta_unsig):
        query_coordinates_unsigmoid = inverse_sigmoid(query_coordinates)
        new_query_coordinates = query_coordinates_unsigmoid + delta_unsig
        new_query_coordinates = new_query_coordinates.sigmoid()
        return new_query_coordinates


class GraphTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 graph_decoder=None):

        super().__init__()
        self.graph_decoder = graph_decoder
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.choker = nn.Linear(in_features=2 * d_model, out_features=d_model)
        # Implementation of Feedforward model
        if self.graph_decoder is None:
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
        elif self.graph_decoder == 'pre':
            self.ffn1 = GCNLayer(d_model, dim_feedforward, batch_first=False)
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
        elif self.graph_decoder == 'post':
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.ffn2 = GCNLayer(dim_feedforward, d_model, batch_first=False)
        else:
            self.ffn1 = GCNLayer(d_model, dim_feedforward, batch_first=False)
            self.ffn2 = GCNLayer(dim_feedforward, d_model, batch_first=False)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                refined_support_feat,
                refined_query_feat,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                skeleton: Optional[list] = None):

        q = k = self.with_pos_embed(
            refined_support_feat,
            query_pos + pos[refined_query_feat.shape[0]:])
        tgt2 = self.self_attn(
            q,
            k,
            value=refined_support_feat,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]

        refined_support_feat = refined_support_feat + self.dropout1(tgt2)
        refined_support_feat = self.norm1(refined_support_feat)

        # concatenate the positional embedding with the content feature,
        # instead of direct addition
        cross_attn_q = torch.cat(
            (refined_support_feat,
             query_pos + pos[refined_query_feat.shape[0]:]),
            dim=-1)
        cross_attn_k = torch.cat(
            (refined_query_feat, pos[:refined_query_feat.shape[0]]), dim=-1)

        tgt2, attn_map = self.multihead_attn(
            query=cross_attn_q,
            key=cross_attn_k,
            value=refined_query_feat,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)

        refined_support_feat = refined_support_feat + self.dropout2(
            self.choker(tgt2))
        refined_support_feat = self.norm2(refined_support_feat)
        if self.graph_decoder is not None:
            num_pts, b, c = refined_support_feat.shape
            adj = adj_from_skeleton(
                num_pts=num_pts,
                skeleton=skeleton,
                mask=tgt_key_padding_mask,
                device=refined_support_feat.device)
            if self.graph_decoder == 'pre':
                tgt2 = self.ffn2(
                    self.dropout(
                        self.activation(self.ffn1(refined_support_feat, adj))))
            elif self.graph_decoder == 'post':
                tgt2 = self.ffn2(
                    self.dropout(
                        self.activation(self.ffn1(refined_support_feat))), adj)
            else:
                tgt2 = self.ffn2(
                    self.dropout(
                        self.activation(self.ffn1(refined_support_feat, adj))),
                    adj)
        else:
            tgt2 = self.ffn2(
                self.dropout(self.activation(self.ffn1(refined_support_feat))))
        refined_support_feat = refined_support_feat + self.dropout3(tgt2)
        refined_support_feat = self.norm3(refined_support_feat)

        return refined_support_feat, attn_map


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                query,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # src: [hw, bs, c]
        # query: [num_query, bs, c]
        # mask: None by default
        # src_key_padding_mask: [bs, hw]
        # query_key_padding_mask: [bs, nq]
        # pos: [hw, bs, c]

        # organize the input
        # implement the attention mask to mask out the useless points
        n, bs, c = src.shape
        src_cat = torch.cat((src, query), dim=0)  # [hw + nq, bs, c]
        mask_cat = torch.cat((src_key_padding_mask, query_key_padding_mask),
                             dim=1)  # [bs, hw+nq]
        output = src_cat

        for layer in self.layers:
            output = layer(
                output,
                query_length=n,
                src_mask=mask,
                src_key_padding_mask=mask_cat,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        # resplit the output into src and query
        refined_query = output[n:, :, :]  # [nq, bs, c]
        output = output[:n, :, :]  # [n, bs, c]

        return output, refined_query


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                query_length,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = self.with_pos_embed(src, pos)
        q = k = src
        # NOTE: compared with original implementation, we add positional
        # embedding into the VALUE.
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b])
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.cat((adj_mx, adj.unsqueeze(0)), dim=0)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    adj = adj * ~mask[..., None] * ~mask[:, None]
    adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
    return adj


class GCNLayer(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=2,
                 use_bias=True,
                 activation=nn.ReLU(inplace=True),
                 batch_first=True):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_features,
            out_features * kernel_size,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=1,
            bias=use_bias)
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_first = batch_first

    def forward(self, x, adj):
        assert adj.size(1) == self.kernel_size
        if not self.batch_first:
            x = x.permute(1, 2, 0)
        else:
            x = x.transpose(1, 2)
        x = self.conv(x)
        b, kc, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))
        if self.activation is not None:
            x = self.activation(x)
        if not self.batch_first:
            x = x.permute(2, 0, 1)
        else:
            x = x.transpose(1, 2)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
