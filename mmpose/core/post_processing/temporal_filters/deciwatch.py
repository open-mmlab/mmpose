import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from mmcv.runner import load_checkpoint

import numpy as np

import math
from .builder import FILTERS
from .filter import TemporalFilter

class PositionEmbeddingSine_1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=True,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, B, L):

        position = torch.arange(0, L, dtype=torch.float32).unsqueeze(0)
        position = position.repeat(B, 1)

        if self.normalize:
            eps = 1e-6
            position = position / (position[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature**(
            2 * (torch.div(dim_t, 1, rounding_mode='trunc')) /
            self.num_pos_feats)

        pe = torch.zeros(B, L, self.num_pos_feats * 2)
        pe[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        pe[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)

        pe = pe.permute(1, 0, 2)

        return pe

@FILTERS.register_module(name=['DeciWatchLoader', 'deciwatchloader'])
class DeciWatchLoader(TemporalFilter):
    """Apply DeciWatch framework for 10x efficiency.

    "DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation",
    arXiv'2022. More details can be found in the `paper
    <https://arxiv.org/pdf/2203.08713>`__ .

    Args:
        input_dim (int): The size of input spatial dimenison, 
            e.g., 15*2 for 2d pose on the jhmdb dataset
        sample_interval (int): DeciWatch argument. See :class:`DeciWatch`
            for details. The intervals of the uniform sampling. 
            The sampling ratio is: 1/sample_interval. Default: 10
        encoder_hidden_dim (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Hidden dimension in the encoder. Default: 64
        decoder_hidden_dim (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Hidden dimension in the decoder. Default: 64
        dropout (float): DeciWatch argument. See :class:`DeciWatch`
            for details. dropout probability. Default: 0.1
        nheads (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Default: 4
        dim_feedforward (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Dimension of feed forward layers.
        enc_layers (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Layers of the encoder. Default: 5
        dec_layers (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Layers of the encoder. Default: 5
        activation (str): DeciWatch argument. See :class:`DeciWatch`
            for details. Activation function in deciwatch.
            Default: 'leaky_relu'
        pre_norm (bool): DeciWatch argument. See :class:`DeciWatch`
            for details. Whether to normalize before positional embedding.
            Default: False
        norm (str): If '2d', use width and height of the image to
            normalize the pixel-wise position into [-1, 1]; 
            If '3d', normalize into relative poses.
        root_index (int, optional): If not None, relative keypoint coordinates
            will be calculated as the SmoothNet input, by centering the
            keypoints around the root point. The model output will be
            converted back to absolute coordinates. Default: None
        width (int): image width
        height (int): image height
        device (str): Device for model inference. Default: 'cpu'
        checkpoint (str, optional): The checkpoint file of the pretrained DeciWatch
            model. Please note that `checkpoint` should be matched with
            DeciWatch argument, like `input_dim`, `hidden_dim`.
    """

    def __init__(
        self,
        input_dim: int,
        sample_interval: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        activation: str,
        pre_norm: bool,
        norm: str = '3d',
        root_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        device: str = 'cpu',
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.norm = norm
        self.root_index = root_index
        self.width = width
        self.height = height 
        self.deciwatch = DeciWatch(
                            input_dim,
                            sample_interval,
                            encoder_hidden_dim,
                            decoder_hidden_dim,
                            dropout,
                            nheads,
                            dim_feedforward, 
                            enc_layers,
                            dec_layers,
                            activation,
                            pre_norm)
        if checkpoint:
            load_checkpoint(self.deciwatch, checkpoint)
        self.deciwatch.to(device)
        self.deciwatch.eval()

        for p in self.deciwatch.parameters():
            p.requires_grad_(False)

    def __call__(self, x: np.ndarray):
        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        root_index = self.root_index
        if self.norm == '3d':
            assert x.shape[-1] == 2
            x_root = x[:, root_index:root_index + 1]
            x = np.delete(x, root_index, axis=1)
            x = x - x_root
        else: #2d
            assert x.shape[-1] == 2
            x = x / self.width * 2 - [1, self.height / self.width]

        # input window size is T = interval*Q, Q>1 (int), only need to estimate poses
        # with Q uniformly sampled frames to boost [interval] times efficiency.
        # input is like [1,0,0,...,0,0,1,0,0,...,0,0,1] for each axis, 
        # 1 is visible frame, 0 is invisible frame
        T, K, C = x.shape

        if T < self.window_size:
            # Skip process the sequence if the input length is less than the interval size
            output = x
        else:
            dtype = x.dtype

            # Convert to tensor and forward the model
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                x = x.view(1, T, K * C)  # to [1, T, KC]
                output = self.deciwatch(x)  # out shape [1, T, KC]

            # Convert model output back to input shape and format
            output = output.view(T, K, C)  # to [T, K, C]
            output = output.cpu().numpy().astype(dtype)  # to numpy.ndarray

        if self.norm == '3d':
            output += x_root
            output = np.concatenate(
                (output[:, :root_index], x_root, output[:, root_index:]),
                axis=1)
        else: #2d, denorm into pixel unit
            output = (output + [1, self.height / self.width]) * self.width / 2
        return output

class DeciWatch(nn.Module):
    """Apply DeciWatch framework for 10x efficiency.

    "DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation",
    arXiv'2022. More details can be found in the `paper
    <https://arxiv.org/pdf/2203.08713>`__ .

    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)

    Args:
        input_dim (int): The size of input spatial dimenison, 
            e.g., 15*2 for 2d pose on the jhmdb dataset
        sample_interval (int): The intervals of the uniform sampling. 
            The sampling ratio is: 1/sample_interval. Default: 10
            choices = [1, 5, 10, 15]
        encoder_hidden_dim (int): Hidden dimension in the encoder. Default: 64
        decoder_hidden_dim (int): Hidden dimension in the decoder. Default: 64
        dropout (float): Dropout probability in Deciwatch. Default: 0.1
        nheads (int): multi-head number in transformer architecture. Default: 4
        dim_feedforward (int): Dimension of feed forward layers. Default: 256
        enc_layers (int): Layers of the encoder. Default: 5
        dec_layers (int): Layers of the encoder. Default: 5
        activation (str): Activation function in deciwatch.
            Default: 'leaky_relu'
        pre_norm (bool): Whether to normalize before positional embedding.
            Default: False

    Shape:
        Input: (N, T, C) the original [noisy & sparse] pose sequence
        Output: (N, T, C) the output [denoise & dense] pose sequence
    """
    def __init__(self,
                 input_dim,
                 sample_interval,
                 encoder_hidden_dim,
                 decoder_hidden_dim,
                 dropout=0.1,
                 nheads=4,
                 dim_feedforward=256, 
                 enc_layers=5,
                 dec_layers=5,
                 activation="leaky_relu",
                 pre_norm=True):
        super(DeciWatch, self).__init__()
        self.pos_embed_dim = encoder_hidden_dim
        self.pos_embed = self.build_position_encoding(self.pos_embed_dim)

        self.sample_interval = sample_interval

        self.deciwatch_par = {
            "input_dim": input_dim,
            "encoder_hidden_dim": encoder_hidden_dim,
            "decoder_hidden_dim": decoder_hidden_dim,
            "dropout": dropout,
            "nheads": nheads,
            "dim_feedforward": dim_feedforward,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "activation": activation,
            "pre_norm": pre_norm
        }

        self.transformer = build_model(self.deciwatch_par)

    def build_position_encoding(self, pos_embed_dim):
        N_steps = pos_embed_dim // 2
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True)
        return position_embedding

    def generate_unifrom_mask(self, L, sample_interval=10):
        # 1 invisible, 0 visible

        seq_len = L
        if (seq_len - 1) % sample_interval != 0:
            raise Exception(
                "The following equation should be satisfied: [Window size] = [sample interval] * Q + 1, where Q is an integer."
            )

        sample_mask = np.ones(seq_len, dtype=np.int32)
        sample_mask[::sample_interval] = 0

        encoder_mask = sample_mask
        decoder_mask = np.array([0] * L, dtype=np.int32)

        return torch.tensor(encoder_mask).cuda(), torch.tensor(
            decoder_mask).cuda()

    def seqence_interpolation(self, motion, rate):

        seq_len = motion.shape[-1]
        indice = torch.arange(seq_len, dtype=int)
        chunk = torch.div(indice, rate, rounding_mode='trunc')
        remain = indice % rate

        prev = motion[:, :, chunk * rate]

        next = torch.cat([
            motion[:, :, (chunk[:-1] + 1) * rate], motion[:, :, -1, np.newaxis]
        ], -1)
        remain = remain.cuda()

        interpolate = (prev / rate * (rate - remain)) + (next / rate * remain)

        return interpolate

    def forward(self, sequence, device):
        N, T, C = sequence.shape
        seq = sequence.permute(0, 2, 1)  # B,C,L

        encoder_mask, decoder_mask = self.generate_unifrom_mask(
            T, sample_interval=self.sample_interval)

        self.input_seq = seq * (1 - encoder_mask.int())
        self.input_seq_interp = self.seqence_interpolation(
            self.input_seq, self.sample_interval)
        self.encoder_mask = encoder_mask.unsqueeze(0).repeat(N, 1).to(device)
        self.decoder_mask = decoder_mask.unsqueeze(0).repeat(N, 1).to(device)

        self.encoder_pos_embed = self.pos_embed(N, T).to(device)
        self.decoder_pos_embed = self.encoder_pos_embed.clone().to(device)

        self.recover, self.denoise = self.transformer.forward(
            input_seq=self.input_seq.to(torch.float32),
            encoder_mask=self.encoder_mask,
            encoder_pos_embed=self.encoder_pos_embed,
            input_seq_interp=self.input_seq_interp,
            decoder_mask=self.decoder_mask,
            decoder_pos_embed=self.decoder_pos_embed,
            sample_interval=self.sample_interval,
            device=device)

        self.recover = self.recover.permute(1, 0, 2).reshape(N, T, C)
        self.denoise = self.denoise.permute(1, 0, 2).reshape(N, T, C)

        return self.recover, self.denoise


class Transformer(nn.Module):

    def __init__(self, 
                 input_nc, 
                 encoder_hidden_dim=64, 
                 decoder_hidden_dim=64,
                 nhead=4, 
                 num_encoder_layers=5,
                 num_decoder_layers=5, 
                 dim_feedforward=256, 
                 dropout=0.1,
                 activation="leaky_relu", 
                 pre_norm=False):
        super(Transformer, self).__init__()

        self.joints_dim = input_nc
        # bring in semantic (5 frames) temporal information into tokens
        self.decoder_embed = nn.Conv1d(self.joints_dim,
                                       decoder_hidden_dim,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.encoder_embed = nn.Linear(self.joints_dim, encoder_hidden_dim)

        encoder_layer = TransformerEncoderLayer(encoder_hidden_dim, nhead,
                                                dim_feedforward, dropout,
                                                activation, pre_norm)
        encoder_norm = nn.LayerNorm(encoder_hidden_dim) if pre_norm else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                        encoder_norm)

        decoder_layer = TransformerDecoderLayer(decoder_hidden_dim, nhead,
                                            dim_feedforward, dropout,
                                            activation, pre_norm)
        decoder_norm = nn.LayerNorm(decoder_hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, 
                                        num_decoder_layers, decoder_norm)

        self.decoder_joints_embed = nn.Linear(decoder_hidden_dim,
                                            self.joints_dim)
        self.encoder_joints_embed = nn.Linear(encoder_hidden_dim,
                                            self.joints_dim)

        # reset parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.nhead = nhead

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def interpolate_embedding(self, input, rate):

        tmp = input.clone()
        seq_len = input.shape[0]
        indice = torch.arange(seq_len, dtype=int).to(self.device)
        chunk = torch.div(indice, rate, rounding_mode='trunc')
        remain = indice % rate

        prev = tmp[chunk * rate]

        next = torch.cat([tmp[(chunk[:-1] + 1) * rate], tmp[-1].unsqueeze(0)],
                         dim=0)

        interpolate = (prev / rate * (rate - remain.view(-1, 1, 1))) + (
            next / rate * remain.view(-1, 1, 1))

        return interpolate

    def forward(self, input_seq, encoder_mask, encoder_pos_embed,
                input_seq_interp, decoder_mask, decoder_pos_embed,
                sample_interval, device):

        self.device = device

        # flatten NxCxL to LxNxC
        input_seq = input_seq.permute(2, 0, 1)
        input_seq_interp = input_seq_interp.permute(2, 0, 1)

        input = input_seq.clone()

        # mask on all sequences:
        trans_src = self.encoder_embed(input_seq)
        mem = self.encode(trans_src, encoder_mask,
                          encoder_pos_embed)  #[l,n,hid]
        reco = self.encoder_joints_embed(mem) + input

        interp = self.interpolate_embedding(reco, sample_interval)
        center = interp.clone()
        trans_tgt = self.decoder_embed(interp.permute(1, 2,
                                                      0)).permute(2, 0, 1)

        output = self.decode(mem, encoder_mask, encoder_pos_embed, trans_tgt,
                             decoder_mask, decoder_pos_embed)

        joints = self.decoder_joints_embed(output) + center
        return joints, reco

    def encode(self, src, src_mask, pos_embed):

        mask = torch.eye(src.shape[0]).bool().cuda()
        memory = self.encoder(src,
                              mask=mask,
                              src_key_padding_mask=src_mask,
                              pos=pos_embed)

        return memory

    def decode(self, memory, memory_mask, memory_pos, tgt, tgt_mask, tgt_pos):
        hs = self.decoder(tgt,
                          memory,
                          tgt_key_padding_mask=tgt_mask,
                          memory_key_padding_mask=memory_mask,
                          pos=memory_pos,
                          query_pos=tgt_pos)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                encoder_hidden_dim, 
                nhead, 
                dim_feedforward=256, 
                dropout=0.1,
                activation="leaky_relu", 
                pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(encoder_hidden_dim,
                                               nhead,
                                               dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(encoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, encoder_hidden_dim)

        self.norm1 = nn.LayerNorm(encoder_hidden_dim)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q,
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

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)  #todo. linear
        src2 = self.self_attn(q,
                              k,
                              value=src2,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.pre_norm:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 decoder_hidden_dim,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="leaky_relu",
                 pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(decoder_hidden_dim,
                                               nhead,
                                               dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(decoder_hidden_dim,
                                                    nhead,
                                                    dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(decoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, decoder_hidden_dim)

        self.norm1 = nn.LayerNorm(decoder_hidden_dim)
        self.norm2 = nn.LayerNorm(decoder_hidden_dim)
        self.norm3 = nn.LayerNorm(decoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.pre_norm:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_model(args):
    return Transformer(
        input_nc=args["input_dim"],
        decoder_hidden_dim=args["decoder_hidden_dim"],
        encoder_hidden_dim=args["encoder_hidden_dim"],
        dropout=args["dropout"],
        nhead=args["nheads"],
        dim_feedforward=args["dim_feedforward"],
        num_encoder_layers=args["enc_layers"],
        num_decoder_layers=args["dec_layers"],
        activation=args["activation"],
        pre_norm=args["pre_norm"],
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
