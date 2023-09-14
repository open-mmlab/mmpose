# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model import BaseModule, ModuleList
from torch import Tensor


class FFN(BaseModule):
    """Very simple multi-layer perceptron with relu. Mostly used in DETR series
    detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers..
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.layers = ModuleList()
        self.layers.append(Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.layers.append(Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class PositionEmbeddingSineHW(BaseModule):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images."""

    def __init__(self,
                 num_pos_feats=64,
                 temperatureH=10000,
                 temperatureW=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor):

        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_tx = self.temperatureW**(2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_ty = self.temperatureH**(2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
