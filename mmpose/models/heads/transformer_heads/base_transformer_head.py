# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, Tuple

import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (Features, OptConfigType, OptMultiConfig,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead


@MODELS.register_module()
class TransformerHead(BaseHead):
    r"""Implementation of `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    Args:
        encoder (ConfigDict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (ConfigDict, optional): Config of the
            Transformer decoder. Defaults to None.
        out_head (ConfigDict, optional): Config for the
            bounding final out head module. Defaults to None.
        positional_encoding (ConfigDict, optional): Config for
            transformer position encoding. Defaults to None.
        num_queries (int): Number of query in Transformer.
        loss (ConfigDict, optional): Config for loss functions.
            Defaults to None.
        init_cfg (ConfigDict, optional): Config to control the initialization.
    """

    def __init__(self,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 out_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.encoder_cfg = encoder
        self.decoder_cfg = decoder
        self.out_head_cfg = out_head
        self.positional_encoding_cfg = positional_encoding
        self.num_queries = num_queries

    def forward(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList = None) -> Dict:
        """Forward the network."""
        encoder_outputs_dict = self.forward_encoder(feats, batch_data_samples)

        decoder_outputs_dict = self.forward_decoder(**encoder_outputs_dict)

        head_outputs_dict = self.forward_out_head(batch_data_samples,
                                                  **decoder_outputs_dict)
        return head_outputs_dict

    @abstractmethod
    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        """Predict results from features."""
        pass

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        pass

    @abstractmethod
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        pass

    @abstractmethod
    def forward_out_head(self, query: Tensor, query_pos: Tensor,
                         memory: Tensor, **kwargs) -> Dict:
        pass

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
