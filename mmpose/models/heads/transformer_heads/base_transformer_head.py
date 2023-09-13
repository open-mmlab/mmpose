# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, Tuple

from torch import Tensor

from mmpose.models.utils.tta import flip_coordinates
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
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        out_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding final out head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer position encoding. Defaults None
        num_queries (int): Number of query in Transformer.
    """
    _version = 2

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

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        """Predict results from features."""

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords = self.forward(_feats)
            _batch_coords_flip = flip_coordinates(
                self.forward(_feats_flip),
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords = self.forward(feats, batch_data_samples)  # (B, K, D)

        return batch_coords

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
