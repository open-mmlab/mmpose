# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.structures import PixelData
from mmengine.utils import is_list_of
from torch import Tensor

from mmpose.models.utils.tta import aggregate_heatmaps, flip_heatmaps
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from .heatmap_head import HeatmapHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class AssociativeEmbeddingHead(HeatmapHead):

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_keypoints: int,
                 tag_dim: int = 1,
                 tag_per_keypoint: bool = True,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 keypoint_loss: ConfigType = dict(type='KeypointMSELoss'),
                 tag_loss: ConfigType = dict(type='AssociativeEmbeddingLoss'),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if tag_per_keypoint:
            out_channels = num_keypoints * (1 + tag_dim)
        else:
            out_channels = num_keypoints + tag_dim

        loss = dict(
            type='CombinedLoss',
            losses=dict(keypoint_loss=keypoint_loss, tag_loss=tag_loss))

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer=final_layer,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg)

        self.num_keypoints = num_keypoints
        self.tag_dim = tag_dim
        self.tag_per_keypoint = tag_per_keypoint

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Features): The features which could be in following forms:

                - Tuple[Tensor]: multi-stage features from the backbone
                - List[Tuple[Tensor]]: multiple features for TTA where either
                    `flip_test` or `multiscale_test` is applied
                - List[List[Tuple[Tensor]]]: multiple features for TTA where
                    both `flip_test` and `multiscale_test` are applied

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

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        # test configs
        multiscale_test = test_cfg.get('multiscale_test', False)
        flip_test = test_cfg.get('flip_test', False)
        shift_heatmap = test_cfg.get('shift_heatmap', False)
        align_corners = test_cfg.get('align_corners', False)
        restore_heatmap_size = test_cfg.get('restore_heatmap_size', False)
        output_heatmaps = test_cfg.get('output_heatmaps', False)

        # enable multi-scale test
        if multiscale_test:
            # TTA: multi-scale test
            assert is_list_of(feats, list if flip_test else tuple)
        else:
            assert is_list_of(feats, tuple if flip_test else Tensor)
            feats = [feats]

        # resize heatmaps to align with with input size
        if restore_heatmap_size:
            img_shape = batch_data_samples[0].metainfo['img_shape']
            assert all(d.metainfo['img_shape'] == img_shape
                       for d in batch_data_samples)
            img_h, img_w = img_shape
            heatmap_size = (img_w, img_h)
        else:
            heatmap_size = None

        multiscale_heatmaps = []
        multiscale_tags = []

        for scale_idx, _feats in enumerate(feats):
            if not flip_test:
                _heatmaps, _tags = self.forward(_feats)

            else:
                # TTA: flip test
                assert isinstance(_feats, list) and len(_feats) == 2
                flip_indices = batch_data_samples[0].metainfo['flip_indices']
                # original
                _feats_orig, _feats_flip = _feats
                _heatmaps_orig, _tags_orig = self.forward(_feats_orig)

                # flipped
                _heatmaps_flip, _tags_flip = self.forward(_feats_flip)
                _heatmaps_flip = flip_heatmaps(
                    _heatmaps_flip,
                    flip_mode='heatmap',
                    flip_indices=flip_indices,
                    shift_heatmap=shift_heatmap)
                _tags_flip = self._flip_tags(
                    _tags_flip,
                    flip_indices=flip_indices,
                    shift_heatmap=shift_heatmap)

                # aggregated heatmaps
                _heatmaps = aggregate_heatmaps(
                    [_heatmaps_orig, _heatmaps_flip],
                    size=heatmap_size,
                    align_corners=align_corners,
                    mode='average')

                # aggregated tags (only at original scale)
                if scale_idx == 0:
                    _tags = aggregate_heatmaps([_tags_orig, _tags_flip],
                                               size=heatmap_size,
                                               align_corners=align_corners,
                                               mode='concat')
                else:
                    _tags = None

            multiscale_heatmaps.append(_heatmaps)
            multiscale_tags.append(_tags)

        # aggregate multi-scale heatmaps
        if len(feats) > 1:
            batch_heatmaps = aggregate_heatmaps(
                multiscale_heatmaps,
                align_corners=align_corners,
                mode='average')
        else:
            batch_heatmaps = multiscale_heatmaps[0]
        # only keep tags at original scale
        batch_tags = multiscale_tags[0]

        batch_outputs = tuple([batch_heatmaps, batch_tags])
        preds = self.decode(batch_outputs)

        if output_heatmaps:
            pred_fields = []
            for _heatmaps, _tags in zip(batch_heatmaps.detach(),
                                        batch_tags.detach()):
                pred_fields.append(PixelData(heatmaps=_heatmaps, tags=_tags))

            return preds, pred_fields
        else:
            return preds

    def _flip_tags(self,
                   tags: Tensor,
                   flip_indices: List[int],
                   shift_heatmap: bool = True):
        """Flip the tagging heatmaps horizontally for test-time augmentation.

        Args:
            tags (Tensor): batched tagging heatmaps to flip
            flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
            shift_heatmap (bool): Shift the flipped heatmaps to align with the
            original heatmaps and improve accuracy. Defaults to ``True``

        Returns:
            Tensor: flipped tagging heatmaps
        """
        B, C, H, W = tags.shape
        K = self.num_keypoints
        L = self.tag_dim

        tags = tags.flip(-1)

        if self.tag_per_keypoint:
            assert C == K * L
            tags = tags.view(B, L, K, H, W)
            tags = tags[:, :, flip_indices]
            tags = tags.view(B, C, H, W)

        if shift_heatmap:
            tags[..., 1:] = tags[..., :-1].clone()

        return tags

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmaps and tags.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            tuple:
            - heatmaps (Tensor): output heatmaps
            - tags (Tensor): output tags
        """

        output = super().forward(feats)
        heatmaps = output[:, :self.num_keypoints]
        tags = output[:, self.num_keypoints:]
        return heatmaps, tags

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
        pred_heatmaps, pred_tags = self.forward(feats)

        if not self.tag_per_keypoint:
            pred_tags = pred_tags.repeat((1, self.num_keypoints, 1, 1))

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_masks = torch.stack(
            [d.gt_fields.heatmap_mask for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        keypoint_indices = [
            d.gt_instance_labels.keypoint_indices for d in batch_data_samples
        ]

        loss_kpt = self.loss_module.keypoint_loss(pred_heatmaps, gt_heatmaps,
                                                  keypoint_weights, gt_masks)

        loss_pull, loss_push = self.loss_module.tag_loss(
            pred_tags, keypoint_indices)

        losses = {
            'loss_kpt': loss_kpt,
            'loss_pull': loss_pull,
            'loss_push': loss_push
        }

        return losses
