# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule, ModuleDict
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead


def smooth_heatmaps(heatmaps: Tensor, blur_kernel_size: int) -> Tensor:
    """Smooth the heatmaps by blurring and averaging.

    Args:
        heatmaps (Tensor): The heatmaps to smooth.
        blur_kernel_size (int): The kernel size for blurring the heatmaps.

    Returns:
        Tensor: The smoothed heatmaps.
    """
    smoothed_heatmaps = torch.nn.functional.avg_pool2d(
        heatmaps, blur_kernel_size, 1, (blur_kernel_size - 1) // 2)
    smoothed_heatmaps = (heatmaps + smoothed_heatmaps) / 2.0
    return smoothed_heatmaps


class TruncSigmoid(nn.Sigmoid):
    """A sigmoid activation function that truncates the output to the given
    range.

    Args:
        min (float, optional): The minimum value to clamp the output to.
            Defaults to 0.0
        max (float, optional): The maximum value to clamp the output to.
            Defaults to 1.0
    """

    def __init__(self, min: float = 0.0, max: float = 1.0):
        super(TruncSigmoid, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input: Tensor) -> Tensor:
        """Computes the truncated sigmoid activation of the input tensor."""
        output = torch.sigmoid(input)
        output = output.clamp(min=self.min, max=self.max)
        return output


class IIAModule(BaseModule):
    """Instance Information Abstraction module introduced in `CID`. This module
    extracts the feature representation vectors for each instance.

    Args:
        in_channels (int): Number of channels in the input feature tensor
        out_channels (int): Number of channels of the output heatmaps
        clamp_delta (float, optional): A small value that prevents the sigmoid
            activation from becoming saturated. Defaults to 1e-4.
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        clamp_delta: float = 1e-4,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.keypoint_root_conv = build_conv_layer(
            dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1))
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)

    def forward(self, feats: Tensor):
        heatmaps = self.keypoint_root_conv(feats)
        heatmaps = self.sigmoid(heatmaps)
        return heatmaps

    def _sample_feats(self, feats: Tensor, indices: Tensor) -> Tensor:
        """Extract feature vectors at the specified indices from the input
        feature map.

        Args:
            feats (Tensor): Input feature map.
            indices (Tensor): Indices of the feature vectors to extract.

        Returns:
            Tensor: Extracted feature vectors.
        """
        assert indices.dtype == torch.long
        if indices.shape[1] == 3:
            b, w, h = [ind.squeeze(-1) for ind in indices.split(1, -1)]
            instance_feats = feats[b, :, h, w]
        elif indices.shape[1] == 2:
            w, h = [ind.squeeze(-1) for ind in indices.split(1, -1)]
            instance_feats = feats[:, :, h, w]
            instance_feats = instance_feats.permute(0, 2, 1)
            instance_feats = instance_feats.reshape(-1,
                                                    instance_feats.shape[-1])

        else:
            raise ValueError(f'`indices` should have 2 or 3 channels, '
                             f'but got f{indices.shape[1]}')
        return instance_feats

    def _hierarchical_pool(self, heatmaps: Tensor) -> Tensor:
        """Conduct max pooling on the input heatmaps with different kernel size
        according to the input size.

        Args:
            heatmaps (Tensor): Input heatmaps.

        Returns:
            Tensor: Result of hierarchical pooling.
        """
        map_size = (heatmaps.shape[-1] + heatmaps.shape[-2]) / 2.0
        if map_size > 300:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 7, 1, 3)
        elif map_size > 200:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 5, 1, 2)
        else:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 3, 1, 1)
        return maxm

    def forward_train(self, feats: Tensor, instance_coords: Tensor,
                      instance_imgids: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass during training.

        Args:
            feats (Tensor): Input feature tensor.
            instance_coords (Tensor): Coordinates of the instance roots.
            instance_imgids (Tensor): Sample indices of each instances
                in the batch.

        Returns:
            Tuple[Tensor, Tensor]: Extracted feature vectors and heatmaps
                for the instances.
        """
        heatmaps = self.forward(feats)
        indices = torch.cat((instance_imgids[:, None], instance_coords), dim=1)
        instance_feats = self._sample_feats(feats, indices)

        return instance_feats, heatmaps

    def forward_test(
        self, feats: Tensor, test_cfg: Dict
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Forward pass during testing.

        Args:
            feats (Tensor): Input feature tensor.
            test_cfg (Dict): Testing configuration, including:
                - blur_kernel_size (int, optional): Kernel size for blurring
                    the heatmaps. Defaults to 3.
                - max_instances (int, optional): Maximum number of instances
                    to extract. Defaults to 30.
                - score_threshold (float, optional): Minimum score for
                    extracting an instance. Defaults to 0.01.
                - flip_test (bool, optional): Whether to compute the average
                    of the heatmaps across the batch dimension.
                    Defaults to False.

        Returns:
            A tuple of Tensor including extracted feature vectors,
            coordinates, and scores of the instances. Any of these can be
            empty Tensor if no instances are extracted.
        """
        blur_kernel_size = test_cfg.get('blur_kernel_size', 3)
        max_instances = test_cfg.get('max_instances', 30)
        score_threshold = test_cfg.get('score_threshold', 0.01)
        H, W = feats.shape[-2:]

        # compute heatmaps
        heatmaps = self.forward(feats).narrow(1, -1, 1)
        if test_cfg.get('flip_test', False):
            heatmaps = heatmaps.mean(dim=0, keepdims=True)
        smoothed_heatmaps = smooth_heatmaps(heatmaps, blur_kernel_size)

        # decode heatmaps
        maximums = self._hierarchical_pool(smoothed_heatmaps)
        maximums = torch.eq(maximums, smoothed_heatmaps).float()
        maximums = (smoothed_heatmaps * maximums).reshape(-1)
        scores, pos_ind = maximums.topk(max_instances, dim=0)
        select_ind = (scores > (score_threshold)).nonzero().squeeze(1)
        scores, pos_ind = scores[select_ind], pos_ind[select_ind]

        # sample feature vectors from feature map
        instance_coords = torch.stack((pos_ind % W, pos_ind // W), dim=1)
        instance_feats = self._sample_feats(feats, instance_coords)

        return instance_feats, instance_coords, scores


class ChannelAttention(nn.Module):
    """Channel-wise attention module introduced in `CID`.

    Args:
        in_channels (int): The number of channels of the input instance
            vectors.
        out_channels (int): The number of channels of the transformed instance
            vectors.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ChannelAttention, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_feats: Tensor, instance_feats: Tensor) -> Tensor:
        """Applies attention to the channel dimension of the input tensor."""

        instance_feats = self.atn(instance_feats).unsqueeze(2).unsqueeze(3)
        return global_feats * instance_feats


class SpatialAttention(nn.Module):
    """Spatial-wise attention module introduced in `CID`.

    Args:
        in_channels (int): The number of channels of the input instance
            vectors.
        out_channels (int): The number of channels of the transformed instance
            vectors.
    """

    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)
        self.feat_stride = 4
        self.conv = nn.Conv2d(3, 1, 5, 1, 2)

    def _get_pixel_coords(self, heatmap_size: Tuple, device: str = 'cpu'):
        """Get pixel coordinates for each element in the heatmap.

        Args:
            heatmap_size (tuple): Size of the heatmap in (W, H) format.
            device (str): Device to put the resulting tensor on.

        Returns:
            Tensor of shape (batch_size, num_pixels, 2) containing the pixel
            coordinates for each element in the heatmap.
        """
        w, h = heatmap_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pixel_coords = torch.stack((x, y), dim=-1).reshape(-1, 2)
        pixel_coords = pixel_coords.float().to(device) + 0.5
        return pixel_coords

    def forward(self, global_feats: Tensor, instance_feats: Tensor,
                instance_coords: Tensor) -> Tensor:
        """Perform spatial attention.

        Args:
            global_feats (Tensor): Tensor containing the global features.
            instance_feats (Tensor): Tensor containing the instance feature
                vectors.
            instance_coords (Tensor): Tensor containing the root coordinates
                of the instances.

        Returns:
            Tensor containing the modulated global features.
        """
        B, C, H, W = global_feats.size()

        instance_feats = self.atn(instance_feats).reshape(B, C, 1, 1)
        feats = global_feats * instance_feats.expand_as(global_feats)
        fsum = torch.sum(feats, dim=1, keepdim=True)

        pixel_coords = self._get_pixel_coords((W, H), feats.device)
        relative_coords = instance_coords.reshape(
            -1, 1, 2) - pixel_coords.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1) / 32.0
        relative_coords = relative_coords.reshape(B, 2, H, W)

        input_feats = torch.cat((fsum, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_feats * mask


class GFDModule(BaseModule):
    """Global Feature Decoupling module introduced in `CID`. This module
    extracts the decoupled heatmaps for each instance.

    Args:
        in_channels (int): Number of channels in the input feature map
        out_channels (int): Number of channels of the output heatmaps
            for each instance
        gfd_channels (int): Number of channels in the transformed feature map
        clamp_delta (float, optional): A small value that prevents the sigmoid
            activation from becoming saturated. Defaults to 1e-4.
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gfd_channels: int,
        clamp_delta: float = 1e-4,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.conv_down = build_conv_layer(
            dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=gfd_channels,
                kernel_size=1))

        self.channel_attention = ChannelAttention(in_channels, gfd_channels)
        self.spatial_attention = SpatialAttention(in_channels, gfd_channels)
        self.fuse_attention = build_conv_layer(
            dict(
                type='Conv2d',
                in_channels=gfd_channels * 2,
                out_channels=gfd_channels,
                kernel_size=1))
        self.heatmap_conv = build_conv_layer(
            dict(
                type='Conv2d',
                in_channels=gfd_channels,
                out_channels=out_channels,
                kernel_size=1))
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)

    def forward(
        self,
        feats: Tensor,
        instance_feats: Tensor,
        instance_coords: Tensor,
        instance_imgids: Tensor,
    ) -> Tensor:
        """Extract decoupled heatmaps for each instance.

        Args:
            feats (Tensor): Input feature maps.
            instance_feats (Tensor): Tensor containing the instance feature
                vectors.
            instance_coords (Tensor): Tensor containing the root coordinates
                of the instances.
            instance_imgids (Tensor): Sample indices of each instances
                in the batch.

        Returns:
            A tensor containing decoupled heatmaps.
        """

        global_feats = self.conv_down(feats)
        global_feats = global_feats[instance_imgids]
        cond_instance_feats = torch.cat(
            (self.channel_attention(global_feats, instance_feats),
             self.spatial_attention(global_feats, instance_feats,
                                    instance_coords)),
            dim=1)

        cond_instance_feats = self.fuse_attention(cond_instance_feats)
        cond_instance_feats = torch.nn.functional.relu(cond_instance_feats)
        cond_instance_feats = self.heatmap_conv(cond_instance_feats)
        heatmaps = self.sigmoid(cond_instance_feats)

        return heatmaps


@MODELS.register_module()
class CIDHead(BaseHead):
    """Contextual Instance Decoupling head introduced in `Contextual Instance
    Decoupling for Robust Multi-Person Pose Estimation (CID)`_ by Wang et al
    (2022). The head is composed of an Instance Information Abstraction (IIA)
    module and a Global Feature Decoupling (GFD) module.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        num_keypoints (int): Number of keypoints
        gfd_channels (int): Number of filters in GFD module
        max_train_instances (int): Maximum number of instances in a batch
            during training. Defaults to 200
        heatmap_loss (Config): Config of the heatmap loss. Defaults to use
            :class:`KeypointMSELoss`
        coupled_heatmap_loss (Config): Config of the loss for coupled heatmaps.
            Defaults to use :class:`SoftWeightSmoothL1Loss`
        decoupled_heatmap_loss (Config): Config of the loss for decoupled
            heatmaps. Defaults to use :class:`SoftWeightSmoothL1Loss`
        contrastive_loss (Config): Config of the contrastive loss for
            representation vectors of instances. Defaults to use
            :class:`InfoNCELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`CID`: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_
    Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_
    CVPR_2022_paper.html
    """
    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 gfd_channels: int,
                 num_keypoints: int,
                 prior_prob: float = 0.01,
                 coupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 decoupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 contrastive_loss: OptConfigType = dict(type='InfoNCELoss'),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # build sub-modules
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.iia_module = IIAModule(
            in_channels,
            num_keypoints + 1,
            init_cfg=init_cfg + [
                dict(
                    type='Normal',
                    layer=['Conv2d', 'Linear'],
                    std=0.001,
                    override=dict(
                        name='keypoint_root_conv',
                        type='Normal',
                        std=0.001,
                        bias=bias_value))
            ])
        self.gfd_module = GFDModule(
            in_channels,
            num_keypoints,
            gfd_channels,
            init_cfg=init_cfg + [
                dict(
                    type='Normal',
                    layer=['Conv2d', 'Linear'],
                    std=0.001,
                    override=dict(
                        name='heatmap_conv',
                        type='Normal',
                        std=0.001,
                        bias=bias_value))
            ])

        # build losses
        self.loss_module = ModuleDict(
            dict(
                heatmap_coupled=MODELS.build(coupled_heatmap_loss),
                heatmap_decoupled=MODELS.build(decoupled_heatmap_loss),
                contrastive=MODELS.build(contrastive_loss),
            ))

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d', 'Linear'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        feats = feats[-1]
        instance_info = self.iia_module.forward_test(feats, {})
        instance_feats, instance_coords, instance_scores = instance_info
        instance_imgids = torch.zeros(
            instance_coords.size(0), dtype=torch.long, device=feats.device)
        instance_heatmaps = self.gfd_module(feats, instance_feats,
                                            instance_coords, instance_imgids)

        return instance_heatmaps

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
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
        metainfo = batch_data_samples[0].metainfo

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2

            feats_flipped = flip_heatmaps(feats[1][-1], shift_heatmap=False)
            feats = torch.cat((feats[0][-1], feats_flipped))
        else:
            feats = feats[-1]

        instance_info = self.iia_module.forward_test(feats, test_cfg)
        instance_feats, instance_coords, instance_scores = instance_info
        if len(instance_coords) > 0:
            instance_imgids = torch.zeros(
                instance_coords.size(0), dtype=torch.long, device=feats.device)
            if test_cfg.get('flip_test', False):
                instance_coords = torch.cat((instance_coords, instance_coords))
                instance_imgids = torch.cat(
                    (instance_imgids, instance_imgids + 1))
            instance_heatmaps = self.gfd_module(feats, instance_feats,
                                                instance_coords,
                                                instance_imgids)
            if test_cfg.get('flip_test', False):
                flip_indices = batch_data_samples[0].metainfo['flip_indices']
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(
                    instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = \
                    instance_heatmaps_flip[:, flip_indices, :, :]
                instance_heatmaps = (instance_heatmaps +
                                     instance_heatmaps_flip) / 2.0
            instance_heatmaps = smooth_heatmaps(
                instance_heatmaps, test_cfg.get('blur_kernel_size', 3))

            preds = self.decode((instance_heatmaps, instance_scores[:, None]))
            preds = InstanceData.cat(preds)
            preds.keypoints[..., 0] += metainfo['input_size'][
                0] / instance_heatmaps.shape[-1] / 2.0
            preds.keypoints[..., 1] += metainfo['input_size'][
                1] / instance_heatmaps.shape[-2] / 2.0
            preds = [preds]

        else:
            preds = [
                InstanceData(
                    keypoints=np.empty((0, self.num_keypoints, 2)),
                    keypoint_scores=np.empty((0, self.num_keypoints)))
            ]
            instance_heatmaps = torch.empty(0, self.num_keypoints,
                                            *feats.shape[-2:])

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(
                    heatmaps=instance_heatmaps.reshape(
                        -1, *instance_heatmaps.shape[-2:]))
            ]
            return preds, pred_fields
        else:
            return preds

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

        # load targets
        gt_heatmaps, gt_instance_coords, keypoint_weights = [], [], []
        heatmap_mask = []
        instance_imgids, gt_instance_heatmaps = [], []
        for i, d in enumerate(batch_data_samples):
            gt_heatmaps.append(d.gt_fields.heatmaps)
            gt_instance_coords.append(d.gt_instance_labels.instance_coords)
            keypoint_weights.append(d.gt_instance_labels.keypoint_weights)
            instance_imgids.append(
                torch.ones(
                    len(d.gt_instance_labels.instance_coords),
                    dtype=torch.long) * i)

            instance_heatmaps = d.gt_fields.instance_heatmaps.reshape(
                -1, self.num_keypoints,
                *d.gt_fields.instance_heatmaps.shape[1:])
            gt_instance_heatmaps.append(instance_heatmaps)

            if 'heatmap_mask' in d.gt_fields:
                heatmap_mask.append(d.gt_fields.heatmap_mask)

        gt_heatmaps = torch.stack(gt_heatmaps)
        heatmap_mask = torch.stack(heatmap_mask) if heatmap_mask else None

        gt_instance_coords = torch.cat(gt_instance_coords, dim=0)
        gt_instance_heatmaps = torch.cat(gt_instance_heatmaps, dim=0)
        keypoint_weights = torch.cat(keypoint_weights, dim=0)
        instance_imgids = torch.cat(instance_imgids).to(gt_heatmaps.device)

        # feed-forward
        feats = feats[-1]
        pred_instance_feats, pred_heatmaps = self.iia_module.forward_train(
            feats, gt_instance_coords, instance_imgids)

        # conpute contrastive loss
        contrastive_loss = 0
        for i in range(len(batch_data_samples)):
            pred_instance_feat = pred_instance_feats[instance_imgids == i]
            contrastive_loss += self.loss_module['contrastive'](
                pred_instance_feat)
        contrastive_loss = contrastive_loss / max(1, len(instance_imgids))

        # limit the number of instances
        max_train_instances = train_cfg.get('max_train_instances', -1)
        if (max_train_instances > 0
                and len(instance_imgids) > max_train_instances):
            selected_indices = torch.randperm(
                len(instance_imgids),
                device=gt_heatmaps.device,
                dtype=torch.long)[:max_train_instances]
            gt_instance_coords = gt_instance_coords[selected_indices]
            keypoint_weights = keypoint_weights[selected_indices]
            gt_instance_heatmaps = gt_instance_heatmaps[selected_indices]
            instance_imgids = instance_imgids[selected_indices]
            pred_instance_feats = pred_instance_feats[selected_indices]

        # calculate the decoupled heatmaps for each instance
        pred_instance_heatmaps = self.gfd_module(feats, pred_instance_feats,
                                                 gt_instance_coords,
                                                 instance_imgids)

        # calculate losses
        losses = {
            'loss/heatmap_coupled':
            self.loss_module['heatmap_coupled'](pred_heatmaps, gt_heatmaps,
                                                None, heatmap_mask)
        }
        if len(instance_imgids) > 0:
            losses.update({
                'loss/heatmap_decoupled':
                self.loss_module['heatmap_decoupled'](pred_instance_heatmaps,
                                                      gt_instance_heatmaps,
                                                      keypoint_weights),
                'loss/contrastive':
                contrastive_loss
            })

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`CIDHead` (before MMPose v1.0.0) to a compatible format
        of :class:`CIDHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_center_conv' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_center_conv',
                              'iia_module.keypoint_root_conv')
                state_dict[k] = v

            if 'conv_down' in k:
                v = state_dict.pop(k)
                k = k.replace('conv_down', 'gfd_module.conv_down')
                state_dict[k] = v

            if 'c_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('c_attn', 'gfd_module.channel_attention')
                state_dict[k] = v

            if 's_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('s_attn', 'gfd_module.spatial_attention')
                state_dict[k] = v

            if 'fuse_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('fuse_attn', 'gfd_module.fuse_attention')
                state_dict[k] = v

            if 'heatmap_conv' in k:
                v = state_dict.pop(k)
                k = k.replace('heatmap_conv', 'gfd_module.heatmap_conv')
                state_dict[k] = v
