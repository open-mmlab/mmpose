# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.model import ModuleDict
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.heads.base_head import BaseHead
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, InstanceList,
                                 OptConfigType, OptSampleList, Predictions)

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SKPSHead(BaseHead):
    """DisEntangled Keypoint Regression head introduced in `Bottom-up human
    pose estimation via disentangled keypoint regression`_ by Geng et al
    (2021). The head is composed of a heatmap branch and a displacement branch.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings


    .. _`Bottom-up human pose estimation via disentangled keypoint regression`:
        https://arxiv.org/abs/2104.02300
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 heatmap_loss: ConfigType = dict(
                     type='AdaptiveWingLoss', use_target_weight=True),
                 offside_loss: ConfigType = dict(
                     type='SoftWingLoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=self.out_channels * 3,
                kernel_size=1,
                bias=True)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # build losses
        self.loss_module = ModuleDict(
            dict(
                heatmap=MODELS.build(heatmap_loss),
                displacement=MODELS.build(offside_loss),
            ))

        # build decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal',
                layer=['Conv2d', 'ConvTranspose2d'],
                std=0.001,
                bias=0),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is a tuple of heatmap and displacement.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tuple[Tensor]: output heatmap and displacement.
        """
        x = feats[-1]

        x = self.conv_layers(x)
        x = self.final_layer(x)
        heatmaps = x[:, :self.out_channels, ...]
        offside = x[:, self.out_channels:, ...]
        return heatmaps, offside

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
        pred_heatmaps, pred_offside = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.stack([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_offside = torch.stack(
            [d.gt_fields.displacements for d in batch_data_samples])

        # calculate losses
        losses = dict()
        heatmap_loss = self.loss_module['heatmap'](pred_heatmaps, gt_heatmaps,
                                                   keypoint_weights)

        offside_loss = self._wing_loss(
            pred_offside, gt_offside, weight=gt_heatmaps)

        losses.update({
            'loss/heatmap': heatmap_loss,
            'loss/offside': offside_loss * 0.01,
        })
        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _wing_loss(self, pre, gt, w=10.0, epsilon=2.0, weight=1.):

        x = pre - gt
        c = w * (1.0 - math.log(1.0 + w / epsilon))

        deta = torch.abs(x)

        losses = torch.where(
            torch.gt(deta, w), deta - c, w * torch.log(1.0 + deta / epsilon))

        weight = torch.cat([weight, weight], dim=1)
        losses = losses * weight
        loss = torch.sum(losses) / torch.sum(weight)
        return loss

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-scale features in TTA)
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
        """

        flip_test = test_cfg.get('flip_test', False)
        metainfo = batch_data_samples[0].metainfo

        if flip_test:
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = metainfo['flip_indices']
            _feat, _feat_flip = feats
            _heatmaps, _displacements = self.forward(_feat)
            _heatmaps_flip, _displacements_flip = self.forward(_feat_flip)

            batch_size = _heatmaps.shape[0]

            _heatmaps = to_numpy(_heatmaps)
            _displacements = to_numpy(_displacements)

            _heatmaps_flip = to_numpy(_heatmaps_flip)
            _displacements_flip = to_numpy(_displacements_flip)
            preds = []
            for b in range(batch_size):
                _keypoints, _keypoint_scores = self.decoder.decode(
                    _heatmaps[b], _displacements[b])

                _keypoints_flip, _keypoint_scores_flip = self.decoder.decode(
                    _heatmaps_flip[b], _displacements_flip[b])


                ##flip the kps coords
                N,C,H,W=_heatmaps.shape
                _keypoints_flip /= (W-1)
                _keypoints_flip = flip_coordinates(
                    _keypoints_flip,
                    flip_indices=flip_indices,
                    shift_coords=False,
                    input_size=((W-1), (H-1)))
                _keypoints_flip *= (W-1)

                _keypoints = (_keypoints + _keypoints_flip) / 2.
                # pack outputs
                preds.append(InstanceData(keypoints=_keypoints))
            return preds
            batch_displacements = (_displacements + _displacements_flip) / 2.0

        else:
            batch_heatmaps, batch_displacements = self.forward(feats)

        preds = self.decode(batch_heatmaps, batch_displacements, test_cfg,
                            metainfo)

        if test_cfg.get('output_heatmaps', False):
            heatmaps = [hm.detach() for hm in batch_heatmaps]
            displacements = [dm.detach() for dm in batch_displacements]
            B = heatmaps[0].shape[0]
            pred_fields = []
            for i in range(B):
                pred_fields.append(
                    PixelData(
                        heatmaps=heatmaps[0][i],
                        displacements=displacements[0][i]))
            return preds, pred_fields
        else:
            return preds

    def decode(self,
               heatmaps: Tuple[Tensor],
               offside: Tuple[Tensor],
               test_cfg: ConfigType = {},
               metainfo: dict = {}) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            heatmaps (Tuple[Tensor]): The output heatmaps inferred from one
                image or multi-scale images.
            offside (Tuple[Tensor]): The output displacement fields
                inferred from one image or multi-scale images.
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}
            metainfo (dict): The metainfo of test dataset. Defaults to {}

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
                decoded pose information of the instances of one data sample.
        """

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        preds = []
        batch_size = heatmaps.shape[0]

        heatmaps = to_numpy(heatmaps)
        offside = to_numpy(offside)

        for b in range(batch_size):
            keypoints, keypoint_scores = self.decoder.decode(
                heatmaps[b], offside[b])

            # pack outputs
            preds.append(
                InstanceData(
                    keypoints=keypoints, keypoint_scores=keypoint_scores))

        return preds

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v
