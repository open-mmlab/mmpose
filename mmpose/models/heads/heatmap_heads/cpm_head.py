# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (Features, MultiConfig, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class CPMHead(BaseHead):
    """Multi-stage heatmap head introduced in `Convolutional Pose Machines`_ by
    Wei et al (2016) and used by `Stacked Hourglass Networks`_ by Newell et al
    (2016). The head consists of multiple branches, each of which has some
    deconv layers and a simple conv2d layer.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature maps.
        out_channels (int): Number of channels in the output heatmaps.
        num_stages (int): Number of stages.
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.
            Defaults to ``(4, 4, 4)``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config | List[Config]): Config of the keypoint loss of different
            stages. Defaults to use :class:`KeypointMSELoss`.
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Convolutional Pose Machines`: https://arxiv.org/abs/1602.00134
    .. _`Stacked Hourglass Networks`: https://arxiv.org/abs/1603.06937
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 num_stages: int,
                 deconv_out_channels: OptIntSeq = None,
                 deconv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: MultiConfig = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(init_cfg)

        self.num_stages = num_stages
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(loss, list):
            if len(loss) != num_stages:
                raise ValueError(
                    f'The length of loss_module({len(loss)}) did not match '
                    f'`num_stages`({num_stages})')
            self.loss_module = nn.ModuleList(
                MODELS.build(_loss) for _loss in loss)
        else:
            self.loss_module = MODELS.build(loss)

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # build multi-stage deconv layers
        self.multi_deconv_layers = nn.ModuleList([])
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            for _ in range(self.num_stages):
                deconv_layers = self._make_deconv_layers(
                    in_channels=in_channels,
                    layer_out_channels=deconv_out_channels,
                    layer_kernel_sizes=deconv_kernel_sizes,
                )
                self.multi_deconv_layers.append(deconv_layers)
            in_channels = deconv_out_channels[-1]
        else:
            for _ in range(self.num_stages):
                self.multi_deconv_layers.append(nn.Identity())

        # build multi-stage final layers
        self.multi_final_layers = nn.ModuleList([])
        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            for _ in range(self.num_stages):
                self.multi_final_layers.append(build_conv_layer(cfg))
        else:
            for _ in range(self.num_stages):
                self.multi_final_layers.append(nn.Identity())

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, feats: Sequence[Tensor]) -> List[Tensor]:
        """Forward the network. The input is multi-stage feature maps and the
        output is a list of heatmaps from multiple stages.

        Args:
            feats (Sequence[Tensor]): Multi-stage feature maps.

        Returns:
            List[Tensor]: A list of output heatmaps from multiple stages.
        """
        out = []
        assert len(feats) == self.num_stages, (
            f'The length of feature maps did not match the '
            f'`num_stages` in {self.__class__.__name__}')
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](feats[i])
            y = self.multi_final_layers[i](y)
            out.append(y)

        return out

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        """Predict results from multi-stage feature maps.

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

        if test_cfg.get('flip_test', False):
            # TTA: flip test
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)[-1]
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip)[-1],
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            multi_stage_heatmaps = self.forward(feats)
            batch_heatmaps = multi_stage_heatmaps[-1]

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Sequence[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Sequence[Tensor]): Multi-stage feature maps.
            batch_data_samples (List[:obj:`PoseDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instances`.
            train_cfg (Config, optional): The training config.

        Returns:
            dict: A dictionary of loss components.
        """
        multi_stage_pred_heatmaps = self.forward(feats)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses over multiple stages
        losses = dict()
        for i in range(self.num_stages):
            if isinstance(self.loss_module, nn.ModuleList):
                # use different loss_module over different stages
                loss_func = self.loss_module[i]
            else:
                # use the same loss_module over different stages
                loss_func = self.loss_module

            # the `gt_heatmaps` and `keypoint_weights` used to calculate loss
            # for different stages are the same
            loss_i = loss_func(multi_stage_pred_heatmaps[i], gt_heatmaps,
                               keypoint_weights)

            if 'loss_kpt' not in losses:
                losses['loss_kpt'] = loss_i
            else:
                losses['loss_kpt'] += loss_i

        # calculate accuracy
        _, avg_acc, _ = pose_pck_accuracy(
            output=to_numpy(multi_stage_pred_heatmaps[-1]),
            target=to_numpy(gt_heatmaps),
            mask=to_numpy(keypoint_weights) > 0)

        acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
        losses.update(acc_pose=acc_pose)

        return losses
