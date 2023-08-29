# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.evaluation.functional import multilabel_classification_accuracy
from mmpose.models.necks import GlobalAveragePooling
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, InstanceList,
                                 OptConfigType, OptSampleList, Predictions)
from ..base_head import BaseHead
from .heatmap_head import HeatmapHead

OptIntSeq = Optional[Sequence[int]]


def make_linear_layers(feat_dims, relu_final=False):
    """Make linear layers."""
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or \
                (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Heatmap3DHead(HeatmapHead):
    """Heatmap3DHead is a sub-module of Interhand3DHead, and outputs 3D
    heatmaps. Heatmap3DHead is composed of (>=0) number of deconv layers and a
    simple conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        depth_size (int): Number of depth discretization size. Defaults to 64.
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``.
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``.
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings.
    """

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 depth_size: int = 64,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 final_layer: dict = dict(kernel_size=1),
                 init_cfg: OptConfigType = None):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            final_layer=final_layer,
            init_cfg=init_cfg)

        assert out_channels % depth_size == 0
        self.depth_size = depth_size

    def forward(self, feats: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tensor): Feature map.

        Returns:
            Tensor: output heatmap.
        """

        x = self.deconv_layers(feats)
        x = self.final_layer(x)
        N, C, H, W = x.shape
        # reshape the 2D heatmap to 3D heatmap
        x = x.reshape(N, C // self.depth_size, self.depth_size, H, W)

        return x


class Heatmap1DHead(nn.Module):
    """Heatmap1DHead is a sub-module of Interhand3DHead, and outputs 1D
    heatmaps.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        heatmap_size (int): Heatmap size. Defaults to 64.
        hidden_dims (Sequence[int]): Number of feature dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self,
                 in_channels: int = 2048,
                 heatmap_size: int = 64,
                 hidden_dims: Sequence[int] = (512, )):

        super().__init__()

        self.in_channels = in_channels
        self.heatmap_size = heatmap_size

        feature_dims = [in_channels, *hidden_dims, heatmap_size]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(
            self.heatmap_size, dtype=heatmap1d.dtype,
            device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.fc(feats)
        x = self.soft_argmax_1d(x).view(-1, 1)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


class MultilabelClassificationHead(nn.Module):
    """MultilabelClassificationHead is a sub-module of Interhand3DHead, and
    outputs hand type classification.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        num_labels (int): Number of labels. Defaults to 2.
        hidden_dims (Sequence[int]): Number of hidden dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self,
                 in_channels: int = 2048,
                 num_labels: int = 2,
                 hidden_dims: Sequence[int] = (512, )):

        super().__init__()

        self.in_channels = in_channels

        feature_dims = [in_channels, *hidden_dims, num_labels]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, x):
        """Forward function."""
        labels = self.fc(x)
        return labels


@MODELS.register_module()
class InternetHead(BaseHead):
    """Internet head introduced in `Interhand 2.6M`_ by Moon et al (2020).

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
            keypoint estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
            hand root depth estimation.
        hand_type_head_cfg (dict): Configs of ``MultilabelClassificationHead``
            for hand type classification.
        loss (Config): Config of the keypoint loss.
            Default: :class:`KeypointMSELoss`.
        loss_root_depth (dict): Config for relative root depth loss.
            Default: :class:`SmoothL1Loss`.
        loss_hand_type (dict): Config for hand type classification
            loss. Default: :class:`BCELoss`.
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Default: ``None``.
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Interhand 2.6M`: https://arxiv.org/abs/2008.09309
    """

    _version = 2

    def __init__(self,
                 keypoint_head_cfg: ConfigType,
                 root_head_cfg: ConfigType,
                 hand_type_head_cfg: ConfigType,
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 loss_root_depth: ConfigType = dict(
                     type='L1Loss', use_target_weight=True),
                 loss_hand_type: ConfigType = dict(
                     type='BCELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        super().__init__()

        # build sub-module heads
        self.right_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.left_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)
        self.hand_type_head = MultilabelClassificationHead(
            **hand_type_head_cfg)
        self.neck = GlobalAveragePooling()

        self.loss_module = MODELS.build(loss)
        self.root_loss_module = MODELS.build(loss_root_depth)
        self.hand_loss_module = MODELS.build(loss_hand_type)

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tuple[Tensor]: Output heatmap, root depth estimation and hand type
                classification.
        """
        x = feats[-1]
        outputs = []
        outputs.append(
            torch.cat([self.right_hand_head(x),
                       self.left_hand_head(x)], dim=1))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

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
            InstanceList: Return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_outputs = self.forward(_feats)
            _batch_heatmaps = _batch_outputs[0]

            _batch_outputs_flip = self.forward(_feats_flip)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_outputs_flip[0],
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))

            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5

            # flip relative hand root depth
            _batch_root = _batch_outputs[1]
            _batch_root_flip = -_batch_outputs_flip[1]
            batch_root = (_batch_root + _batch_root_flip) * 0.5

            # flip hand type
            _batch_type = _batch_outputs[2]
            _batch_type_flip = torch.empty_like(_batch_outputs_flip[2])
            _batch_type_flip[:, 0] = _batch_type[:, 1]
            _batch_type_flip[:, 1] = _batch_type[:, 0]
            batch_type = (_batch_type + _batch_type_flip) * 0.5

            batch_outputs = [batch_heatmaps, batch_root, batch_type]

        else:
            batch_outputs = self.forward(feats)

        preds = self.decode(tuple(batch_outputs))

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
        pred_fields = self.forward(feats)
        pred_heatmaps = pred_fields[0]
        _, K, D, W, H = pred_heatmaps.shape
        gt_heatmaps = torch.stack([
            d.gt_fields.heatmaps.reshape(K, D, W, H)
            for d in batch_data_samples
        ])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()

        # hand keypoint loss
        loss = self.loss_module(pred_heatmaps, gt_heatmaps, keypoint_weights)
        losses.update(loss_kpt=loss)

        # relative root depth loss
        gt_roots = torch.stack(
            [d.gt_instance_labels.root_depth for d in batch_data_samples])
        root_weights = torch.stack([
            d.gt_instance_labels.root_depth_weight for d in batch_data_samples
        ])
        loss_root = self.root_loss_module(pred_fields[1], gt_roots,
                                          root_weights)
        losses.update(loss_rel_root=loss_root)

        # hand type loss
        gt_types = torch.stack([
            d.gt_instance_labels.type.reshape(-1) for d in batch_data_samples
        ])
        type_weights = torch.stack(
            [d.gt_instance_labels.type_weight for d in batch_data_samples])
        loss_type = self.hand_loss_module(pred_fields[2], gt_types,
                                          type_weights)
        losses.update(loss_hand_type=loss_type)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            acc = multilabel_classification_accuracy(
                pred=to_numpy(pred_fields[2]),
                gt=to_numpy(gt_types),
                mask=to_numpy(type_weights))

            acc_pose = torch.tensor(acc, device=gt_types.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def decode(self, batch_outputs: Union[Tensor,
                                          Tuple[Tensor]]) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        batch_output_np = to_numpy(batch_outputs[0], unzip=True)
        batch_root_np = to_numpy(batch_outputs[1], unzip=True)
        batch_type_np = to_numpy(batch_outputs[2], unzip=True)
        batch_keypoints = []
        batch_scores = []
        batch_roots = []
        batch_types = []
        for outputs, roots, types in zip(batch_output_np, batch_root_np,
                                         batch_type_np):
            keypoints, scores, rel_root_depth, hand_type = _pack_and_call(
                tuple([outputs, roots, types]), self.decoder.decode)
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)
            batch_roots.append(rel_root_depth)
            batch_types.append(hand_type)

        preds = [
            InstanceData(
                keypoints=keypoints,
                keypoint_scores=scores,
                rel_root_depth=rel_root_depth,
                hand_type=hand_type)
            for keypoints, scores, rel_root_depth, hand_type in zip(
                batch_keypoints, batch_scores, batch_roots, batch_types)
        ]

        return preds
