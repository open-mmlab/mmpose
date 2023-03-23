# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple, Union

import torch
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.model import BaseModule, ModuleDict, Sequential
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmpose.evaluation.functional.nms import nearby_joints_nms
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, InstanceList,
                                 OptConfigType, OptSampleList, Predictions)
from ...backbones.resnet import BasicBlock
from ..base_head import BaseHead

try:
    from mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


class AdaptiveActivationBlock(BaseModule):
    """Adaptive activation convolution block. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        groups (int): Number of groups. Generally equal to the
            number of joints.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AdaptiveActivationBlock, self).__init__(init_cfg=init_cfg)

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups

        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())

        self.transform_matrix_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=in_channels,
            out_channels=6 * groups,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=True)

        if has_mmcv_full:
            self.adapt_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=groups,
                deform_groups=groups)
        else:
            raise ImportError('Please install the full version of mmcv '
                              'to use `DeformConv2d`.')

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        B, _, H, W = x.size()
        residual = x

        affine_matrix = self.transform_matrix_conv(x)
        affine_matrix = affine_matrix.permute(0, 2, 3, 1).contiguous()
        affine_matrix = affine_matrix.view(B, H, W, self.groups, 2, 3)
        offset = torch.matmul(affine_matrix, self.regular_matrix)
        offset = offset.transpose(4, 5).reshape(B, H, W, self.groups * 18)
        offset = offset.permute(0, 3, 1, 2).contiguous()

        x = self.adapt_conv(x, offset)
        x = self.norm(x)
        x = self.act(x + residual)

        return x


class RescoreNet(BaseModule):
    """Rescore net used to predict the OKS score of predicted pose. We use the
    off-the-shelf rescore net pretrained by authors of DEKR.

    Args:
        in_channels (int): Input channels
        norm_indexes (Tuple(int)): Indices of torso in skeleton
        init_cfg (dict, optional): Initialization config dict
    """

    def __init__(
        self,
        in_channels,
        norm_indexes,
        init_cfg=None,
    ):
        super(RescoreNet, self).__init__(init_cfg=init_cfg)

        self.norm_indexes = norm_indexes

        hidden = 256

        self.l1 = torch.nn.Linear(in_channels, hidden, bias=True)
        self.l2 = torch.nn.Linear(hidden, hidden, bias=True)
        self.l3 = torch.nn.Linear(hidden, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def make_feature(self, keypoints, keypoint_scores, skeleton):
        """Combine original scores, joint distance and relative distance to
        make feature.

        Args:
            keypoints (torch.Tensor): predicetd keypoints
            keypoint_scores (torch.Tensor): predicetd keypoint scores
            skeleton (list(list(int))): joint links

        Returns:
            torch.Tensor: feature for each instance
        """
        joint_1, joint_2 = zip(*skeleton)
        num_link = len(skeleton)

        joint_relate = (keypoints[:, joint_1] -
                        keypoints[:, joint_2])[:, :, :2]
        joint_length = joint_relate.norm(dim=2)

        # To use the torso distance to normalize
        normalize = (joint_length[:, self.norm_indexes[0]] +
                     joint_length[:, self.norm_indexes[1]]) / 2
        normalize = normalize.unsqueeze(1).expand(normalize.size(0), num_link)
        normalize = normalize.clamp(min=1).contiguous()

        joint_length = joint_length / normalize[:, :]
        joint_relate = joint_relate / normalize.unsqueeze(-1)
        joint_relate = joint_relate.flatten(1)

        feature = torch.cat((joint_relate, joint_length, keypoint_scores),
                            dim=1).float()
        return feature

    def forward(self, keypoints, keypoint_scores, skeleton):
        feature = self.make_feature(keypoints, keypoint_scores, skeleton)
        x = self.relu(self.l1(feature))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze(1)


@MODELS.register_module()
class DEKRHead(BaseHead):
    """DisEntangled Keypoint Regression head introduced in `Bottom-up human
    pose estimation via disentangled keypoint regression`_ by Geng et al
    (2021). The head is composed of a heatmap branch and a displacement branch.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        num_joints (int): Number of joints
        num_heatmap_filters (int): Number of filters for heatmap branch.
            Defaults to 32
        num_offset_filters_per_joint (int): Number of filters for each joint
            in displacement branch. Defaults to 15
        heatmap_loss (Config): Config of the heatmap loss. Defaults to use
            :class:`KeypointMSELoss`
        displacement_loss (Config): Config of the displacement regression loss.
            Defaults to use :class:`SoftWeightSmoothL1Loss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        rescore_cfg (Config, optional): The config for rescore net which
            estimates OKS via predicted keypoints and keypoint scores.
            Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Bottom-up human pose estimation via disentangled keypoint regression`:
        https://arxiv.org/abs/2104.02300
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_keypoints: int,
                 num_heatmap_filters: int = 32,
                 num_displacement_filters_per_keypoint: int = 15,
                 heatmap_loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 displacement_loss: ConfigType = dict(
                     type='SoftWeightSmoothL1Loss',
                     use_target_weight=True,
                     supervise_empty=False),
                 decoder: OptConfigType = None,
                 rescore_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        # build heatmap branch
        self.heatmap_conv_layers = self._make_heatmap_conv_layers(
            in_channels=in_channels,
            out_channels=1 + num_keypoints,
            num_filters=num_heatmap_filters,
        )

        # build displacement branch
        self.displacement_conv_layers = self._make_displacement_conv_layers(
            in_channels=in_channels,
            out_channels=2 * num_keypoints,
            num_filters=num_keypoints * num_displacement_filters_per_keypoint,
            groups=num_keypoints)

        # build losses
        self.loss_module = ModuleDict(
            dict(
                heatmap=MODELS.build(heatmap_loss),
                displacement=MODELS.build(displacement_loss),
            ))

        # build decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # build rescore net
        if rescore_cfg is not None:
            self.rescore_net = RescoreNet(**rescore_cfg)
        else:
            self.rescore_net = None

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def _make_heatmap_conv_layers(self, in_channels: int, out_channels: int,
                                  num_filters: int):
        """Create convolutional layers of heatmap branch by given
        parameters."""
        layers = [
            ConvModule(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_filters, num_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters,
                out_channels=out_channels,
                kernel_size=1),
        ]

        return Sequential(*layers)

    def _make_displacement_conv_layers(self, in_channels: int,
                                       out_channels: int, num_filters: int,
                                       groups: int):
        """Create convolutional layers of displacement branch by given
        parameters."""
        layers = [
            ConvModule(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            AdaptiveActivationBlock(num_filters, num_filters, groups=groups),
            AdaptiveActivationBlock(num_filters, num_filters, groups=groups),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups)
        ]

        return Sequential(*layers)

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is a tuple of heatmap and displacement.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tuple[Tensor]: output heatmap and displacement.
        """
        x = feats[-1]

        heatmaps = self.heatmap_conv_layers(x)
        displacements = self.displacement_conv_layers(x)

        return heatmaps, displacements

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
        pred_heatmaps, pred_displacements = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        heatmap_weights = torch.stack(
            [d.gt_fields.heatmap_weights for d in batch_data_samples])
        gt_displacements = torch.stack(
            [d.gt_fields.displacements for d in batch_data_samples])
        displacement_weights = torch.stack(
            [d.gt_fields.displacement_weights for d in batch_data_samples])

        if 'heatmap_mask' in batch_data_samples[0].gt_fields.keys():
            heatmap_mask = torch.stack(
                [d.gt_fields.heatmap_mask for d in batch_data_samples])
        else:
            heatmap_mask = None

        # calculate losses
        losses = dict()
        heatmap_loss = self.loss_module['heatmap'](pred_heatmaps, gt_heatmaps,
                                                   heatmap_weights,
                                                   heatmap_mask)
        displacement_loss = self.loss_module['displacement'](
            pred_displacements, gt_displacements, displacement_weights)

        losses.update({
            'loss/heatmap': heatmap_loss,
            'loss/displacement': displacement_loss,
        })

        return losses

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

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (1, h, w)
                    or (K+1, h, w) if keypoint heatmaps are predicted
                - displacements (Tensor): The predicted displacement fields
                    in shape (K*2, h, w)
        """

        assert len(batch_data_samples) == 1, f'DEKRHead only supports ' \
            f'prediction with batch_size 1, but got {len(batch_data_samples)}'

        multiscale_test = test_cfg.get('multiscale_test', False)
        flip_test = test_cfg.get('flip_test', False)
        metainfo = batch_data_samples[0].metainfo
        aug_scales = [1]

        if not multiscale_test:
            feats = [feats]
        else:
            aug_scales = aug_scales + metainfo['aug_scales']

        heatmaps, displacements = [], []
        for feat, s in zip(feats, aug_scales):
            if flip_test:
                assert isinstance(feat, list) and len(feat) == 2
                flip_indices = metainfo['flip_indices']
                _feat, _feat_flip = feat
                _heatmaps, _displacements = self.forward(_feat)
                _heatmaps_flip, _displacements_flip = self.forward(_feat_flip)

                _heatmaps_flip = flip_heatmaps(
                    _heatmaps_flip,
                    flip_mode='heatmap',
                    flip_indices=flip_indices + [len(flip_indices)],
                    shift_heatmap=test_cfg.get('shift_heatmap', False))
                _heatmaps = (_heatmaps + _heatmaps_flip) / 2.0

                _displacements_flip = flip_heatmaps(
                    _displacements_flip,
                    flip_mode='offset',
                    flip_indices=flip_indices,
                    shift_heatmap=False)

                # this is a coordinate amendment.
                x_scale_factor = s * (
                    metainfo['input_size'][0] / _heatmaps.shape[-1])
                _displacements_flip[:, ::2] += (x_scale_factor - 1) / (
                    x_scale_factor)
                _displacements = (_displacements + _displacements_flip) / 2.0

            else:
                _heatmaps, _displacements = self.forward(feat)

            heatmaps.append(_heatmaps)
            displacements.append(_displacements)

        preds = self.decode(heatmaps, displacements, test_cfg, metainfo)

        if test_cfg.get('output_heatmaps', False):
            heatmaps = [hm.detach() for hm in heatmaps]
            displacements = [dm.detach() for dm in displacements]
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
               displacements: Tuple[Tensor],
               test_cfg: ConfigType = {},
               metainfo: dict = {}) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            heatmaps (Tuple[Tensor]): The output heatmaps inferred from one
                image or multi-scale images.
            displacements (Tuple[Tensor]): The output displacement fields
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

        multiscale_test = test_cfg.get('multiscale_test', False)
        skeleton = metainfo.get('skeleton_links', None)

        preds = []
        batch_size = heatmaps[0].shape[0]

        for b in range(batch_size):
            if multiscale_test:
                raise NotImplementedError
            else:
                keypoints, (root_scores,
                            keypoint_scores) = self.decoder.decode(
                                heatmaps[0][b], displacements[0][b])

            # rescore each instance
            if self.rescore_net is not None and skeleton and len(
                    keypoints) > 0:
                instance_scores = self.rescore_net(keypoints, keypoint_scores,
                                                   skeleton)
                instance_scores[torch.isnan(instance_scores)] = 0
                root_scores = root_scores * instance_scores

            # nms
            keypoints, keypoint_scores = to_numpy((keypoints, keypoint_scores))
            scores = to_numpy(root_scores)[..., None] * keypoint_scores
            if len(keypoints) > 0 and test_cfg.get('nms_dist_thr', 0) > 0:
                kpts_db = []
                for i in range(len(keypoints)):
                    kpts_db.append(
                        dict(keypoints=keypoints[i], score=keypoint_scores[i]))
                keep_instance_inds = nearby_joints_nms(
                    kpts_db,
                    test_cfg['nms_dist_thr'],
                    test_cfg.get('nms_joints_thr', None),
                    score_per_joint=True,
                    max_dets=test_cfg.get('max_num_people', 30))
                keypoints = keypoints[keep_instance_inds]
                scores = scores[keep_instance_inds]

            # pack outputs
            preds.append(
                InstanceData(keypoints=keypoints, keypoint_scores=scores))

        return preds

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DEKRHead` (before MMPose v1.0.0) to a compatible format
        of :class:`DEKRHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'offset_conv_layer' in k:
                v = state_dict.pop(k)
                k = k.replace('offset_conv_layers', 'displacement_conv_layers')
                if 'displacement_conv_layers.3.' in k:
                    # the source and target of displacement vectors are
                    # opposite between two versions.
                    v = -v
                state_dict[k] = v

            if 'heatmap_conv_layers.2' in k:
                # root heatmap is at the first/last channel of the
                # heatmap tensor in MMPose v0.x/1.x, respectively.
                v = state_dict.pop(k)
                state_dict[k] = torch.cat((v[1:], v[:1]))

            if 'rescore_net' in k:
                v = state_dict.pop(k)
                k = k.replace('rescore_net', 'head.rescore_net')
                state_dict[k] = v
