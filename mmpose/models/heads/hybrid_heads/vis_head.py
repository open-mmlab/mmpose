# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from torch import Tensor, nn

from mmpose.models.utils.tta import flip_visibility
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead


@MODELS.register_module()
class VisPredictHead(BaseHead):
    """VisPredictHead must be used together with other heads. It can predict
    keypoints coordinates of and their visibility simultaneously. In the
    current version, it only supports top-down approaches.

    Args:
        pose_cfg (Config): Config to construct keypoints prediction head
        loss (Config): Config for visibility loss. Defaults to use
            :class:`BCELoss`
        use_sigmoid (bool): Whether to use sigmoid activation function
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(self,
                 pose_cfg: ConfigType,
                 loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=False,
                     with_logits=True),
                 use_sigmoid: bool = False,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = pose_cfg['in_channels']
        if pose_cfg.get('num_joints', None) is not None:
            self.out_channels = pose_cfg['num_joints']
        elif pose_cfg.get('out_channels', None) is not None:
            self.out_channels = pose_cfg['out_channels']
        else:
            raise ValueError('VisPredictHead requires \'num_joints\' or'
                             ' \'out_channels\' in the pose_cfg.')

        self.loss_module = MODELS.build(loss)

        self.pose_head = MODELS.build(pose_cfg)
        self.pose_cfg = pose_cfg

        self.use_sigmoid = use_sigmoid

        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels, self.out_channels)
        ]
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.vis_head = nn.Sequential(*modules)

    def vis_forward(self, feats: Tuple[Tensor]):
        """Forward the vis_head. The input is multi scale feature maps and the
        output is coordinates visibility.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates visibility.
        """
        x = feats[-1]
        while len(x.shape) < 4:
            x.unsqueeze_(-1)
        x = self.vis_head(x)
        return x.reshape(-1, self.out_channels)

    def forward(self, feats: Tuple[Tensor]):
        """Forward the network. The input is multi scale feature maps and the
        output is coordinates and coordinates visibility.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tuple[Tensor]: output coordinates and coordinates visibility.
        """
        x_pose = self.pose_head.forward(feats)
        x_vis = self.vis_forward(feats)

        return x_pose, x_vis

    def integrate(self, batch_vis: Tensor,
                  pose_preds: Union[Tuple, Predictions]) -> InstanceList:
        """Add keypoints visibility prediction to pose prediction.

        Overwrite the original keypoint_scores.
        """
        if isinstance(pose_preds, tuple):
            pose_pred_instances, pose_pred_fields = pose_preds
        else:
            pose_pred_instances = pose_preds
            pose_pred_fields = None

        batch_vis_np = to_numpy(batch_vis, unzip=True)

        assert len(pose_pred_instances) == len(batch_vis_np)
        for index, _ in enumerate(pose_pred_instances):
            pose_pred_instances[index].keypoint_scores = batch_vis_np[index]

        return pose_pred_instances, pose_pred_fields

    def predict(self,
                feats: Tuple[Tensor],
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
            posehead's ``test_cfg['output_heatmap']==True``, return both
            pose and heatmap prediction; otherwise only return the pose
            prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_visibility (np.ndarray): predicted keypoints
                    visibility in shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_vis = self.vis_forward(_feats)
            _batch_vis_flip = flip_visibility(
                self.vis_forward(_feats_flip), flip_indices=flip_indices)
            batch_vis = (_batch_vis + _batch_vis_flip) * 0.5
        else:
            batch_vis = self.vis_forward(feats)  # (B, K, D)

        batch_vis.unsqueeze_(dim=1)  # (B, N, K, D)

        if not self.use_sigmoid:
            batch_vis = torch.sigmoid(batch_vis)

        batch_pose = self.pose_head.predict(feats, batch_data_samples,
                                            test_cfg)

        return self.integrate(batch_vis, batch_pose)

    def vis_accuracy(self, vis_pred_outputs, vis_labels):
        """Calculate visibility prediction accuracy."""
        probabilities = torch.sigmoid(torch.flatten(vis_pred_outputs))
        threshold = 0.5
        predictions = (probabilities >= threshold).int()
        labels = torch.flatten(vis_labels)
        correct = torch.sum(predictions == labels).item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
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
        vis_pred_outputs = self.vis_forward(feats)
        vis_labels = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate vis losses
        losses = dict()
        loss_vis = self.loss_module(vis_pred_outputs, vis_labels)

        losses.update(loss_vis=loss_vis)

        # calculate vis accuracy
        acc_vis = self.vis_accuracy(vis_pred_outputs, vis_labels)
        losses.update(acc_vis=acc_vis)

        # calculate keypoints losses
        loss_kpt = self.pose_head.loss(feats, batch_data_samples)
        losses.update(loss_kpt)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
