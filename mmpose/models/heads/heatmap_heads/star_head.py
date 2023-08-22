# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptSampleList
from .heatmap_head import HeatmapHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class STARHead(HeatmapHead):

    _version = 1

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
        # get normed gt_keypoints without torch.stack method
        # gt_keypoints_norm = torch.stack(
        #     [d.gt_instance_labels.keypoint_labels for
        #                            d in batch_data_samples])
        gt_keypoints_norm = batch_data_samples[
            0].gt_instance_labels.keypoint_labels

        w, h = self.decoder.heatmap_size
        gt_keypoints = gt_keypoints_norm * torch.from_numpy(np.array(
            [w, h])).to(gt_keypoints_norm)

        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_keypoints, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_keypoints),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_keypoints.device)
            losses.update(acc_pose=acc_pose)

        return losses
