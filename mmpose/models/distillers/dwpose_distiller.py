# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.logging import MessageHub
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import load_checkpoint
from torch import Tensor

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models import build_pose_estimator
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ForwardResults, OptConfigType, OptMultiConfig,
                                 OptSampleList, SampleList)


@MODELS.register_module()
class DWPoseDistiller(BaseModel, metaclass=ABCMeta):
    """Distiller introduced in `DWPose`_ by Yang et al (2023). This distiller
    is designed for distillation of RTMPose.

    It typically consists of teacher_model and student_model. Please use the
    script `tools/misc/pth_transfer.py` to transfer the distilled model to the
    original RTMPose model.

    Args:
        teacher_cfg (str): Config file of the teacher model.
        student_cfg (str): Config file of the student model.
        two_dis (bool): Whether this is the second stage of distillation.
            Defaults to False.
        distill_cfg (dict): Config for distillation. Defaults to None.
        teacher_pretrained (str): Path of the pretrained teacher model.
            Defaults to None.
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``

    .. _`DWPose`: https://arxiv.org/abs/2307.15880
    """

    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 two_dis=False,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 train_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.teacher = build_pose_estimator(
            (Config.fromfile(teacher_cfg)).model)
        self.teacher_pretrained = teacher_pretrained
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = build_pose_estimator(
            (Config.fromfile(student_cfg)).model)

        self.distill_cfg = distill_cfg
        self.distill_losses = nn.ModuleDict()
        if self.distill_cfg is not None:
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    use_this = item_loss.use_this
                    if use_this:
                        self.distill_losses[loss_name] = MODELS.build(
                            item_loss)

        self.two_dis = two_dis
        self.train_cfg = train_cfg if train_cfg else self.student.train_cfg
        self.test_cfg = self.student.test_cfg
        self.metainfo = self.student.metainfo

    def init_weights(self):
        if self.teacher_pretrained is not None:
            load_checkpoint(
                self.teacher, self.teacher_pretrained, map_location='cpu')
        self.student.init_weights()

    def set_epoch(self):
        """Set epoch for distiller.

        Used for the decay of distillation loss.
        """
        self.message_hub = MessageHub.get_current_instance()
        self.epoch = self.message_hub.get_info('epoch')
        self.max_epochs = self.message_hub.get_info('max_epochs')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        self.set_epoch()

        losses = dict()

        with torch.no_grad():
            fea_t = self.teacher.extract_feat(inputs)
            lt_x, lt_y = self.teacher.head(fea_t)
            pred_t = (lt_x, lt_y)

        if not self.two_dis:
            fea_s = self.student.extract_feat(inputs)
            ori_loss, pred, gt, target_weight = self.head_loss(
                fea_s, data_samples, train_cfg=self.train_cfg)
            losses.update(ori_loss)
        else:
            ori_loss, pred, gt, target_weight = self.head_loss(
                fea_t, data_samples, train_cfg=self.train_cfg)

        all_keys = self.distill_losses.keys()

        if 'loss_fea' in all_keys:
            loss_name = 'loss_fea'
            losses[loss_name] = self.distill_losses[loss_name](fea_s[-1],
                                                               fea_t[-1])
            if not self.two_dis:
                losses[loss_name] = (
                    1 - self.epoch / self.max_epochs) * losses[loss_name]

        if 'loss_logit' in all_keys:
            loss_name = 'loss_logit'
            losses[loss_name] = self.distill_losses[loss_name](
                pred, pred_t, self.student.head.loss_module.beta,
                target_weight)
            if not self.two_dis:
                losses[loss_name] = (
                    1 - self.epoch / self.max_epochs) * losses[loss_name]

        return losses

    def predict(self, inputs, data_samples):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        if self.two_dis:
            assert self.student.with_head, (
                'The model must have head to perform prediction.')

            if self.test_cfg.get('flip_test', False):
                _feats = self.extract_feat(inputs)
                _feats_flip = self.extract_feat(inputs.flip(-1))
                feats = [_feats, _feats_flip]
            else:
                feats = self.extract_feat(inputs)

            preds = self.student.head.predict(
                feats, data_samples, test_cfg=self.student.test_cfg)

            if isinstance(preds, tuple):
                batch_pred_instances, batch_pred_fields = preds
            else:
                batch_pred_instances = preds
                batch_pred_fields = None

            results = self.student.add_pred_to_datasample(
                batch_pred_instances, batch_pred_fields, data_samples)

            return results
        else:
            return self.student.predict(inputs, data_samples)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.teacher.extract_feat(inputs)
        return x

    def head_loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.student.head.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.student.head.loss_module(pred_simcc, gt_simcc,
                                             keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.student.head.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses, pred_simcc, gt_simcc, keypoint_weights

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """
        return self.student._forward(inputs)
