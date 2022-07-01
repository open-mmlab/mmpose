# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmpose.core.utils.typing import (ConfigType, OptConfigType,
                                      OptMultiConfig, SampleList)
from mmpose.registry import MODELS
from .base import BasePoseEstimator


@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            _head = head.copy()
            _head.update(train_cfg=train_cfg)
            _head.update(test_cfg=test_cfg)
            self.head = MODELS.build(_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: SampleList = None):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        x = self.extract_feat(batch_inputs)
        if self.with_head:
            x = self.head.forward(x, batch_data_samples)

        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(batch_inputs)

        losses = dict()

        if self.with_head:
            losses.update(self.head.loss(feats, batch_data_samples))

        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is PoseDataSample instances with
            ``pred_instances`` field, which usually contains the following
            keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        feats = self.extract_feat(batch_inputs)
        preds = self.head.predict(feats, batch_inputs, batch_data_samples)

        return preds
