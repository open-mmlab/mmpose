# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional, Tuple

from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator


@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
    """

    _version = 2

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
            self.head = MODELS.build(head)

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)

        return x

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
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
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, Tuple):
            pred_instances, pred_fields = preds
        else:
            pred_instances = preds
            pred_fields = None

        results = self.convert_to_datasample(pred_instances, pred_fields,
                                             data_samples)

        return results

    def convert_to_datasample(self, pred_instances: InstanceList,
                              pred_fields: Optional[PixelDataList],
                              data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` or ``pred_fields`` field of
            each data sample.
        """
        assert len(pred_instances) == len(data_samples)
        if pred_fields is None:
            pred_fields = []

        for pred_instance, pred_field, data_sample in zip_longest(
                pred_instances, pred_fields, data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            pred_instance.keypoints = pred_instance.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            data_sample.pred_instances = pred_instance

            if pred_field is not None:
                data_sample.pred_fields = pred_field

        return data_samples

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_head' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_head', 'head')
                state_dict[k] = v
