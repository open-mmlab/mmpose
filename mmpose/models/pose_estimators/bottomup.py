# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import List, Optional, Union

from mmengine.utils import is_list_of
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator


@MODELS.register_module()
class BottomupPoseEstimator(BasePoseEstimator):
    """Base class for bottom-up pose estimators.

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

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

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

    def predict(self, inputs: Union[Tensor, List[Tensor]],
                data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor | List[Tensor]): Input image in tensor or image
                pyramid as a list of tensors. Each tensor is in shape
                [B, C, H, W]
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

        multiscale_test = self.test_cfg.get('multiscale_test', False)
        flip_test = self.test_cfg.get('flip_test', False)

        # enable multi-scale test
        aug_scales = data_samples[0].metainfo.get('aug_scales', None)
        if multiscale_test:
            assert isinstance(aug_scales, list)
            assert is_list_of(inputs, Tensor)
            # `inputs` includes images in original and augmented scales
            assert len(inputs) == len(aug_scales) + 1
        else:
            assert isinstance(inputs, Tensor)
            # single-scale test
            inputs = [inputs]

        feats = []
        for _inputs in inputs:
            if flip_test:
                _feats_orig = self.extract_feat(_inputs)
                _feats_flip = self.extract_feat(_inputs.flip(-1))
                _feats = [_feats_orig, _feats_flip]
            else:
                _feats = self.extract_feat(_inputs)

            feats.append(_feats)

        if not multiscale_test:
            feats = feats[0]

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
            The length of the list is the batch size when ``merge==False``, or
            1 when ``merge==True``.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            # convert keypoint coordinates from input space to image space
            input_size = data_sample.metainfo['input_size']
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']

            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * input_scale + input_center - 0.5 * input_scale

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                data_sample.pred_fields = pred_fields

        return batch_data_samples
