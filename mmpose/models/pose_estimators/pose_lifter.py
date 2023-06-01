# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Tuple, Union

from torch import Tensor

from mmpose.models.utils import check_and_update_config
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 Optional, OptMultiConfig, OptSampleList,
                                 PixelDataList, SampleList)
from .base import BasePoseEstimator


@MODELS.register_module()
class PoseLifter(BasePoseEstimator):
    """Base class for pose lifter.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        traj_backbone (dict, optional): The backbone config for trajectory
            model. Defaults to ``None``
        traj_neck (dict, optional): The neck config for trajectory model.
            Defaults to ``None``
        traj_head (dict, optional): The head config for trajectory model.
            Defaults to ``None``
        semi_loss (dict, optional): The semi-supervised loss config.
            Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 traj_backbone: OptConfigType = None,
                 traj_neck: OptConfigType = None,
                 traj_head: OptConfigType = None,
                 semi_loss: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        # trajectory model
        self.share_backbone = False
        if traj_head is not None:
            if traj_backbone is not None:
                self.traj_backbone = MODELS.build(traj_backbone)
            else:
                self.share_backbone = True

            # the PR #2108 and #2126 modified the interface of neck and head.
            # The following function automatically detects outdated
            # configurations and updates them accordingly, while also providing
            # clear and concise information on the changes made.
            traj_neck, traj_head = check_and_update_config(
                traj_neck, traj_head)

            if traj_neck is not None:
                self.traj_neck = MODELS.build(traj_neck)

            self.traj_head = MODELS.build(traj_head)

        # semi-supervised loss
        self.semi_supervised = semi_loss is not None
        if self.semi_supervised:
            assert any([head, traj_head])
            self.semi_loss = MODELS.build(semi_loss)

    @property
    def with_traj_backbone(self):
        """bool: Whether the pose lifter has trajectory backbone."""
        return hasattr(self, 'traj_backbone') and \
            self.traj_backbone is not None

    @property
    def with_traj_neck(self):
        """bool: Whether the pose lifter has trajectory neck."""
        return hasattr(self, 'traj_neck') and self.traj_neck is not None

    @property
    def with_traj(self):
        """bool: Whether the pose lifter has trajectory head."""
        return hasattr(self, 'traj_head')

    @property
    def causal(self):
        """bool: Whether the pose lifter is causal."""
        if hasattr(self.backbone, 'causal'):
            return self.backbone.causal
        else:
            raise AttributeError('A PoseLifter\'s backbone should have '
                                 'the bool attribute "causal" to indicate if'
                                 'it performs causal inference.')

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, K, C, T).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        # supervised learning
        # pose model
        feats = self.backbone(inputs)
        if self.with_neck:
            feats = self.neck(feats)

        # trajectory model
        if self.with_traj:
            if self.share_backbone:
                traj_x = feats
            else:
                traj_x = self.traj_backbone(inputs)

            if self.with_traj_neck:
                traj_x = self.traj_neck(traj_x)
            return feats, traj_x
        else:
            return feats

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None
                 ) -> Union[Tensor, Tuple[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, K, C, T).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """
        feats = self.extract_feat(inputs)

        if self.with_traj:
            # forward with trajectory model
            x, traj_x = feats
            if self.with_head:
                x = self.head.forward(x)

            traj_x = self.traj_head.forward(traj_x)
            return x, traj_x
        else:
            # forward without trajectory model
            x = feats
            if self.with_head:
                x = self.head.forward(x)
            return x

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, K, C, T).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = {}

        if self.with_traj:
            x, traj_x = feats
            # loss of trajectory model
            losses.update(
                self.traj_head.loss(
                    traj_x, data_samples, train_cfg=self.train_cfg))
        else:
            x = feats

        if self.with_head:
            # loss of pose model
            losses.update(
                self.head.loss(x, data_samples, train_cfg=self.train_cfg))

        # TODO: support semi-supervised learning
        if self.semi_supervised:
            losses.update(semi_loss=self.semi_loss(inputs, data_samples))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Note:
            - batch_size: B
            - num_input_keypoints: K
            - input_keypoint_dim: C
            - input_sequence_len: T

        Args:
            inputs (Tensor): Inputs with shape like (B, K, C, T).
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

        feats = self.extract_feat(inputs)

        pose_preds, batch_pred_instances, batch_pred_fields = None, None, None
        traj_preds, batch_traj_instances, batch_traj_fields = None, None, None
        if self.with_traj:
            x, traj_x = feats
            traj_preds = self.traj_head.predict(
                traj_x, data_samples, test_cfg=self.test_cfg)
        else:
            x = feats

        if self.with_head:
            pose_preds = self.head.predict(
                x, data_samples, test_cfg=self.test_cfg)

        if isinstance(pose_preds, tuple):
            batch_pred_instances, batch_pred_fields = pose_preds
        else:
            batch_pred_instances = pose_preds

        if isinstance(traj_preds, tuple):
            batch_traj_instances, batch_traj_fields = traj_preds
        else:
            batch_traj_instances = traj_preds

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields,
                                              batch_traj_instances,
                                              batch_traj_fields, data_samples)

        return results

    def add_pred_to_datasample(
        self,
        batch_pred_instances: InstanceList,
        batch_pred_fields: Optional[PixelDataList],
        batch_traj_instances: InstanceList,
        batch_traj_fields: Optional[PixelDataList],
        batch_data_samples: SampleList,
    ) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_traj_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_traj_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields, batch_traj_fields = [], []
        if batch_traj_instances is None:
            batch_traj_instances = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for (pred_instances, pred_fields, traj_instances, traj_fields,
             data_sample) in zip_longest(batch_pred_instances,
                                         batch_pred_fields,
                                         batch_traj_instances,
                                         batch_traj_fields,
                                         batch_data_samples):

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
