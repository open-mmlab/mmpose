# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.data import BaseDataElement, InstanceData, PixelData
from mmengine.utils import is_list_of


class PoseDataSample(BaseDataElement):
    """The base data structure of MMPose that is used as the interface between
    modules.

    The attributes of ``PoseDataSample`` includes:

        - ``gt_instances``(InstanceData): Ground truth of instances with
            keypoint annotations
        - ``pred_instances``(InstanceData): Instances with keypoint
            predictions
        - ``gt_fields``(PixelData): Ground truth of spatial distribution
            annotations like keypoint heatmaps and part affine fields (PAF)
        - ``pred_fields``(PixelData): Predictions of spatial distributions

    Examples:
        >>> import torch
        >>> from mmengine.data import InstanceData, PixelData
        >>> from mmpose.core import PoseDataSample

        >>> pose_meta = dict(img_shape=(800, 1216),
        ...                  crop_size=(256, 192),
        ...                  heatmap_size=(64, 48))
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.rand((1, 4))
        >>> gt_instances.keypoints = torch.rand((1, 17, 2))
        >>> gt_instances.keypoints_visible = torch.rand((1, 17, 1))
        >>> gt_fields = PixelData()
        >>> gt_fields.heatmaps = torch.rand((17, 64, 48))

        >>> data_sample = PoseDataSample(gt_instances=gt_instances,
        ...                              gt_fields=gt_fields,
        ...                              metainfo=pose_meta)
        >>> assert 'img_shape' in data_sample
        >>> len(data_sample.gt_intances)
        1
    """

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_instance_labels(self) -> InstanceData:
        return self._gt_instance_labels

    @gt_instance_labels.setter
    def gt_instance_labels(self, value: InstanceData):
        self.set_field(value, '_gt_instance_labels', dtype=InstanceData)

    @gt_instance_labels.deleter
    def gt_instance_labels(self):
        del self._gt_instance_labels

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def gt_heatmaps(self) -> PixelData:
        return self._gt_heatmaps

    @gt_heatmaps.setter
    def gt_heatmaps(self, value: PixelData):
        self.set_field(value, '_gt_heatmaps', dtype=PixelData)

    @gt_heatmaps.deleter
    def gt_heatmaps(self):
        del self._gt_heatmaps

    @property
    def pred_heatmaps(self) -> PixelData:
        return self._pred_heatmaps

    @pred_heatmaps.setter
    def pred_heatmaps(self, value: PixelData):
        self.set_field(value, '_pred_heatmaps', dtype=PixelData)

    @pred_heatmaps.deleter
    def pred_heatmaps(self):
        del self._pred_heatmaps

    @classmethod
    def merge(cls, data_samples: List['PoseDataSample']) -> 'PoseDataSample':
        """Merge the given data samples into a single data sample.

        This function can be used to merge the top-down predictions with
        bboxes from the same image. The merged data sample will contain all
        instances from the input data samples, and the identical metainfo with
        the first input data sample.

        Args:
            data_samples (List[:obj:`PoseDataSample`]): The data samples to
                merge

        Returns:
            PoseDataSample: The merged data sample.
        """

        if not is_list_of(data_samples, cls):
            raise ValueError('Invalid input type, should be a list of '
                             f':obj:`{cls.__name__}`')

        assert len(data_samples) > 0

        merged = cls(metainfo=data_samples[0].metainfo)

        if 'gt_instances' in data_samples[0]:
            merged.gt_instances = InstanceData.cat(
                [d.gt_instances for d in data_samples])

        if 'pred_instances' in data_samples[0]:
            merged.pred_instances = InstanceData.cat(
                [d.pred_instances for d in data_samples])

        # TODO: Support merging ``gt_fields`` and ``pred_fields``

        return merged
