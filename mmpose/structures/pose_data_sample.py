# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.structures import BaseDataElement, InstanceData, PixelData

from mmpose.structures import MultilevelPixelData


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
        >>> from mmengine.structures import InstanceData, PixelData
        >>> from mmpose.structures import PoseDataSample

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
    def gt_fields(self) -> Union[PixelData, MultilevelPixelData]:
        return self._gt_fields

    @gt_fields.setter
    def gt_fields(self, value: Union[PixelData, MultilevelPixelData]):
        self.set_field(value, '_gt_fields', dtype=type(value))

    @gt_fields.deleter
    def gt_fields(self):
        del self._gt_fields

    @property
    def pred_fields(self) -> PixelData:
        return self._pred_heatmaps

    @pred_fields.setter
    def pred_fields(self, value: PixelData):
        self.set_field(value, '_pred_heatmaps', dtype=PixelData)

    @pred_fields.deleter
    def pred_fields(self):
        del self._pred_heatmaps
