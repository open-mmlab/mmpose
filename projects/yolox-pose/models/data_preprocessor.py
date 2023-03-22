# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmdet.models import BatchSyncRandomResize
from mmyolo.registry import MODELS
from torch import Tensor

from mmpose.structures import PoseDataSample


@MODELS.register_module()
class PoseBatchSyncRandomResize(BatchSyncRandomResize):
    """Batch random resize which synchronizes the random size across ranks.

    This transform is similar to `mmdet.BatchSyncRandomResize`, but it also
    rescales the keypoints coordinates simultaneously.
    """

    def forward(self, inputs: Tensor, data_samples: List[PoseDataSample]
                ) -> Tuple[Tensor, List[PoseDataSample]]:

        inputs = inputs.float()
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            for data_sample in data_samples:
                data_sample.gt_instances.keypoints[..., 0] *= scale_x
                data_sample.gt_instances.keypoints[..., 1] *= scale_y

        return super().forward(inputs, data_samples)
