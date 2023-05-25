# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

from mmengine.dataset import force_full_init
from mmyolo.registry import DATASETS

from mmpose.datasets import CocoDataset as MMPoseCocoDataset


@DATASETS.register_module()
class CocoDataset(MMPoseCocoDataset):

    @force_full_init
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)
