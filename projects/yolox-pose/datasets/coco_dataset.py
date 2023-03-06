from copy import deepcopy
from typing import Any

from mmengine.dataset import force_full_init
from mmyolo.registry import DATASETS

from mmpose.datasets import CocoDataset as MMPoseCocoDataset


@DATASETS.register_module()
class CocoDataset(MMPoseCocoDataset):

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    @force_full_init
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)
