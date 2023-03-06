# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union

from mmengine.dataset import BaseDataset
from mmengine.registry import build_from_cfg

from mmpose.registry import DATASETS
from .datasets.utils import parse_pose_metainfo


@DATASETS.register_module()
class CombinedDataset(BaseDataset):
    """A wrapper of combined dataset.

    Args:
        metainfo (dict): The meta information of combined dataset.
        datasets (list): The configs of datasets to be combined.
        pipeline (list, optional): Processing pipeline. Defaults to [].
    """

    def __init__(self,
                 metainfo: dict,
                 datasets: list,
                 pipeline: List[Union[dict, Callable]] = [],
                 **kwargs):

        self.datasets = []

        for cfg in datasets:
            dataset = build_from_cfg(cfg, DATASETS)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        self._len = sum(self._lens)

        super(CombinedDataset, self).__init__(pipeline=pipeline, **kwargs)
        self._metainfo = parse_pose_metainfo(metainfo)

    @property
    def metainfo(self):
        return deepcopy(self._metainfo)

    def __len__(self):
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
        """Given a data sample's global index, return the index of the sub-
        dataset the data sample belongs to, and the local index within that
        sub-dataset.

        Args:
            index (int): The global data sample index

        Returns:
            tuple[int, int]:
            - subset_index (int): The index of the sub-dataset
            - local_index (int): The index of the data sample within
                the sub-dataset
        """
        if index >= len(self) or index < -len(self):
            raise ValueError(
                f'index({index}) is out of bounds for dataset with '
                f'length({len(self)}).')

        if index < 0:
            index = index + len(self)

        subset_index = 0
        while index >= self._lens[subset_index]:
            index -= self._lens[subset_index]
            subset_index += 1
        return subset_index, index

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.The source dataset is
        depending on the index.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """

        data_info = self.get_data_info(idx)

        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``CombinedDataset``.
        Returns:
            dict: The idx-th annotation of the datasets.
        """
        subset_idx, sample_idx = self._get_subset_index(idx)
        # Get data sample processed by ``subset.pipeline``
        data_info = self.datasets[subset_idx][sample_idx]

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices'
        ]

        for key in metainfo_keys:
            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def full_init(self):
        """Fully initialize all sub datasets."""

        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()
        self._fully_initialized = True
