# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
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
        sample_ratio_factor (list, optional): A list of sampling ratio
            factors for each dataset. Defaults to None
    """

    def __init__(self,
                 metainfo: dict,
                 datasets: list,
                 pipeline: List[Union[dict, Callable]] = [],
                 sample_ratio_factor: Optional[List[float]] = None,
                 **kwargs):

        self.datasets = []
        self.resample = sample_ratio_factor is not None

        for cfg in datasets:
            dataset = build_from_cfg(cfg, DATASETS)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        if self.resample:
            assert len(sample_ratio_factor) == len(datasets), f'the length ' \
                f'of `sample_ratio_factor` {len(sample_ratio_factor)} does ' \
                f'not match the length of `datasets` {len(datasets)}'
            assert min(sample_ratio_factor) >= 0.0, 'the ratio values in ' \
                '`sample_ratio_factor` should not be negative.'
            self._lens_ori = self._lens
            self._lens = [
                round(l * sample_ratio_factor[i])
                for i, l in enumerate(self._lens_ori)
            ]

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

        if self.resample:
            gap = (self._lens_ori[subset_index] -
                   1e-4) / self._lens[subset_index]
            index = round(gap * index + np.random.rand() * gap - 0.5)

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

        # the assignment of 'dataset' should not be performed within the
        # `get_data_info` function. Otherwise, it can lead to the mixed
        # data augmentation process getting stuck.
        data_info['dataset'] = self

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

        if 'dataset' in data_info:
            data_info.pop('dataset')

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
