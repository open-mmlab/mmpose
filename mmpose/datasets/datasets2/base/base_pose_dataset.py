# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any

from mmengine.dataset import BaseDataset, force_full_init

from ..utils import parse_pose_metainfo


class BasePoseDataset(BaseDataset):
    """Base class for all datasets in mmpose."""
    METAINFO: dict = dict()

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """
        metainfo = super()._load_metainfo(metainfo)
        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BasePoseDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the data_info before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'keypoint_weights'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return self.pipeline(data_info)
