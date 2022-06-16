# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset

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

    @property
    def img_prefix(self) -> str:
        """The prefix of images.

        Returns:
            str: The prefix of images.
        """
        img_prefix = self.data_prefix.get('img', None)
        return img_prefix or ''
