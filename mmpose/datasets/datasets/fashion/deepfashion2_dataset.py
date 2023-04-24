# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module(name='DeepFashion2Dataset')
class DeepFashion2Dataset(BaseCocoStyleDataset):
    """DeepFashion2 dataset for fashion landmark detection."""

    METAINFO: dict = dict(from_file='configs/_base_/datasets/deepfashion2.py')
