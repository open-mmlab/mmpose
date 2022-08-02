# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class DeepFashionDataset(BaseCocoStyleDataset):
    """DeepFashion dataset (full-body clothes) for fashion landmark detection.

    "DeepFashion: Powering Robust Clothes Recognition
    and Retrieval with Rich Annotations", CVPR'2016.
    "Fashion Landmark Detection in the Wild", ECCV'2016.

    The dataset contains 3 categories for full-body, upper-body and lower-body.

    Fashion landmark indexes for upper-body clothes::

        0: 'left collar',
        1: 'right collar',
        2: 'left sleeve',
        3: 'right sleeve',
        4: 'left hem',
        5: 'right hem'

    Fashion landmark indexes for lower-body clothes::

        0: 'left waistline',
        1: 'right waistline',
        2: 'left hem',
        3: 'right hem'

    Fashion landmark indexes for full-body clothes::

        0: 'left collar',
        1: 'right collar',
        2: 'left sleeve',
        3: 'right sleeve',
        4: 'left waistline',
        5: 'right waistline',
        6: 'left hem',
        7: 'right hem'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        subset (str): Specifies the subset of body: ``'full'``, ``'upper'`` or
            ``'lower'``. Default: '', which means ``'full'``.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    def __init__(self,
                 ann_file: str = '',
                 subset: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self._check_subset_and_metainfo(subset)

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @classmethod
    def _check_subset_and_metainfo(cls, subset: str = '') -> None:
        """Check the subset of body and set the corresponding metainfo.

        Args:
            subset(str): the subset of body: could be ``'full'``, ``'upper'``
            or ``'lower'``. Default: '', which means ``'full'``.
        """
        if subset == '' or subset == 'full':
            cls.METAINFO = dict(
                from_file='configs/_base_/datasets/deepfashion_full.py')
        elif subset == 'upper':
            cls.METAINFO = dict(
                from_file='configs/_base_/datasets/deepfashion_upper.py')
        elif subset == 'lower':
            cls.METAINFO = dict(
                from_file='configs/_base_/datasets/deepfashion_lower.py')
        else:
            raise ValueError(
                f'{cls.__class__.__name__} got invalid subset: '
                f'{subset}. Should be "full", "lower" or "upper".')
