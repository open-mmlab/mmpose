# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class Horse10Dataset(BaseCocoStyleDataset):
    """Horse10Dataset for animal pose estimation.

    "Pretraining boosts out-of-domain robustness for pose estimation"
    WACV'2021. More details can be found in the `paper
    <https://arxiv.org/pdf/1909.11229.pdf>`__ .

    Horse-10 keypoints::

        0: 'Nose',
        1: 'Eye',
        2: 'Nearknee',
        3: 'Nearfrontfetlock',
        4: 'Nearfrontfoot',
        5: 'Offknee',
        6: 'Offfrontfetlock',
        7: 'Offfrontfoot',
        8: 'Shoulder',
        9: 'Midshoulder',
        10: 'Elbow',
        11: 'Girth',
        12: 'Wither',
        13: 'Nearhindhock',
        14: 'Nearhindfetlock',
        15: 'Nearhindfoot',
        16: 'Hip',
        17: 'Stifle',
        18: 'Offhindhock',
        19: 'Offhindfetlock',
        20: 'Offhindfoot',
        21: 'Ischium'

    Args:
        ann_file (str): Annotation file path. Default: ''.
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
            ``dict(img=None, ann=None)``.
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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/horse10.py')
