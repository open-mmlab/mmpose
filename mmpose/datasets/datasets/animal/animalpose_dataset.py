# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AnimalPoseDataset(BaseCocoStyleDataset):
    """Animal-Pose dataset for animal pose estimation.

    "Cross-domain Adaptation For Animal Pose Estimation" ICCV'2019
    More details can be found in the `paper
    <https://arxiv.org/abs/1908.05806>`__ .

    Animal-Pose keypoints::

        0: 'L_Eye',
        1: 'R_Eye',
        2: 'L_EarBase',
        3: 'R_EarBase',
        4: 'Nose',
        5: 'Throat',
        6: 'TailBase',
        7: 'Withers',
        8: 'L_F_Elbow',
        9: 'R_F_Elbow',
        10: 'L_B_Elbow',
        11: 'R_B_Elbow',
        12: 'L_F_Knee',
        13: 'R_F_Knee',
        14: 'L_B_Knee',
        15: 'R_B_Knee',
        16: 'L_F_Paw',
        17: 'R_F_Paw',
        18: 'L_B_Paw',
        19: 'R_B_Paw'

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/animalpose.py')
