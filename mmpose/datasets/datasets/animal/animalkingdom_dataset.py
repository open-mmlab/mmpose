# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AnimalKingdomDataset(BaseCocoStyleDataset):
    """Animal Kingdom dataset for animal pose estimation.

    "[CVPR2022] Animal Kingdom:
     A Large and Diverse Dataset for Animal Behavior Understanding"
    More details can be found in the `paper
    <https://www.researchgate.net/publication/
    359816954_Animal_Kingdom_A_Large_and_Diverse
    _Dataset_for_Animal_Behavior_Understanding>`__ .

    Website: <https://sutdcv.github.io/Animal-Kingdom>

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Animal Kingdom keypoint indexes::

        0: 'Head_Mid_Top',
        1: 'Eye_Left',
        2: 'Eye_Right',
        3: 'Mouth_Front_Top',
        4: 'Mouth_Back_Left',
        5: 'Mouth_Back_Right',
        6: 'Mouth_Front_Bottom',
        7: 'Shoulder_Left',
        8: 'Shoulder_Right',
        9: 'Elbow_Left',
        10: 'Elbow_Right',
        11: 'Wrist_Left',
        12: 'Wrist_Right',
        13: 'Torso_Mid_Back',
        14: 'Hip_Left',
        15: 'Hip_Right',
        16: 'Knee_Left',
        17: 'Knee_Right',
        18: 'Ankle_Left ',
        19: 'Ankle_Right',
        20: 'Tail_Top_Back',
        21: 'Tail_Mid_Back',
        22: 'Tail_End_Back

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ak.py')
