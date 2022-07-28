# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class FlyDataset(BaseCocoStyleDataset):
    """FlyDataset for animal pose estimation.

    "Fast animal pose estimation using deep neural networks"
    Nature methods'2019. More details can be found in the `paper
    <https://www.biorxiv.org/content/biorxiv/\
    early/2018/05/25/331181.full.pdf>`__ .

    Vinegar Fly keypoints::

        0: "head",
        1: "eyeL",
        2: "eyeR",
        3: "neck",
        4: "thorax",
        5: "abdomen",
        6: "forelegR1",
        7: "forelegR2",
        8: "forelegR3",
        9: "forelegR4",
        10: "midlegR1",
        11: "midlegR2",
        12: "midlegR3",
        13: "midlegR4",
        14: "hindlegR1",
        15: "hindlegR2",
        16: "hindlegR3",
        17: "hindlegR4",
        18: "forelegL1",
        19: "forelegL2",
        20: "forelegL3",
        21: "forelegL4",
        22: "midlegL1",
        23: "midlegL2",
        24: "midlegL3",
        25: "midlegL4",
        26: "hindlegL1",
        27: "hindlegL2",
        28: "hindlegL3",
        29: "hindlegL4",
        30: "wingL",
        31: "wingR"

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/fly.py')
