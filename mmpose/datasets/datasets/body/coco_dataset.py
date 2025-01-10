# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Callable, List, Sequence
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


def generate_simple_dataset_info(labels: list) -> dict:
    dataset_info = dict(
        dataset_name='coco',
        paper_info=dict(
            author='',
            title=f'Auto: {len(labels)} keypoints',
            container='',
            year='',
            homepage='',
        ),
        keypoint_info={},
        skeleton_info={},
        joint_weights=[1.0] * len(labels),
        sigmas=[0.05] * len(labels)
    )

    # Generate keypoint_info
    last_label = labels[-1]
    for idx, label in enumerate(labels):
        dataset_info['keypoint_info'][idx] = dict(
            name=label,
            id=idx,
            color=[random.randint(0, 255) for _ in range(3)],  # Random RGB color
            type='upper',
            swap=last_label
        )
        last_label = label

    return dataset_info


@DATASETS.register_module()
class CocoDataset(BaseCocoStyleDataset):
    """COCO dataset for keypoints estimation.

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

    def __init__(self, labels, *args, **kwargs):
        self.default_config = 'configs/_base_/datasets/coco.py'
        super().__init__(metainfo=self.get_dataset_info(labels), *args, **kwargs)

    def get_dataset_info(self, labels: list) -> dict:
        print(f"Building dataset for labels: {labels}")
        if not isinstance(labels, list):
            print(f"Please specify labels in CocoDataset, used dataset config: {self.default_config}")
            return dict(from_file=self.default_config)

        dataset_info_path = f'configs/_base_/datasets/coco_{len(labels)}kp.py'
        if os.path.exists(dataset_info_path):
            print(f"Found custom dataset config: {self.default_config}")
            return dict(from_file=dataset_info_path)

        print(f"Dataset config not found, trying to generate automatically...")
        return generate_simple_dataset_info(labels)
