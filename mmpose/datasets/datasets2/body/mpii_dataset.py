# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

import numpy as np
from mmengine.utils import check_file_exist

from mmpose.registry import DATASETS
from ..base import BaseCocoDataset


@DATASETS.register_module()
class MpiiDataset(BaseCocoDataset):
    """MPII Dataset for pose estimation.

    "2D Human Pose Estimation: New Benchmark and State of the Art Analysis"
    ,CVPR'2014. More details can be found in the `paper
    <http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf>`__ .

    MPII keypoints::

        0: 'right_ankle'
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii.py')

    def _load_annotations(self) -> List[dict]:
        """Load data from annotations in MPII format."""

        check_file_exist(self.ann_file)
        with open(self.ann_file) as anno_file:
            anns = json.load(anno_file)

        data_list = []
        ann_id = 0

        # mpii bbox scales are normalized with factor 200.
        pixel_std = 200.

        for ann in anns:
            center = np.array(ann['center'], dtype=np.float32)
            scale = np.array([ann['scale'], ann['scale']],
                             dtype=np.float32) * pixel_std

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15 / pixel_std * scale[1]

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1

            # the ground truth keypoint information is available
            # only when ``test_mode=False``
            # keypoints in shape [1, K, 2] and keypoints_visible in [1, K, 1]
            if not self.test_mode:
                keypoints = np.array(ann['joints']).reshape(1, -1, 2)
                keypoints_visible = np.array(ann['joints_vis']).reshape(
                    1, -1, 1)
            else:
                # use dummy keypoint location and visibility
                num_keypoints = self.metainfo['num_keypoints']
                keypoints = np.zeros((1, num_keypoints, 2), dtype=np.float32)
                keypoints_visible = np.ones((1, num_keypoints, 1),
                                            dtype=np.float32)
            data_info = {
                'id': ann_id,
                'img_id': int(ann['image'].split('.')[0]),
                'img_path': osp.join(self.data_prefix['img_path'],
                                     ann['image']),
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox_score': np.ones(1, dtype=np.float32),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
            }
            data_list.append(data_info)
            ann_id = ann_id + 1

        return data_list
