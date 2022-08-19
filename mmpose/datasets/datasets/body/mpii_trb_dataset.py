# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

import numpy as np
from mmengine.utils import check_file_exist

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_cs2xyxy
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class MpiiTrbDataset(BaseCocoStyleDataset):
    """MPII-TRB Dataset dataset for pose estimation.

    "TRB: A Novel Triplet Representation for Understanding 2D Human Body",
    ICCV'2019. More details can be found in the `paper
    <https://arxiv.org/abs/1910.11535>`__ .

    MPII-TRB keypoints::

        0: 'left_shoulder'
        1: 'right_shoulder'
        2: 'left_elbow'
        3: 'right_elbow'
        4: 'left_wrist'
        5: 'right_wrist'
        6: 'left_hip'
        7: 'right_hip'
        8: 'left_knee'
        9: 'right_knee'
        10: 'left_ankle'
        11: 'right_ankle'
        12: 'head'
        13: 'neck'

        14: 'right_neck'
        15: 'left_neck'
        16: 'medial_right_shoulder'
        17: 'lateral_right_shoulder'
        18: 'medial_right_bow'
        19: 'lateral_right_bow'
        20: 'medial_right_wrist'
        21: 'lateral_right_wrist'
        22: 'medial_left_shoulder'
        23: 'lateral_left_shoulder'
        24: 'medial_left_bow'
        25: 'lateral_left_bow'
        26: 'medial_left_wrist'
        27: 'lateral_left_wrist'
        28: 'medial_right_hip'
        29: 'lateral_right_hip'
        30: 'medial_right_knee'
        31: 'lateral_right_knee'
        32: 'medial_right_ankle'
        33: 'lateral_right_ankle'
        34: 'medial_left_hip'
        35: 'lateral_left_hip'
        36: 'medial_left_knee'
        37: 'lateral_left_knee'
        38: 'medial_left_ankle'
        39: 'lateral_left_ankle'

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii_trb.py')

    def _load_annotations(self) -> List[dict]:
        """Load data from annotations in MPII-TRB format."""

        check_file_exist(self.ann_file)
        with open(self.ann_file) as anno_file:
            data = json.load(anno_file)

        imgid2info = {img['id']: img for img in data['images']}

        data_list = []

        # mpii-trb bbox scales are normalized with factor 200.
        pixel_std = 200.

        for ann in data['annotations']:
            img_id = ann['image_id']

            # center, scale in shape [1, 2] and bbox in [1, 4]
            center = np.array([ann['center']], dtype=np.float32)
            scale = np.array([[ann['scale'], ann['scale']]],
                             dtype=np.float32) * pixel_std
            bbox = bbox_cs2xyxy(center, scale)

            # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            _keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints = _keypoints[..., :2]
            keypoints_visible = np.minimum(1, _keypoints[..., 2])

            img_path = osp.join(self.data_prefix['img'],
                                imgid2info[img_id]['file_name'])

            data_info = {
                'id': ann['id'],
                'img_id': img_id,
                'img_path': img_path,
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'num_keypoints': ann['num_joints'],
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'iscrowd': ann['iscrowd'],
            }

            # val set
            if 'headbox' in ann:
                data_info['headbox'] = np.array(
                    ann['headbox'], dtype=np.float32)

            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x['id'])
        return data_list
