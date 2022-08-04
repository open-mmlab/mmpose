# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import numpy as np
from mmengine.utils import check_file_exist
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class CocoWholeBodyHandDataset(BaseCocoStyleDataset):
    """CocoWholeBodyDataset for hand pose estimation.

    "Whole-Body Human Pose Estimation in the Wild", ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/abs/2007.11858>`__ .

    COCO-WholeBody Hand keypoints::

        0: 'wrist',
        1: 'thumb1',
        2: 'thumb2',
        3: 'thumb3',
        4: 'thumb4',
        5: 'forefinger1',
        6: 'forefinger2',
        7: 'forefinger3',
        8: 'forefinger4',
        9: 'middle_finger1',
        10: 'middle_finger2',
        11: 'middle_finger3',
        12: 'middle_finger4',
        13: 'ring_finger1',
        14: 'ring_finger2',
        15: 'ring_finger3',
        16: 'ring_finger4',
        17: 'pinky_finger1',
        18: 'pinky_finger2',
        19: 'pinky_finger3',
        20: 'pinky_finger4'

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

    METAINFO: dict = dict(
        from_file='configs/_base_/datasets/coco_wholebody_hand.py')

    def _load_annotations(self) -> List[dict]:
        """Load data from annotations in COCO format."""

        check_file_exist(self.ann_file)

        coco = COCO(self.ann_file)
        data_list = []
        id = 0

        for img_id in coco.getImgIds():
            img = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                for type in ['left', 'right']:
                    # filter invalid hand annotations, there might be two
                    # valid instances (left and right hand) in one image
                    if ann[f'{type}hand_valid'] and max(
                            ann[f'{type}hand_kpts']) > 0:
                        img_path = osp.join(self.data_prefix['img'],
                                            img['file_name'])

                        bbox_xywh = np.array(
                            ann[f'{type}hand_box'],
                            dtype=np.float32).reshape(1, 4)

                        bbox = bbox_xywh2xyxy(bbox_xywh)

                        _keypoints = np.array(
                            ann[f'{type}hand_kpts'],
                            dtype=np.float32).reshape(1, -1, 3)
                        keypoints = _keypoints[..., :2]
                        keypoints_visible = np.minimum(1, _keypoints[..., 2])

                        num_keypoints = np.count_nonzero(keypoints.max(axis=2))

                        data_info = {
                            'img_id': ann['image_id'],
                            'img_path': img_path,
                            'bbox': bbox,
                            'bbox_score': np.ones(1, dtype=np.float32),
                            'num_keypoints': num_keypoints,
                            'keypoints': keypoints,
                            'keypoints_visible': keypoints_visible,
                            'iscrowd': ann['iscrowd'],
                            'segmentation': ann['segmentation'],
                            'id': id,
                        }
                        data_list.append(data_info)
                        id = id + 1

        data_list = sorted(data_list, key=lambda x: x['id'])
        return data_list
