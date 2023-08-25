# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import numpy as np
from mmengine.fileio import exists, get_local_path
from xtcocotools.coco import COCO

from mmpose.datasets.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class UBody3dDataset(BaseCocoStyleDataset):
    """Ubody3d dataset for 3D human pose estimation.

    "One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer",
    CVPR'2023. More details can be found in the `paper
    <https://arxiv.org/abs/2303.16160>`__ .

    Ubody3D keypoints::

        0-24: 25 body keypoints,
        25-64: 40 hand keypoints,
        65-136: 72 face keypoints,

        In total, we have 137 keypoints for wholebody 3D pose estimation.

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
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
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
        sample_interval (int, optional): The sample interval of the dataset.
            Default: 1.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ubody3d.py')

    def _load_annotations(self):
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), (
            f'Annotation file `{self.ann_file}`does not exist')

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            if img_id % self.sample_interval != 0:
                continue
            img = self.coco_loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):
                if instance_info := self.parse_data_info(
                        dict(raw_ann_info=ann, raw_img_info=img)):
                    instance_list.append(instance_info)

        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
                dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        if 'bbox' not in ann or 'keypoints3d' not in ann:
            return None

        img = raw_data_info['raw_img_info']
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        _keypoints_3d = np.array(
            ann['keypoints3d'], dtype=np.float32).reshape(1, -1, 4)
        keypoints_3d = _keypoints_3d[..., :3]
        keypoints_3d_visible = keypoints_visible

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        scale = ann.get('scale', 0.0)
        center = ann.get('center', np.array([0.0, 0.0]))

        instance_info = {
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'keypoints_3d': keypoints_3d,
            'keypoints_3d_visible': keypoints_3d_visible,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'scale': scale,
            'center': center,
            'id': ann['id'],
            'category_id': 1,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'img_path': img['img_path'],
            'img_id': ann['image_id'],
            'lifting_target': keypoints_3d[[-1]],
            'lifting_target_visible': keypoints_3d_visible[[-1]],
            'target_img_path': img['img_path'],
        }
        if 'crowdIndex' in img:
            instance_info['crowd_index'] = img['crowdIndex']
        return instance_info
