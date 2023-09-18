# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from mmengine.fileio import get_local_path
from xtcocotools.coco import COCO

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class UBody3dDataset(BaseMocapDataset):
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
        seq_len (int): Number of frames in a sequence. Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
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
    """

    def __init__(self,
                 multiple_target: int = 0,
                 multiple_target_step: int = 0,
                 seq_step: int = 1,
                 pad_video_seq: bool = False,
                 **kwargs):
        self.seq_step = seq_step
        self.pad_video_seq = pad_video_seq

        if multiple_target > 0 and multiple_target_step == 0:
            multiple_target_step = multiple_target
        self.multiple_target_step = multiple_target_step

        super().__init__(multiple_target=multiple_target, **kwargs)

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ubody3d.py')

    def _load_ann_file(self, ann_file: str) -> dict:
        """Load annotation file."""
        with get_local_path(ann_file) as local_path:
            self.ann_data = COCO(local_path)

    def get_sequence_indices(self) -> List[List[int]]:
        video_frames = defaultdict(list)
        img_ids = self.ann_data.getImgIds()
        for img_id in img_ids:
            img_info = self.ann_data.loadImgs(img_id)[0]
            subj, _, _ = self._parse_image_name(img_info['file_name'])
            video_frames[subj].append(img_id)

        sequence_indices = []
        _len = (self.seq_len - 1) * self.seq_step + 1
        _step = self.seq_step

        if self.multiple_target:
            for _, _img_ids in sorted(video_frames.items()):
                n_frame = len(_img_ids)
                _ann_ids = self.ann_data.getAnnIds(imgIds=_img_ids)
                seqs_from_video = [
                    _ann_ids[i:(i + self.multiple_target):_step]
                    for i in range(0, n_frame, self.multiple_target_step)
                ][:(n_frame + self.multiple_target_step -
                    self.multiple_target) // self.multiple_target_step]
                sequence_indices.extend(seqs_from_video)
        else:
            for _, _img_ids in sorted(video_frames.items()):
                n_frame = len(_img_ids)
                _ann_ids = self.ann_data.getAnnIds(imgIds=_img_ids)
                if self.pad_video_seq:
                    # Pad the sequence so that every frame in the sequence will
                    # be predicted.
                    if self.causal:
                        frames_left = self.seq_len - 1
                        frames_right = 0
                    else:
                        frames_left = (self.seq_len - 1) // 2
                        frames_right = frames_left
                    for i in range(n_frame):
                        pad_left = max(0, frames_left - i // _step)
                        pad_right = max(
                            0, frames_right - (n_frame - 1 - i) // _step)
                        start = max(i % _step, i - frames_left * _step)
                        end = min(n_frame - (n_frame - 1 - i) % _step,
                                  i + frames_right * _step + 1)
                        sequence_indices.append([_ann_ids[0]] * pad_left +
                                                _ann_ids[start:end:_step] +
                                                [_ann_ids[-1]] * pad_right)
                else:
                    seqs_from_video = [
                        _ann_ids[i:(i + _len):_step]
                        for i in range(0, n_frame - _len + 1, _step)
                    ]
                    sequence_indices.extend(seqs_from_video)

        # reduce dataset size if needed
        subset_size = int(len(sequence_indices) * self.subset_frac)
        start = np.random.randint(0, len(sequence_indices) - subset_size + 1)
        end = start + subset_size

        sequence_indices = sequence_indices[start:end]

        return sequence_indices

    def _parse_image_name(self, image_path: str) -> Tuple[str, int]:
        """Parse image name to get video name and frame index.

        Args:
            image_name (str): Image name.

        Returns:
            tuple[str, int]: Video name and frame index.
        """
        trim, file_name = image_path.split('/')[2:]
        frame_id, suffix = file_name.split('.')
        return trim, frame_id, suffix

    def _load_annotations(self):
        """Load data from annotations in COCO format."""
        num_keypoints = self.metainfo['num_keypoints']
        self._metainfo['CLASSES'] = self.ann_data.loadCats(
            self.ann_data.getCatIds())

        instance_list = []
        image_list = []

        for i, _ann_ids in enumerate(self.sequence_indices):
            expected_num_frames = self.seq_len
            if self.multiple_target:
                expected_num_frames = self.multiple_target

            assert len(_ann_ids) == (expected_num_frames), (
                f'Expected `frame_ids` == {expected_num_frames}, but '
                f'got {len(_ann_ids)} ')

            anns = self.ann_data.loadAnns(_ann_ids)
            imgs = self.ann_data.loadImgs([ann['image_id'] for ann in anns])

            _kpts = np.array([ann['keypoints'] for ann in anns],
                             dtype=np.float32)
            _kpts_3d = np.array([ann['keypoints_3d'] for ann in anns],
                                dtype=np.float32)
            _keypoints_visible = np.array(
                [ann['keypoints_valid'] for ann in anns], dtype=np.float32)
            _camera_params = np.array([ann['camera_param'] for ann in anns])
            _scales = np.zeros(len(imgs), dtype=np.float32)
            _centers = np.zeros((len(imgs), 2), dtype=np.float32)
            _img_paths = np.array([img['file_name'] for img in imgs])

            target_idx = [-1] if self.causal else [int(self.seq_len // 2)]
            if self.multiple_target:
                target_idx = list(range(self.multiple_target))

            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': _kpts,
                'keypoints_3d': _kpts_3d,
                'keypoints_visible': _keypoints_visible,
                'scale': _scales,
                'center': _centers,
                'id': i,
                'category_id': 1,
                'iscrowd': 0,
                'img_paths': list(_img_paths),
                'img_ids': [img['id'] for img in imgs],
                'lifting_target': _kpts_3d[target_idx],
                'lifting_target_visible': _keypoints_visible[target_idx],
                'target_img_paths': _img_paths[target_idx],
                'camera_param': _camera_params,
            }

            instance_list.append(instance_info)

        for img_id in self.ann_data.getImgIds():
            img = self.ann_data.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

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
        if 'bbox' not in ann or 'keypoints_3d' not in ann:
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
            ann['keypoints_3d'], dtype=np.float32).reshape(1, -1, 4)
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
