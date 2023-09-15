# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os.path as osp
from typing import List, Tuple

import numpy as np
from mmengine.fileio import get_local_path
from mmengine.logging import print_log

from mmpose.registry import DATASETS
from ..body3d import Human36mDataset


@DATASETS.register_module()
class H36MWholeBodyDataset(Human36mDataset):
    """H36MWholeBodyDataset dataset for pose estimation.

    H36M-WholeBody keypoints::

        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints

        In total, we have 133 keypoints for wholebody pose estimation.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        seq_step (int): The interval for extracting frames from the video.
            Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        multiple_target_step (int): The interval for merging sequence. Only
            valid when ``multiple_target`` is larger than 0. Default: 0.
        pad_video_seq (bool): Whether to pad the video so that poses will be
            predicted for every frame in the video. Default: ``False``.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        keypoint_2d_src (str): Specifies 2D keypoint information options, which
            should be one of the following options:

            - ``'gt'``: load from the annotation file
            - ``'detection'``: load from a detection
              result file of 2D keypoint
            - 'pipeline': the information will be generated by the pipeline

            Default: ``'gt'``.
        keypoint_2d_det_file (str, optional): The 2D keypoint detection file.
            If set, 2d keypoint loaded from this file will be used instead of
            ground-truth keypoints. This setting is only when
            ``keypoint_2d_src`` is ``'detection'``. Default: ``None``.
        factor_file (str, optional): The projection factors' file. If set,
            factor loaded from this file will be used instead of calculated
            factors. Default: ``None``.
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

    METAINFO: dict = dict(
        from_file='configs/_base_/datasets/coco_wholebody.py')
    SUPPORTED_keypoint_2d_src = {'gt', 'detection', 'pipeline'}

    def __init__(self, ann_file: str, data_root: str, data_prefix: dict,
                 **kwargs):
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = data_prefix

        # Process img_names
        _ann_file = osp.join(data_root, ann_file)
        with get_local_path(_ann_file) as local_path:
            self.ann_data = json.load(open(local_path))
            self._process_image_names(self.ann_data)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

    def _process_image_names(self, ann_data: dict) -> List[str]:
        """Process image names."""
        image_folder = self.data_prefix['img']
        img_names = [ann_data[i]['image_path'] for i in ann_data]
        image_paths = []
        for image_name in img_names:
            scene, _, sub, frame = image_name.split('/')
            frame, suffix = frame.split('.')
            frame_id = f'{int(frame.split("_")[-1]) + 1:06d}'
            sub = '_'.join(sub.split(' '))
            path = f'{scene}/{scene}_{sub}/{scene}_{sub}_{frame_id}.{suffix}'
            if not osp.exists(osp.join(self.data_root, image_folder, path)):
                print_log(
                    f'Failed to read image {path}.',
                    logger='current',
                    level=logging.WARN)
                continue
            image_paths.append(path)
        self.image_names = image_paths

    def get_sequence_indices(self) -> List[List[int]]:
        self.ann_data['imgname'] = self.image_names
        return super().get_sequence_indices()

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        num_keypoints = self.metainfo['num_keypoints']

        img_names = np.array(self.image_names)
        num_imgs = len(img_names)

        scales = np.zeros(num_imgs, dtype=np.float32)
        centers = np.zeros((num_imgs, 2), dtype=np.float32)

        kpts_3d, kpts_2d = [], []
        for _, ann in self.ann_data.items():
            if not isinstance(ann, dict):
                continue
            kpts_2d_i, kpts_3d_i = self._get_kpts(ann)
            kpts_3d.append(kpts_3d_i)
            kpts_2d.append(kpts_2d_i)

        kpts_3d = np.concatenate(kpts_3d, axis=0)
        kpts_2d = np.concatenate(kpts_2d, axis=0)
        kpts_visible = np.ones_like(kpts_2d[..., 0], dtype=np.float32)

        if self.factor_file:
            with get_local_path(self.factor_file) as local_path:
                factors = np.load(local_path).astype(np.float32)
        else:
            factors = np.zeros((kpts_3d.shape[0], ), dtype=np.float32)

        instance_list = []
        for idx, frame_ids in enumerate(self.sequence_indices):
            expected_num_frames = self.seq_len
            if self.multiple_target:
                expected_num_frames = self.multiple_target

            assert len(frame_ids) == (expected_num_frames), (
                f'Expected `frame_ids` == {expected_num_frames}, but '
                f'got {len(frame_ids)} ')

            _img_names = img_names[frame_ids]
            _kpts_2d = kpts_2d[frame_ids]
            _kpts_3d = kpts_3d[frame_ids]
            _kpts_visible = kpts_visible[frame_ids]
            factor = factors[frame_ids].astype(np.float32)

            target_idx = [-1] if self.causal else [int(self.seq_len) // 2]
            if self.multiple_target > 0:
                target_idx = list(range(self.multiple_target))

            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': _kpts_2d,
                'keypoints_3d': _kpts_3d,
                'keypoints_visible': _kpts_visible,
                'keypoints_3d_visible': _kpts_visible,
                'scale': scales[idx],
                'center': centers[idx].astype(np.float32).reshape(1, -1),
                'factor': factor,
                'id': idx,
                'category_id': 1,
                'iscrowd': 0,
                'img_paths': list(_img_names),
                'img_ids': frame_ids,
                'lifting_target': _kpts_3d[target_idx],
                'lifting_target_visible': _kpts_visible[target_idx],
                'target_img_path': _img_names[target_idx],
            }

            if self.camera_param_file:
                _cam_param = self.get_camera_param(_img_names[0])
            else:
                # Use the max value of camera parameters in Human3.6M dataset
                _cam_param = {
                    'w': 1000,
                    'h': 1002,
                    'f': np.array([[1149.67569987], [1148.79896857]]),
                    'c': np.array([[519.81583718], [515.45148698]])
                }
            instance_info['camera_param'] = _cam_param
            instance_list.append(instance_info)

        image_list = []
        for idx, img_name in enumerate(img_names):
            img_info = self.get_img_info(idx, img_name)
            image_list.append(img_info)

        return instance_list, image_list

    def _get_kpts(self, ann: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D keypoints and 3D keypoints from annotation."""
        kpts = ann['keypoints_3d']
        kpts_3d = np.array([[v for _, v in joint.items()]
                            for _, joint in kpts.items()],
                           dtype=np.float32).reshape(1, -1, 3)
        kpts_2d = kpts_3d[..., :2]
        return kpts_2d, kpts_3d
