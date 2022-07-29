# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning

from mmpose.datasets.builder import DATASETS
from ..base import Kpt3dSviewDepthImgTopDownDataset


@DATASETS.register_module()
class NYUHandDataset(Kpt3dSviewDepthImgTopDownDataset):
    """TODO, add more detail doc.

    Args:
        ann_file (str): Path to the annotation file.
        camera_file (str): Path to the camera file.
        joint_file (str): Path to the joint file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        use_refined_center (bool): Using refined bbox center.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 use_refined_center=False,
                 align_uvd_xyz_direction=True,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/nyu.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['cube_size'] = np.array(data_cfg['cube_size'])
        self.ann_info['use_different_joint_weights'] = False

        self.camera_file = camera_file
        self.joint_file = joint_file
        self.align_uvd_xyz_direction = align_uvd_xyz_direction
        self.use_refined_center = use_refined_center
        if self.align_uvd_xyz_direction:
            self.flip_y = -1
        else:
            self.flip_y = 1
        self.meter2millimeter = 1 / 1000.

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)

        gt_db = []
        bbox_id = 0
        for img_id in self.img_ids:
            num_joints = self.ann_info['num_joints']

            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]

            frame_idx = str(img['frame_idx'])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            focal = np.array([cameras['fx'], cameras['fy']], dtype=np.float32)
            principal_pt = np.array([cameras['cx'], cameras['cy']],
                                    dtype=np.float32)

            joint_uvd = np.array(
                joints[frame_idx]['joint_uvd'], dtype=np.float32)
            joint_xyz = np.array(
                joints[frame_idx]['joint_xyz'], dtype=np.float32)
            joint_xyz[:, 1] *= self.flip_y

            # calculate bbox online
            # using center_xyz and cube_size, then project to 2D as bbox
            if self.use_refined_center:
                center_xyz = np.array(
                    ann['center_refined_xyz'],
                    dtype=np.float32).reshape(-1, 1)
            else:
                center_xyz = np.mean(joint_xyz, axis=0, keepdims=True)
            center_depth = center_xyz[0, 2]
            center_uvd = self._xyz2uvd(center_xyz, focal, principal_pt)

            if self.test_mode and img_id >= 2440:
                cube_size = np.array(
                    self.ann_info['cube_size'], dtype=np.float32) * 5.0 / 6.0
            else:
                cube_size = np.array(
                    self.ann_info['cube_size'], dtype=np.float32)

            bounds_uvd = self._center2bounds(center_uvd, cube_size, focal)
            bbox = np.array([
                bounds_uvd[0, 0], bounds_uvd[0, 2], bounds_uvd[0, 1] -
                bounds_uvd[0, 0], bounds_uvd[0, 3] - bounds_uvd[0, 2]
            ],
                            dtype=np.float32)

            valid_joints_idx = self.ann_info['dataset_channel']
            joint_valid = np.zeros(joint_xyz.shape[0], dtype=np.float32)
            joint_valid[valid_joints_idx] = 1.0

            # joint_3d will be normalized in pre-processing pipeline
            # uv are processed by TopDownAffine
            # depth are processed by DepthToTensor
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d[:, :2] = joint_uvd[:, :2]
            joints_3d[:, 2] = joint_uvd[:, 2]

            joints_3d_visible[...] = np.minimum(1, joint_valid.reshape(-1, 1))

            gt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'joints_cam': joint_xyz,
                'joints_uvd': joint_uvd,
                'cube_size': cube_size,
                'center_depth': center_depth,
                'focal': focal,
                'princpt': principal_pt,
                'dataset': self.dataset_name,
                'bbox': bbox,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='EPE', **kwargs):
        raise NotImplementedError
