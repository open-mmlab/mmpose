# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmengine.fileio import load

from mmpose.datasets.transforms import RandomFlipAroundRoot


def get_h36m_sample():

    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m image filename is like:
        S1_Directions_1.54138969_000001.jpg
        """
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        return subj, action, camera

    ann_flle = 'tests/data/h36m/test_h36m_body3d.npz'
    camera_param_file = 'tests/data/h36m/cameras.pkl'

    data = np.load(ann_flle)
    cameras = load(camera_param_file)

    imgnames = data['imgname']
    keypoints = data['part'].astype(np.float32)
    keypoints_3d = data['S'].astype(np.float32)
    centers = data['center'].astype(np.float32)
    scales = data['scale'].astype(np.float32)

    idx = 0
    target_idx = 0

    data_info = {
        'keypoints': keypoints[idx, :, :2].reshape(1, -1, 2),
        'keypoints_visible': keypoints[idx, :, 2].reshape(1, -1),
        'keypoints_3d': keypoints_3d[idx, :, :3].reshape(1, -1, 3),
        'keypoints_3d_visible': keypoints_3d[idx, :, 3].reshape(1, -1),
        'scale': scales[idx],
        'center': centers[idx].astype(np.float32).reshape(1, -1),
        'id': idx,
        'img_ids': [idx],
        'img_paths': [imgnames[idx]],
        'category_id': 1,
        'iscrowd': 0,
        'sample_idx': idx,
        'lifting_target': keypoints_3d[target_idx, :, :3],
        'lifting_target_visible': keypoints_3d[target_idx, :, 3],
        'target_img_path': osp.join('tests/data/h36m', imgnames[target_idx]),
    }

    # add camera parameters
    subj, _, camera = _parse_h36m_imgname(imgnames[idx])
    data_info['camera_param'] = cameras[(subj, camera)]

    # add ann_info
    ann_info = {}
    ann_info['num_keypoints'] = 17
    ann_info['dataset_keypoint_weights'] = np.full(17, 1.0, dtype=np.float32)
    ann_info['flip_pairs'] = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15],
                              [13, 16]]
    ann_info['skeleton_links'] = []
    ann_info['upper_body_ids'] = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    ann_info['lower_body_ids'] = (1, 2, 3, 4, 5, 6)
    ann_info['flip_indices'] = [
        0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13
    ]

    data_info.update(ann_info)

    return data_info


class TestRandomFlipAroundRoot(TestCase):

    def setUp(self):
        self.data_info = get_h36m_sample()
        self.keypoints_flip_cfg = dict(center_mode='static', center_x=0.)
        self.target_flip_cfg = dict(center_mode='root', center_index=0)

    def test_init(self):
        _ = RandomFlipAroundRoot(
            self.keypoints_flip_cfg,
            self.target_flip_cfg,
            flip_prob=0.5,
            flip_camera=False)

    def test_transform(self):
        kpts1 = self.data_info['keypoints']
        kpts_vis1 = self.data_info['keypoints_visible']
        tar1 = self.data_info['lifting_target']
        tar_vis1 = self.data_info['lifting_target_visible']

        transform = RandomFlipAroundRoot(
            self.keypoints_flip_cfg, self.target_flip_cfg, flip_prob=1)
        results = deepcopy(self.data_info)
        results = transform(results)

        kpts2 = results['keypoints']
        kpts_vis2 = results['keypoints_visible']
        tar2 = results['lifting_target']
        tar_vis2 = results['lifting_target_visible']

        self.assertEqual(kpts_vis2.shape, (1, 17))
        self.assertEqual(tar_vis2.shape, (17, ))
        self.assertEqual(kpts2.shape, (1, 17, 2))
        self.assertEqual(tar2.shape, (17, 3))

        flip_indices = [
            0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13
        ]
        for left, right in enumerate(flip_indices):
            self.assertTrue(
                np.allclose(-kpts1[0][left][:1], kpts2[0][right][:1], atol=4.))
            self.assertTrue(
                np.allclose(kpts1[0][left][1:], kpts2[0][right][1:], atol=4.))
            self.assertTrue(
                np.allclose(tar1[left][1:], tar2[right][1:], atol=4.))

            self.assertTrue(
                np.allclose(kpts_vis1[0][left], kpts_vis2[0][right], atol=4.))
            self.assertTrue(
                np.allclose(tar_vis1[left], tar_vis2[right], atol=4.))

        # test camera flipping
        transform = RandomFlipAroundRoot(
            self.keypoints_flip_cfg,
            self.target_flip_cfg,
            flip_prob=1,
            flip_camera=True)
        results = deepcopy(self.data_info)
        results = transform(results)

        camera2 = results['camera_param']
        self.assertTrue(
            np.allclose(
                -self.data_info['camera_param']['c'][0],
                camera2['c'][0],
                atol=4.))
        self.assertTrue(
            np.allclose(
                -self.data_info['camera_param']['p'][0],
                camera2['p'][0],
                atol=4.))
