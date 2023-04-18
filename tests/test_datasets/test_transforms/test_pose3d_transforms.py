# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose
from mmengine.fileio import load

from mmpose.datasets.transforms import (NormalizeKeypointsWithImage,
                                        PackPoseInputs, ZeroCenterPose)


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
        'target': keypoints_3d[target_idx, :, :3],
        'target_visible': keypoints_3d[target_idx, :, 3],
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


class TestZeroCenterPose(TestCase):

    def setUp(self):
        self.data_info = get_h36m_sample()

    def test_init(self):
        _ = ZeroCenterPose(
            item='target',
            root_index=0,
            remove_root=True,
            visible_item='target_visible',
            save_index=True)

    def test_transform(self):
        # test transform without removing root
        transform = ZeroCenterPose(
            item='target', root_index=0, save_index=True)
        results = deepcopy(self.data_info)
        results = transform(results)

        kpt1 = results['target']
        kpt2 = self.data_info['target']
        self.assertTrue(np.allclose(kpt1[0], np.zeros(3)))
        self.assertTrue(np.allclose(results['target_root'], kpt2[0]))

        # test transform removing root
        transform = ZeroCenterPose(
            item='target',
            root_index=0,
            remove_root=True,
            visible_item='target_visible',
            save_index=True)
        results = deepcopy(self.data_info)
        results = transform(results)

        kpt1 = results['target']
        kpt2 = self.data_info['target']
        self.assertEqual(kpt1.shape[0], kpt2.shape[0] - 1)
        self.assertEqual(results['target_root_index'], 0)


class TestNormalizeKeypointsWithImage(TestCase):

    def setUp(self):
        self.data_info = get_h36m_sample()

    def test_init(self):
        _ = NormalizeKeypointsWithImage(item='keypoints')

    def test_transform(self):
        transform = NormalizeKeypointsWithImage(
            item='keypoints', normalize_camera=True)
        results = deepcopy(self.data_info)
        results = transform(results)

        # test transformation of input
        kpts1 = self.data_info['keypoints']
        kpts2 = results['keypoints']
        camera_param = self.data_info['camera_param']
        center = np.array([0.5 * camera_param['w'], 0.5 * camera_param['h']],
                          dtype=np.float32)
        scale = np.array(0.5 * camera_param['w'], dtype=np.float32)
        kpts1 = (kpts1 - center) / scale
        np.testing.assert_array_almost_equal(kpts1, kpts2)

        # test transformation of camera
        camera1 = deepcopy(self.data_info['camera_param'])
        camera2 = results['camera_param']
        camera1['f'] = camera1['f'] / scale
        camera1['c'] = (camera1['c'] - center[:, None]) / scale
        np.testing.assert_array_almost_equal(camera1['f'], camera2['f'])
        np.testing.assert_array_almost_equal(camera1['c'], camera2['c'])


class Test3dPipeline(TestCase):

    def setUp(self):
        self.data_info = get_h36m_sample()
        self.meta_keys = ('id', 'img_id', 'img_path', 'flip_indices',
                          'target_img_path', 'target_root',
                          'target_root_index', 'target_mean', 'target_std',
                          'target_img_id', 'target_img_path')

    def test_transform(self):
        pipeline = Compose([
            ZeroCenterPose(
                item='target',
                root_index=0,
                remove_root=True,
                visible_item='target_visible',
                save_index=True),
            NormalizeKeypointsWithImage(item='keypoints'),
            PackPoseInputs(meta_keys=self.meta_keys)
        ])
        results = deepcopy(self.data_info)
        results = pipeline(results)
        gt_instances = results['data_samples'].gt_instances
        meta_info = results['data_samples'].metainfo

        self.assertEqual(results['inputs'].shape, (1, 17, 2))
        self.assertEqual(gt_instances['target'].shape, (16, 3))
        self.assertEqual(gt_instances['target_visible'].shape, (16, ))
        self.assertEqual(gt_instances['keypoints'].shape, (1, 17, 2))
        self.assertEqual(gt_instances['keypoints_visible'].shape, (1, 17))
        self.assertEqual(meta_info['target_root'].shape, (1, 3))
