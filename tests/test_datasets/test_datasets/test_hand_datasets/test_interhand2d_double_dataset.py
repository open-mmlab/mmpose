# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.hand import InterHand2DDoubleDataset


class TestInterHand2DDoubleDataset(TestCase):

    def build_interhand2d_dataset(self, **kwargs):

        cfg = dict(
            ann_file='test_interhand2.6m_data.json',
            camera_param_file='test_interhand2.6m_camera.json',
            joint_file='test_interhand2.6m_joint_3d.json',
            data_mode='topdown',
            data_root='tests/data/interhand2.6m',
            pipeline=[],
            test_mode=False)

        cfg.update(kwargs)
        return InterHand2DDoubleDataset(**cfg)

    def check_metainfo_keys(self, metainfo: dict):
        expected_keys = dict(
            dataset_name=str,
            num_keypoints=int,
            keypoint_id2name=dict,
            keypoint_name2id=dict,
            upper_body_ids=list,
            lower_body_ids=list,
            flip_indices=list,
            flip_pairs=list,
            keypoint_colors=np.ndarray,
            num_skeleton_links=int,
            skeleton_links=list,
            skeleton_link_colors=np.ndarray,
            dataset_keypoint_weights=np.ndarray)

        for key, type_ in expected_keys.items():
            self.assertIn(key, metainfo)
            self.assertIsInstance(metainfo[key], type_, key)

    def check_data_info_keys(self,
                             data_info: dict,
                             data_mode: str = 'topdown'):
        if data_mode == 'topdown':
            expected_keys = dict(
                img_id=int,
                img_path=str,
                bbox=np.ndarray,
                bbox_score=np.ndarray,
                keypoints=np.ndarray,
                keypoints_visible=np.ndarray,
                hand_type=np.ndarray,
                hand_type_valid=np.ndarray,
                num_keypoints=int,
                iscrowd=bool,
                id=int)
        elif data_mode == 'bottomup':
            expected_keys = dict(
                img_id=int,
                img_path=str,
                bbox=np.ndarray,
                bbox_score=np.ndarray,
                keypoints=np.ndarray,
                keypoints_visible=np.ndarray,
                hand_type=np.ndarray,
                hand_type_valid=np.ndarray,
                num_keypoints=list,
                iscrowd=list,
                invalid_segs=list,
                id=list)
        else:
            raise ValueError(f'Invalid data_mode {data_mode}')

        for key, type_ in expected_keys.items():
            self.assertIn(key, data_info)
            self.assertIsInstance(data_info[key], type_, key)

    def test_metainfo(self):
        dataset = self.build_interhand2d_dataset()
        self.check_metainfo_keys(dataset.metainfo)
        # test dataset_name
        self.assertEqual(dataset.metainfo['dataset_name'], 'interhand3d')

        # test number of keypoints
        num_keypoints = 42
        self.assertEqual(dataset.metainfo['num_keypoints'], num_keypoints)
        self.assertEqual(
            len(dataset.metainfo['keypoint_colors']), num_keypoints)
        self.assertEqual(
            len(dataset.metainfo['dataset_keypoint_weights']), num_keypoints)

        # test some extra metainfo
        self.assertEqual(
            len(dataset.metainfo['skeleton_links']),
            len(dataset.metainfo['skeleton_link_colors']))

    def test_topdown(self):
        # test topdown training
        dataset = self.build_interhand2d_dataset(data_mode='topdown')
        self.assertEqual(dataset.data_mode, 'topdown')
        self.assertEqual(dataset.bbox_file, None)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

        # test topdown testing
        dataset = self.build_interhand2d_dataset(
            data_mode='topdown', test_mode=True)
        self.assertEqual(dataset.data_mode, 'topdown')
        self.assertEqual(dataset.bbox_file, None)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

    def test_bottomup(self):
        # test bottomup training
        dataset = self.build_interhand2d_dataset(data_mode='bottomup')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

        # test bottomup testing
        dataset = self.build_interhand2d_dataset(
            data_mode='bottomup', test_mode=True)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

    def test_exceptions_and_warnings(self):

        with self.assertRaisesRegex(ValueError, 'got invalid data_mode'):
            _ = self.build_interhand2d_dataset(data_mode='invalid')
