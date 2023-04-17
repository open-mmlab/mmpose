# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.body3d import Human36mDataset


class TestH36MDataset(TestCase):

    def build_h36m_dataset(self, **kwargs):

        cfg = dict(
            ann_file='test_h36m_body3d.npz',
            data_mode='topdown',
            data_root='tests/data/h36m',
            pipeline=[],
            test_mode=False)

        cfg.update(kwargs)
        return Human36mDataset(**cfg)

    def check_data_info_keys(self,
                             data_info: dict,
                             data_mode: str = 'topdown'):
        if data_mode == 'topdown':
            expected_keys = dict(
                img_ids=list,
                img_paths=list,
                keypoints=np.ndarray,
                keypoints_3d=np.ndarray,
                scale=np.float32,
                center=np.ndarray,
                id=int)
        elif data_mode == 'bottomup':
            expected_keys = dict(
                img_ids=list,
                img_paths=list,
                keypoints=np.ndarray,
                keypoints_3d=np.ndarray,
                scale=list,
                center=np.ndarray,
                invalid_segs=list,
                id=list)
        else:
            raise ValueError(f'Invalid data_mode {data_mode}')

        for key, type_ in expected_keys.items():
            self.assertIn(key, data_info)
            self.assertIsInstance(data_info[key], type_, key)

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

    def test_metainfo(self):
        dataset = self.build_h36m_dataset()
        self.check_metainfo_keys(dataset.metainfo)
        # test dataset_name
        self.assertEqual(dataset.metainfo['dataset_name'], 'h36m')

        # test number of keypoints
        num_keypoints = 17
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
        dataset = self.build_h36m_dataset(data_mode='topdown')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

        # test topdown testing
        dataset = self.build_h36m_dataset(data_mode='topdown', test_mode=True)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

        # test topdown training with camera file
        dataset = self.build_h36m_dataset(
            data_mode='topdown', camera_param_file='cameras.pkl')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

        # test topdown training with sequence config
        dataset = self.build_h36m_dataset(
            data_mode='topdown',
            seq_len=27,
            seq_step=1,
            causal=False,
            pad_video_seq=True,
            camera_param_file='cameras.pkl')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

        # test topdown testing with 2d keypoint detection file and
        # sequence config
        dataset = self.build_h36m_dataset(
            data_mode='topdown',
            seq_len=27,
            seq_step=1,
            causal=False,
            pad_video_seq=True,
            test_mode=True,
            keypoint_2d_src='detection',
            keypoint_2d_det_file='test_h36m_2d_detection.npy')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0])

    def test_bottomup(self):
        # test bottomup training
        dataset = self.build_h36m_dataset(data_mode='bottomup')
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

        # test bottomup training
        dataset = self.build_h36m_dataset(
            data_mode='bottomup',
            seq_len=27,
            seq_step=1,
            causal=False,
            pad_video_seq=True)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

        # test bottomup testing
        dataset = self.build_h36m_dataset(data_mode='bottomup', test_mode=True)
        self.assertEqual(len(dataset), 4)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

    def test_exceptions_and_warnings(self):

        with self.assertRaisesRegex(ValueError, 'got invalid data_mode'):
            _ = self.build_h36m_dataset(data_mode='invalid')

        SUPPORTED_keypoint_2d_src = {'gt', 'detection', 'pipeline'}
        with self.assertRaisesRegex(
                ValueError, 'Unsupported `keypoint_2d_src` "invalid". '
                f'Supported options are {SUPPORTED_keypoint_2d_src}'):
            _ = self.build_h36m_dataset(
                data_mode='topdown',
                test_mode=False,
                keypoint_2d_src='invalid')

        with self.assertRaisesRegex(AssertionError,
                                    'Annotation file does not exist'):
            _ = self.build_h36m_dataset(
                data_mode='topdown', test_mode=False, ann_file='invalid')

        with self.assertRaisesRegex(AssertionError,
                                    'Unsupported `subset_frac` 2.'):
            _ = self.build_h36m_dataset(data_mode='topdown', subset_frac=2)
