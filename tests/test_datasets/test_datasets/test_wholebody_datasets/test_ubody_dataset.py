# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.wholebody3d import UBody3dDataset


class TestUBody3dDataset(TestCase):

    def build_ubody3d_dataset(self, **kwargs):

        cfg = dict(
            ann_file='ubody3d_train.json',
            data_mode='topdown',
            data_root='tests/data/ubody3d',
            pipeline=[],
            test_mode=False)

        cfg.update(kwargs)
        return UBody3dDataset(**cfg)

    def check_data_info_keys(self, data_info: dict):
        expected_keys = dict(
            img_paths=list,
            keypoints=np.ndarray,
            keypoints_3d=np.ndarray,
            scale=np.ndarray,
            center=np.ndarray,
            id=int)

        for key, type_ in expected_keys.items():
            self.assertIn(key, data_info)
            self.assertIsInstance(data_info[key], type_, key)

    def test_metainfo(self):
        dataset = self.build_ubody3d_dataset()
        # test dataset_name
        self.assertEqual(dataset.metainfo['dataset_name'], 'ubody3d')

        # test number of keypoints
        num_keypoints = 137
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
        dataset = self.build_ubody3d_dataset(data_mode='topdown')
        dataset.full_init()
        self.assertEqual(len(dataset), 1)
        self.check_data_info_keys(dataset[0])

        # test topdown testing
        dataset = self.build_ubody3d_dataset(
            data_mode='topdown', test_mode=True)
        dataset.full_init()
        self.assertEqual(len(dataset), 1)
        self.check_data_info_keys(dataset[0])

        # test topdown training with sequence config
        dataset = self.build_ubody3d_dataset(
            data_mode='topdown',
            seq_len=1,
            seq_step=1,
            causal=False,
            pad_video_seq=True)
        dataset.full_init()
        self.assertEqual(len(dataset), 1)
        self.check_data_info_keys(dataset[0])
