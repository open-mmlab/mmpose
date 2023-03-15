# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.dataset_wrappers import CombinedDataset


class TestCombinedDataset(TestCase):

    def build_combined_dataset(self, **kwargs):

        coco_cfg = dict(
            type='CocoDataset',
            ann_file='test_coco.json',
            bbox_file=None,
            data_mode='topdown',
            data_root='tests/data/coco',
            pipeline=[],
            test_mode=False)

        aic_cfg = dict(
            type='AicDataset',
            ann_file='test_aic.json',
            bbox_file=None,
            data_mode='topdown',
            data_root='tests/data/aic',
            pipeline=[],
            test_mode=False)

        cfg = dict(
            metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
            datasets=[coco_cfg, aic_cfg],
            pipeline=[])
        cfg.update(kwargs)
        return CombinedDataset(**cfg)

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
                id=int)
        elif data_mode == 'bottomup':
            expected_keys = dict(
                img_id=int,
                img_path=str,
                bbox=np.ndarray,
                bbox_score=np.ndarray,
                keypoints=np.ndarray,
                keypoints_visible=np.ndarray,
                invalid_segs=list,
                id=list)
        else:
            raise ValueError(f'Invalid data_mode {data_mode}')

        for key, type_ in expected_keys.items():
            self.assertIn(key, data_info)
            self.assertIsInstance(data_info[key], type_, key)

    def test_get_subset_index(self):
        dataset = self.build_combined_dataset()
        lens = dataset._lens

        with self.assertRaises(ValueError):
            subset_idx, sample_idx = dataset._get_subset_index(sum(lens))

        index = lens[0]
        subset_idx, sample_idx = dataset._get_subset_index(index)
        self.assertEqual(subset_idx, 1)
        self.assertEqual(sample_idx, 0)

        index = -lens[1] - 1
        subset_idx, sample_idx = dataset._get_subset_index(index)
        self.assertEqual(subset_idx, 0)
        self.assertEqual(sample_idx, lens[0] - 1)

    def test_prepare_data(self):
        dataset = self.build_combined_dataset()
        lens = dataset._lens

        data_info = dataset[lens[0]]
        self.check_data_info_keys(data_info)
