# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from unittest import TestCase

import numpy as np
from mmcv import Config

from mmpose.core.post_processing.temporal_filters import build_filter


class TestTemporalFilter(TestCase):
    cfg_folder = 'configs/_base_/filters'

    def get_filter_input(self,
                         num_frame: int,
                         num_keypoint: int = 17,
                         keypoint_dim: int = 2):
        return np.random.rand(num_frame, num_keypoint,
                              keypoint_dim).astype(np.float32)

    def get_filter_configs(self):
        cfg_files = glob.glob(osp.join(self.cfg_folder, '*.py'))
        for cfg_file in cfg_files:
            cfg = Config.fromfile(cfg_file)
            assert 'filter_cfg' in cfg
            yield cfg.filter_cfg

    def test_temporal_filter(self):
        for filter_cfg in self.get_filter_configs():
            with self.subTest(msg=f'Test {filter_cfg.type}'):
                # Do not load checkpoint in CI
                if 'checkpoint' in filter_cfg:
                    filter_cfg.checkpoint = None
                filter = build_filter(filter_cfg)

                # Test input with single frame
                x = self.get_filter_input(num_frame=1)
                y = filter(x)
                self.assertTrue(isinstance(y, np.ndarray))
                self.assertEqual(x.shape, y.shape)

                # Test input with length > window_size
                window_size = filter.window_size
                x = self.get_filter_input(num_frame=window_size + 1)
                y = filter(x)
                self.assertTrue(isinstance(y, np.ndarray))
                self.assertEqual(x.shape, y.shape)
