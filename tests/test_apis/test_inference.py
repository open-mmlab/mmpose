# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from unittest import TestCase

import torch
from parameterized import parameterized

from mmpose.apis import init_model
from mmpose.utils import register_all_modules


class TestInference(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    @parameterized.expand([('configs/tests/hrnet_w32_coco_256x192.py',
                            ('cpu', 'cuda'))])
    def test_init_model(self, config, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                self.skipTest()
            # test init_model with str path
            _ = init_model(config_file, device=device)

            # test init_model with :obj:`Path`
            _ = init_model(Path(config_file), device=device)

            # test init_detector with undesirable type
            with self.assertRaisesRegex(
                    TypeError, 'config must be a filename or Config object'):
                config_list = [config_file]
                _ = init_model(config_list)
