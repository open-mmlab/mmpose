# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from mim.commands import download


class TestMIM(TestCase):

    def test_download(self):
        with TemporaryDirectory() as tmp_dir:
            ckpts = download(
                'mmpose',
                configs=['td-hm_hrnet-w48_8xb32-210e_coco-256x192'],
                dest_root=tmp_dir)

            self.assertEqual(len(ckpts), 1)
            self.assertIn('td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
                          os.listdir(tmp_dir))
            self.assertIn(ckpts[0], os.listdir(tmp_dir))
