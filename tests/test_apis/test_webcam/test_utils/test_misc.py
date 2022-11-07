# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
import unittest

import mmcv
import numpy as np

from mmpose.apis.webcam.utils.misc import (copy_and_paste, expand_and_clamp,
                                           get_cached_file_path,
                                           get_config_path, is_image_file,
                                           screen_matting)


class TestMISC(unittest.TestCase):

    def test_get_cached_file_path(self):
        url = 'https://user-images.githubusercontent.com/15977946/' \
              '170850839-acc59e26-c6b3-48c9-a9ec-87556edb99ed.jpg'
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_file = get_cached_file_path(
                url, save_dir=tmpdir, file_name='sunglasses.jpg')
            self.assertTrue(os.path.exists(cached_file))
            # check if image is successfully cached
            img = mmcv.imread(cached_file)
            self.assertIsNotNone(img)

    def test_get_config_path(self):
        cfg_path = 'configs/_base_/datasets/coco.py'
        path_in_module = get_config_path(cfg_path, 'mmpose')
        self.assertEqual(cfg_path, path_in_module)

        cfg_path = '_base_/datasets/coco.py'
        with self.assertRaises(FileNotFoundError):
            _ = get_config_path(cfg_path, 'mmpose')

    def test_is_image_file(self):
        self.assertTrue(is_image_file('example.png'))
        self.assertFalse(is_image_file('example.mp4'))

    def test_expand_and_clamp(self):
        img_shape = [125, 125, 3]
        bbox = [0, 0, 40, 40]  # [x1, y1, x2, y2]

        expanded_bbox = expand_and_clamp(bbox, img_shape)
        self.assertListEqual(expanded_bbox, [0, 0, 45, 45])

    def test_screen_matting(self):
        img = np.random.randint(0, 256, size=(100, 100, 3))

        # test with supported colors
        for color in 'gbkw':
            img_mat = screen_matting(img, color=color)
            self.assertEqual(len(img_mat.shape), 2)
            self.assertTupleEqual(img_mat.shape, img.shape[:2])

        # test with unsupported arguments
        with self.assertRaises(ValueError):
            screen_matting(img)

        with self.assertRaises(NotImplementedError):
            screen_matting(img, color='r')

    def test_copy_and_paste(self):
        img = np.random.randint(0, 256, size=(50, 50, 3))
        background_img = np.random.randint(0, 256, size=(200, 200, 3))
        mask = screen_matting(background_img, color='b')

        output_img = copy_and_paste(img, background_img, mask)
        self.assertTupleEqual(output_img.shape, background_img.shape)
