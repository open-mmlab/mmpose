# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from mmcv import imread

from mmpose.datasets.transforms.loading import LoadImage


class TestLoadImage(TestCase):

    def test_load_image(self):

        transform = LoadImage()
        results = dict(img_path='tests/data/coco/000000000785.jpg')

        results = transform(results)

        self.assertIsInstance(results['img'], np.ndarray)

    def test_with_input_image(self):
        transform = LoadImage(to_float32=True)

        img_path = 'tests/data/coco/000000000785.jpg'
        results = dict(
            img_path=img_path, img=imread(img_path).astype(np.uint8))

        results = transform(results)

        self.assertIsInstance(results['img'], np.ndarray)
        self.assertTrue(results['img'].dtype, np.float32)
