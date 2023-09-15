# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import numpy as np

from mmpose.datasets.transforms import Mosaic, YOLOXMixUp


class TestMosaic(TestCase):

    def setUp(self):
        # Create a sample data dictionary for testing
        sample_data = {
            'img':
            np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            'bbox': np.random.rand(2, 4),
            'bbox_score': np.random.rand(2, ),
            'category_id': [1, 2],
            'keypoints': np.random.rand(2, 3, 2),
            'keypoints_visible': np.random.rand(2, 3),
            'area': np.random.rand(2, )
        }
        mixed_data_list = [sample_data.copy() for _ in range(3)]
        sample_data.update({'mixed_data_list': mixed_data_list})

        self.sample_data = sample_data

    def test_apply_mix(self):
        mosaic = Mosaic()
        transformed_data = mosaic.apply_mix(self.sample_data)

        # Check if the transformed data has the expected keys
        self.assertTrue('img' in transformed_data)
        self.assertTrue('img_shape' in transformed_data)
        self.assertTrue('bbox' in transformed_data)
        self.assertTrue('category_id' in transformed_data)
        self.assertTrue('bbox_score' in transformed_data)
        self.assertTrue('keypoints' in transformed_data)
        self.assertTrue('keypoints_visible' in transformed_data)
        self.assertTrue('area' in transformed_data)

    def test_create_mosaic_image(self):
        mosaic = Mosaic()
        mosaic_img, annos = mosaic._create_mosaic_image(
            self.sample_data, self.sample_data['mixed_data_list'])

        # Check if the mosaic image and annotations are generated correctly
        self.assertEqual(mosaic_img.shape, (1280, 1280, 3))
        self.assertTrue('bboxes' in annos)
        self.assertTrue('bbox_scores' in annos)
        self.assertTrue('category_id' in annos)
        self.assertTrue('keypoints' in annos)
        self.assertTrue('keypoints_visible' in annos)
        self.assertTrue('area' in annos)

    def test_mosaic_combine(self):
        mosaic = Mosaic()
        center = (320, 240)
        img_shape = (480, 640)
        paste_coord, crop_coord = mosaic._mosaic_combine(
            'top_left', center, img_shape)

        # Check if the coordinates are calculated correctly
        self.assertEqual(paste_coord, (0, 0, 320, 240))
        self.assertEqual(crop_coord, (160, 400, 480, 640))


class TestYOLOXMixUp(TestCase):

    def setUp(self):
        # Create a sample data dictionary for testing
        sample_data = {
            'img':
            np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            'bbox': np.random.rand(2, 4),
            'bbox_score': np.random.rand(2, ),
            'category_id': [1, 2],
            'keypoints': np.random.rand(2, 3, 2),
            'keypoints_visible': np.random.rand(2, 3),
            'area': np.random.rand(2, ),
            'flip_indices': [0, 2, 1]
        }
        mixed_data_list = [sample_data.copy() for _ in range(1)]
        sample_data.update({'mixed_data_list': mixed_data_list})

        self.sample_data = sample_data

    def test_apply_mix(self):
        mixup = YOLOXMixUp()
        transformed_data = mixup.apply_mix(self.sample_data)

        # Check if the transformed data has the expected keys
        self.assertTrue('img' in transformed_data)
        self.assertTrue('img_shape' in transformed_data)
        self.assertTrue('bbox' in transformed_data)
        self.assertTrue('category_id' in transformed_data)
        self.assertTrue('bbox_score' in transformed_data)
        self.assertTrue('keypoints' in transformed_data)
        self.assertTrue('keypoints_visible' in transformed_data)
        self.assertTrue('area' in transformed_data)

    def test_create_mixup_image(self):
        mixup = YOLOXMixUp()
        mixup_img, annos = mixup._create_mixup_image(
            self.sample_data, self.sample_data['mixed_data_list'])

        # Check if the mosaic image and annotations are generated correctly
        self.assertEqual(mixup_img.shape, (480, 640, 3))
        self.assertTrue('bboxes' in annos)
        self.assertTrue('bbox_scores' in annos)
        self.assertTrue('category_id' in annos)
        self.assertTrue('keypoints' in annos)
        self.assertTrue('keypoints_visible' in annos)
        self.assertTrue('area' in annos)
