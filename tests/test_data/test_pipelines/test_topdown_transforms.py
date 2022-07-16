# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose

from mmpose.datasets.pipelines2 import (TopDownAffine, TopDownGenerateHeatmap,
                                        TopDownGenerateRegressionLabel)


class TestTopDownAffine(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_rotation=np.array([45], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(10, 50, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.ones((1, 17)).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            dataset_dataset_keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        # without udp
        transform = TopDownAffine(input_size=(192, 256), use_udp=False)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))

        # with udp
        transform = TopDownAffine(input_size=(192, 256), use_udp=True)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))

    def test_repr(self):
        transform = TopDownAffine(input_size=(192, 256), use_udp=False)
        self.assertEqual(
            repr(transform),
            'TopDownAffine(input_size=(192, 256), use_udp=False)')


class TestTopDownGenerateHeatmap(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_rotation=np.array([45], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(10, 50, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.ones((1, 17)).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            dataset_keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        # encoding: msra
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='msra',
                sigma=2.0,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # encoding: msra + dark
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='msra',
                sigma=2.0,
                unbiased=True,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # encoding: msra (multi-scale)
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='msra',
                sigma=[2.0, 3.0],
                unbiased=True,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (2, 17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 2, 17))))

        # encoding: megvii
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='megvii',
                kernel_size=(11, 11),
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # encoding: megvii (multi-scale)
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='megvii',
                kernel_size=[(11, 11), (13, 13)],
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (2, 17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 2, 17))))

        # encoding: udp gaussian
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256), use_udp=True),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
                sigma=2.0,
                udp_combined_map=False,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # encoding: udp gaussian (multi-scale)
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256), use_udp=True),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
                sigma=[2.0, 3.0],
                udp_combined_map=False,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (2, 17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 2, 17))))

        # encoding: udp combined
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256), use_udp=True),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
                udp_radius_factor=0.05,
                udp_combined_map=True,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (51, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # encoding: udp combined (multi-scale)
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256), use_udp=True),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
                udp_radius_factor=[0.05, 0.07],
                udp_combined_map=True,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (2, 51, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 2, 17))))

        # test using meta keypoint weights
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='msra',
                sigma=2.0,
                use_dataset_keypoint_weights=True)
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertEqual(results['keypoint_weights'].shape, (1, 17))
        self.assertTrue(
            np.allclose(results['keypoint_weights'],
                        self.data_info['dataset_keypoint_weights'][None]))

    def test_erros(self):
        # invalid encoding method
        with self.assertRaisesRegex(AssertionError, 'Invalid encoding type'):
            _ = TopDownGenerateHeatmap(
                heatmap_size=(48, 64), encoding='invalid encoding')

        # invalid heatmap size
        with self.assertRaisesRegex(AssertionError, 'heatmap_size'):
            _ = TopDownGenerateHeatmap(heatmap_size=(100, 100, 100))

    def test_repr(self):
        transform = TopDownGenerateHeatmap(
            heatmap_size=(48, 64),
            encoding='msra',
            sigma=2,
            unbiased=True,
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopDownGenerateHeatmap(heatmap_size=(48, 64), encoding="msra", '
            'sigma=2, unbiased=True, use_dataset_keypoint_weights=True)')

        transform = TopDownGenerateHeatmap(
            heatmap_size=(48, 64),
            encoding='megvii',
            kernel_size=(11, 11),
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopDownGenerateHeatmap(heatmap_size=(48, 64), encoding="megvii"'
            ', kernel_size=(11, 11), use_dataset_keypoint_weights=True)')

        transform = TopDownGenerateHeatmap(
            heatmap_size=(48, 64),
            encoding='udp',
            sigma=2,
            udp_combined_map=False,
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopDownGenerateHeatmap(heatmap_size=(48, 64), encoding="udp", '
            'combined_map=False, sigma=2, use_dataset_keypoint_weights=True)')

        transform = TopDownGenerateHeatmap(
            heatmap_size=(48, 64),
            encoding='udp',
            udp_combined_map=True,
            udp_radius_factor=0.05,
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopDownGenerateHeatmap(heatmap_size=(48, 64), encoding="udp", '
            'combined_map=True, radius_factor=0.05, '
            'use_dataset_keypoint_weights=True)')


class TestTopDownGenerateRegressionLabel(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_rotation=np.array([45], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(10, 50, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.ones((1, 17)).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            dataset_keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        # test w/o dataset keypoint weights
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateRegressionLabel()
        ])

        results = pipeline(deepcopy(self.data_info))
        self.assertEqual(results['reg_label'].shape, (1, 17, 2))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # test w/ dataset dataset weights
        pipeline = Compose([
            TopDownAffine(input_size=(192, 256)),
            TopDownGenerateRegressionLabel(use_dataset_keypoint_weights=True)
        ])

        results = pipeline(deepcopy(self.data_info))
        self.assertEqual(results['reg_label'].shape, (1, 17, 2))
        self.assertEqual(results['keypoint_weights'].shape, (1, 17))
        self.assertTrue(
            np.allclose(results['keypoint_weights'],
                        self.data_info['dataset_keypoint_weights'][None]))

    def test_repr(self):
        transform = TopDownGenerateRegressionLabel(
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopDownGenerateRegressionLabel(use_dataset_keypoint_weights=True)'
        )
