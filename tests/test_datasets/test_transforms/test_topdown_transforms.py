# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose

from mmpose.datasets.transforms import (TopdownAffine, TopdownGenerateHeatmap,
                                        TopdownGenerateRegressionLabel)


class TestTopdownAffine(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640),
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
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))

        # with udp
        transform = TopdownAffine(input_size=(192, 256), use_udp=True)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))

    def test_repr(self):
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        self.assertEqual(
            repr(transform),
            'TopdownAffine(input_size=(192, 256), use_udp=False)')


class TestTopdownGenerateHeatmap(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640),
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
        # single-level heatmap
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            TopdownGenerateHeatmap(encoder=encoder)
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # multi-level heatmap
        encoder = [
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.0),
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=3.0)
        ]

        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            TopdownGenerateHeatmap(encoder=encoder)
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (2, 17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 2, 17))))

        # use meta keypoint weights
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            TopdownGenerateHeatmap(
                encoder=encoder,
                use_dataset_keypoint_weights=True,
            )
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertEqual(results['keypoint_weights'].shape, (1, 17))
        self.assertTrue(
            np.allclose(results['keypoint_weights'],
                        self.data_info['dataset_keypoint_weights'][None]))

    def test_repr(self):
        pass


class TestTopdownGenerateRegressionLabel(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640),
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
            TopdownAffine(input_size=(192, 256)),
            TopdownGenerateRegressionLabel()
        ])

        results = pipeline(deepcopy(self.data_info))
        self.assertEqual(results['reg_label'].shape, (1, 17, 2))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # test w/ dataset dataset weights
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            TopdownGenerateRegressionLabel(use_dataset_keypoint_weights=True)
        ])

        results = pipeline(deepcopy(self.data_info))
        self.assertEqual(results['reg_label'].shape, (1, 17, 2))
        self.assertEqual(results['keypoint_weights'].shape, (1, 17))
        self.assertTrue(
            np.allclose(results['keypoint_weights'],
                        self.data_info['dataset_keypoint_weights'][None]))

    def test_repr(self):
        transform = TopdownGenerateRegressionLabel(
            use_dataset_keypoint_weights=True)
        self.assertEqual(
            repr(transform),
            'TopdownGenerateRegressionLabel(use_dataset_keypoint_weights=True)'
        )
