# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np

from mmpose.datasets.pipelines2 import (GetBboxCenterScale,
                                        RandomBboxTransform, RandomFlip,
                                        RandomHalfBody)


class TestGetBboxCenterScale(TestCase):

    def setUp(self):

        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.full((1, 17, 1), 2).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        # test converting bbox to center and scale
        padding = 1.25

        transform = GetBboxCenterScale(padding=padding)
        results = deepcopy(self.data_info)
        results = transform(results)

        center = results['bbox'][:, :2] + results['bbox'][:, 2:4] * 0.5
        scale = results['bbox'][:, 2:4] * padding

        self.assertTrue(np.allclose(results['bbox_center'], center))
        self.assertTrue(np.allclose(results['bbox_scale'], scale))

        # test using existing bbox center and scale
        results = deepcopy(self.data_info)
        center = np.random.rand(1, 2).astype(np.float32)
        scale = np.random.rand(1, 2).astype(np.float32)
        results.update(bbox_center=center, bbox_scale=scale)
        results = transform(results)
        self.assertTrue(np.allclose(results['bbox_center'], center))
        self.assertTrue(np.allclose(results['bbox_scale'], scale))

    def test_repr(self):
        transform = GetBboxCenterScale(padding=1.25)
        self.assertEqual(repr(transform), 'GetBboxCenterScale(padding=1.25)')


class TestRandomFlip(TestCase):

    def setUp(self):

        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.full((1, 17, 1), 2).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

        # assign different visibility for keypoints in symmetric pairs
        for idx, _ in self.data_info['flip_pairs']:
            self.data_info['keypoints_visible'][0, idx] = 0

    def test_init(self):
        # prob: float, direction: str
        _ = RandomFlip(prob=0.5, direction='horizontal')

        # prob: float, direction: list
        _ = RandomFlip(prob=0.5, direction=['horizontal', 'vertical'])

        # prob: list, direction: list
        _ = RandomFlip(prob=[0.3, 0.3], direction=['horizontal', 'vertical'])

    def test_transform(self):
        # test horizontal flip
        transform = RandomFlip(prob=1., direction='horizontal')
        results = deepcopy(self.data_info)
        results = transform(results)

        ids1, ids2 = zip(*(self.data_info['flip_pairs']))
        kpts1 = self.data_info['keypoints'][:, ids1]
        kpts1_vis = self.data_info['keypoints_visible'][:, ids1]
        kpts2 = results['keypoints'][:, ids2]
        kpts2_vis = results['keypoints_visible'][:, ids2]

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][:, ::-1]))
        self.assertTrue(np.allclose(results['bbox_center'], [[589., 50.]]))
        self.assertTrue(np.allclose(kpts1[..., 0], 640 - kpts2[..., 0] - 1))
        self.assertTrue(np.allclose(kpts1[..., 1], kpts2[..., 1]))
        self.assertTrue(np.allclose(kpts1_vis, kpts2_vis))

        # test vertical flip
        transform = RandomFlip(prob=1., direction='vertical')
        results = deepcopy(self.data_info)
        results = transform(results)

        ids1, ids2 = zip(*(self.data_info['flip_pairs']))
        kpts1 = self.data_info['keypoints'][:, ids1]
        kpts1_vis = self.data_info['keypoints_visible'][:, ids1]
        kpts2 = results['keypoints'][:, ids2]
        kpts2_vis = results['keypoints_visible'][:, ids2]

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][::-1]))
        self.assertTrue(np.allclose(results['bbox_center'], [[50., 429.]]))
        self.assertTrue(np.allclose(kpts1[..., 0], kpts2[..., 0]))
        self.assertTrue(np.allclose(kpts1[..., 1], 480 - kpts2[..., 1] - 1))
        self.assertTrue(np.allclose(kpts1_vis, kpts2_vis))

        # test diagonal flip
        transform = RandomFlip(prob=1., direction='diagonal')
        results = deepcopy(self.data_info)
        results = transform(results)

        kpts1 = self.data_info['keypoints']
        kpts1_vis = self.data_info['keypoints_visible']
        kpts2 = results['keypoints']
        kpts2_vis = results['keypoints_visible']

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][::-1, ::-1]))
        self.assertTrue(np.allclose(results['bbox_center'], [[589., 429.]]))
        self.assertTrue(np.allclose(kpts1[..., 0], 640 - kpts2[..., 0] - 1))
        self.assertTrue(np.allclose(kpts1[..., 1], 480 - kpts2[..., 1] - 1))
        self.assertTrue(np.allclose(kpts1_vis, kpts2_vis))

    def test_errors(self):
        # invalid arguments
        with self.assertRaisesRegex(ValueError,
                                    'probs must be float or list of float'):
            _ = RandomFlip(prob=None)

        with self.assertRaisesRegex(
                ValueError, 'direction must be either str or list of str'):
            _ = RandomFlip(direction=None)

        with self.assertRaises(AssertionError):
            _ = RandomFlip(prob=2.0)

        with self.assertRaises(AssertionError):
            _ = RandomFlip(direction='invalid_direction')

    def test_repr(self):
        transform = RandomFlip(prob=0.5, direction='horizontal')
        self.assertEqual(
            repr(transform), 'RandomFlip(prob=0.5, direction=horizontal)')


class TestRandomHalfBody(TestCase):

    def setUp(self):

        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.full((1, 17, 1), 2).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        padding = 1.5

        # keep upper body
        transform = RandomHalfBody(prob=1.)
        results = deepcopy(self.data_info)
        results['keypoints_visible'][:, results['lower_body_ids']] = 0
        results = transform(results)

        kpts = self.data_info['keypoints'][:, self.data_info['upper_body_ids']]
        self.assertTrue(np.allclose(results['bbox_center'], kpts.mean(axis=1)))
        self.assertTrue(
            np.allclose(results['bbox_scale'],
                        (kpts.max(axis=1) - kpts.min(axis=1)) * padding))

        # keep lower body
        transform = RandomHalfBody(prob=1.)
        results = deepcopy(self.data_info)
        results['keypoints_visible'][:, results['upper_body_ids']] = 0
        results = transform(results)

        kpts = self.data_info['keypoints'][:, self.data_info['lower_body_ids']]
        self.assertTrue(np.allclose(results['bbox_center'], kpts.mean(axis=1)))
        self.assertTrue(
            np.allclose(results['bbox_scale'],
                        (kpts.max(axis=1) - kpts.min(axis=1)) * padding))

        # no transform due to prob
        transform = RandomHalfBody(prob=0.)
        results = transform(deepcopy(self.data_info))

        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))

        # no transform due to insufficient valid keypoints
        transform = RandomHalfBody(prob=1.)
        results = deepcopy(self.data_info)
        results['keypoints_visible'].fill(0)
        results = transform(results)

        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))

    def test_repr(self):
        transform = RandomHalfBody(
            min_total_keypoints=8, min_half_keypoints=2, padding=1.5, prob=0.3)
        self.assertEqual(
            repr(transform),
            'RandomHalfBody(min_total_keypoints=8, min_half_keypoints=2, '
            'padding=1.5, prob=0.3)')


class TestRandomBboxTransform(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = dict(
            img=np.zeros((480, 640, 3), dtype=np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[0, 0, 100, 100]], dtype=np.float32),
            bbox_center=np.array([[50, 50]], dtype=np.float32),
            bbox_scale=np.array([[125, 125]], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(0, 100, (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.full((1, 17, 1), 2).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

    def test_transform(self):
        shfit_factor = 0.16
        scale_factor = 0.5
        rotate_factor = 40.

        # test random shift
        transform = RandomBboxTransform(
            shift_factor=shfit_factor,
            shift_prob=1.0,
            scale_prob=0.0,
            rotate_prob=0.0)
        results = transform(deepcopy(self.data_info))

        center = self.data_info['bbox_center']
        scale = self.data_info['bbox_scale']
        center_range = [
            center - scale * shfit_factor,
            center + scale * shfit_factor,
        ]

        self.assertFalse(np.allclose(results['bbox_center'], center))
        self.assertTrue(((results['bbox_center'] > center_range[0]) &
                         (results['bbox_center'] < center_range[1])).all())
        self.assertTrue(np.allclose(results['bbox_scale'], scale))
        self.assertTrue(np.allclose(results['bbox_rotation'], 0.))

        # test random resizing
        transform = RandomBboxTransform(
            scale_factor=scale_factor,
            shift_prob=0.0,
            scale_prob=1.0,
            rotate_prob=0.0)

        results = transform(deepcopy(self.data_info))
        center = self.data_info['bbox_center']
        scale = self.data_info['bbox_scale']
        scale_range = [scale * (1 - scale_factor), scale * (1 + scale_factor)]

        self.assertTrue(np.allclose(results['bbox_center'], center))
        self.assertFalse(np.allclose(results['bbox_scale'], scale))
        self.assertTrue(((results['bbox_scale'] > scale_range[0]) &
                         (results['bbox_scale'] < scale_range[1])).all())
        self.assertTrue(np.allclose(results['bbox_rotation'], 0.))

        # test random rotation
        transform = RandomBboxTransform(
            rotate_factor=rotate_factor,
            shift_prob=0.0,
            scale_prob=0.0,
            rotate_prob=1.0)

        results = transform(deepcopy(self.data_info))
        rotation_range = [-2 * rotate_factor, 2 * rotate_factor]
        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))
        self.assertFalse(np.allclose(results['bbox_rotation'], 0))
        self.assertTrue(((results['bbox_rotation'] > rotation_range[0]) &
                         (results['bbox_rotation'] < rotation_range[1])).all())

        # test hybrid transform
        transform = RandomBboxTransform(
            shift_factor=shfit_factor,
            scale_factor=scale_factor,
            rotate_factor=rotate_factor,
            shift_prob=1.0,
            scale_prob=1.0,
            rotate_prob=1.0)

        results = transform(deepcopy(self.data_info))
        center = self.data_info['bbox_center']
        scale = self.data_info['bbox_scale']

        center_range = [
            center - scale * shfit_factor,
            center + scale * shfit_factor,
        ]
        scale_range = [scale * (1 - scale_factor), scale * (1 + scale_factor)]
        rotation_range = [-2 * rotate_factor, 2 * rotate_factor]

        self.assertFalse(np.allclose(results['bbox_center'], center))
        self.assertTrue(((results['bbox_center'] > center_range[0]) &
                         (results['bbox_center'] < center_range[1])).all())
        self.assertFalse(np.allclose(results['bbox_scale'], scale))
        self.assertTrue(((results['bbox_scale'] > scale_range[0]) &
                         (results['bbox_scale'] < scale_range[1])).all())
        self.assertFalse(np.allclose(results['bbox_rotation'], 0))
        self.assertTrue(((results['bbox_rotation'] > rotation_range[0]) &
                         (results['bbox_rotation'] < rotation_range[1])).all())

    def test_errors(self):
        # invalid arguments
        with self.assertRaises(AssertionError):
            _ = RandomBboxTransform(scale_factor=2.)

    def test_repr(self):
        transform = RandomBboxTransform(
            shift_factor=0.16,
            shift_prob=0.3,
            scale_factor=0.5,
            scale_prob=1.0,
            rotate_factor=40.0,
            rotate_prob=0.6)

        self.assertEqual(
            repr(transform),
            'RandomBboxTransform(shift_prob=0.3, shift_factor=0.16, '
            'scale_prob=1.0, scale_factor=0.5, rotate_prob=0.6, '
            'rotate_factor=40.0)')
