# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose, LoadImageFromFile
from mmengine.utils import is_list_of

from mmpose.datasets.transforms import (Albumentation, GenerateTarget,
                                        GetBBoxCenterScale,
                                        PhotometricDistortion,
                                        RandomBBoxTransform, RandomFlip,
                                        RandomHalfBody, TopdownAffine)
from mmpose.testing import get_coco_sample


class TestGetBBoxCenterScale(TestCase):

    def setUp(self):

        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(480, 640),
            num_instances=1,
            with_bbox_cs=True,
            with_img_mask=True,
            random_keypoints_visible=True)

    def test_transform(self):
        # test converting bbox to center and scale
        padding = 1.25

        transform = GetBBoxCenterScale(padding=padding)
        results = deepcopy(self.data_info)
        results = transform(results)

        center = (results['bbox'][:, :2] + results['bbox'][:, 2:4]) * 0.5
        scale = (results['bbox'][:, 2:4] - results['bbox'][:, :2]) * padding

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
        transform = GetBBoxCenterScale(padding=1.25)
        self.assertEqual(repr(transform), 'GetBBoxCenterScale(padding=1.25)')


class TestRandomFlip(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(480, 640),
            num_instances=1,
            with_bbox_cs=True,
            with_img_mask=True,
            random_keypoints_visible=True)

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
        bbox_center_flipped = self.data_info['bbox_center'].copy()
        bbox_center_flipped[:, 0] = 640 - 1 - bbox_center_flipped[:, 0]

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][:, ::-1]))
        self.assertTrue(
            np.allclose(results['img_mask'],
                        self.data_info['img_mask'][:, ::-1]))
        self.assertTrue(
            np.allclose(results['bbox_center'], bbox_center_flipped))
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
        bbox_center_flipped = self.data_info['bbox_center'].copy()
        bbox_center_flipped[:, 1] = 480 - 1 - bbox_center_flipped[:, 1]

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][::-1]))
        self.assertTrue(
            np.allclose(results['img_mask'], self.data_info['img_mask'][::-1]))
        self.assertTrue(
            np.allclose(results['bbox_center'], bbox_center_flipped))
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
        bbox_center_flipped = self.data_info['bbox_center'].copy()
        bbox_center_flipped[:, 0] = 640 - 1 - bbox_center_flipped[:, 0]
        bbox_center_flipped[:, 1] = 480 - 1 - bbox_center_flipped[:, 1]

        self.assertTrue(
            np.allclose(results['img'], self.data_info['img'][::-1, ::-1]))
        self.assertTrue(
            np.allclose(results['img_mask'],
                        self.data_info['img_mask'][::-1, ::-1]))
        self.assertTrue(
            np.allclose(results['bbox_center'], bbox_center_flipped))
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
        self.data_info = get_coco_sample(
            img_shape=(480, 640),
            num_instances=1,
            with_bbox_cs=True,
            with_img_mask=True)

    def test_transform(self):
        padding = 1.5

        # keep upper body
        transform = RandomHalfBody(
            prob=1.,
            min_total_keypoints=8,
            min_upper_keypoints=2,
            min_lower_keypoints=2)
        results = deepcopy(self.data_info)
        results['keypoints_visible'][:, results['lower_body_ids']] = 0
        results = transform(results)

        kpts = self.data_info['keypoints'][:, self.data_info['upper_body_ids']]
        self.assertTrue(np.allclose(results['bbox_center'], kpts.mean(axis=1)))
        self.assertTrue(
            np.allclose(results['bbox_scale'],
                        (kpts.max(axis=1) - kpts.min(axis=1)) * padding))

        # keep lower body
        transform = RandomHalfBody(
            prob=1.,
            min_total_keypoints=6,
            min_upper_keypoints=4,
            min_lower_keypoints=4)
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

        # no transform due to insufficient valid total keypoints
        transform = RandomHalfBody(
            prob=1.,
            min_total_keypoints=8,
            min_upper_keypoints=2,
            min_lower_keypoints=2)
        results = deepcopy(self.data_info)
        results['keypoints_visible'].fill(0)
        results = transform(results)

        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))

        # no transform due to insufficient valid half-body keypoints
        transform = RandomHalfBody(
            prob=1.,
            min_total_keypoints=4,
            min_upper_keypoints=3,
            min_lower_keypoints=3)
        results = deepcopy(self.data_info)
        results['keypoints_visible'][:, results['upper_body_ids'][2:]] = 0
        results['keypoints_visible'][:, results['lower_body_ids'][2:]] = 0
        results = transform(results)

        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))

    def test_repr(self):
        transform = RandomHalfBody(
            min_total_keypoints=8,
            min_upper_keypoints=2,
            min_lower_keypoints=2,
            padding=1.5,
            prob=0.3,
            upper_prioritized_prob=0.7)
        self.assertEqual(
            repr(transform),
            'RandomHalfBody(min_total_keypoints=8, min_upper_keypoints=2, '
            'min_lower_keypoints=2, padding=1.5, prob=0.3, '
            'upper_prioritized_prob=0.7)')


class TestRandomBBoxTransform(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(480, 640),
            num_instances=1,
            with_bbox_cs=True,
            with_img_mask=True)

    def test_transform(self):
        shfit_factor = 0.16
        scale_factor = (0.5, 1.5)
        rotate_factor = 90.

        # test random shift
        transform = RandomBBoxTransform(
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
        self.assertTrue(
            np.allclose(results['bbox_rotation'], np.zeros((1, 17))))

        # test random resizing
        transform = RandomBBoxTransform(
            scale_factor=scale_factor,
            shift_prob=0.0,
            scale_prob=1.0,
            rotate_prob=0.0)

        results = transform(deepcopy(self.data_info))
        center = self.data_info['bbox_center']
        scale = self.data_info['bbox_scale']
        scale_range = [scale * scale_factor[0], scale * scale_factor[1]]

        self.assertTrue(np.allclose(results['bbox_center'], center))
        self.assertFalse(np.allclose(results['bbox_scale'], scale))
        self.assertTrue(((results['bbox_scale'] > scale_range[0]) &
                         (results['bbox_scale'] < scale_range[1])).all())
        self.assertTrue(
            np.allclose(results['bbox_rotation'], np.zeros((1, 17))))

        # test random rotation
        transform = RandomBBoxTransform(
            rotate_factor=rotate_factor,
            shift_prob=0.0,
            scale_prob=0.0,
            rotate_prob=1.0)

        results = transform(deepcopy(self.data_info))
        rotation_range = [-rotate_factor, rotate_factor]
        bbox_rotation_min = np.full((1, 17), rotation_range[0])
        bbox_rotation_max = np.full((1, 17), rotation_range[1])

        self.assertTrue(
            np.allclose(results['bbox_center'], self.data_info['bbox_center']))
        self.assertTrue(
            np.allclose(results['bbox_scale'], self.data_info['bbox_scale']))
        self.assertFalse(np.allclose(results['bbox_rotation'], 0))
        self.assertTrue(((results['bbox_rotation'] > bbox_rotation_min) &
                         (results['bbox_rotation'] < bbox_rotation_max)).all())

        # test hybrid transform
        transform = RandomBBoxTransform(
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
        scale_range = [scale * scale_factor[0], scale * scale_factor[1]]
        rotation_range = [-rotate_factor, rotate_factor]

        self.assertFalse(np.allclose(results['bbox_center'], center))
        self.assertTrue(((results['bbox_center'] > center_range[0]) &
                         (results['bbox_center'] < center_range[1])).all())
        self.assertFalse(np.allclose(results['bbox_scale'], scale))
        self.assertTrue(((results['bbox_scale'] > scale_range[0]) &
                         (results['bbox_scale'] < scale_range[1])).all())
        self.assertFalse(np.allclose(results['bbox_rotation'], 0))
        self.assertTrue(((results['bbox_rotation'] > rotation_range[0]) &
                         (results['bbox_rotation'] < rotation_range[1])).all())

    def test_repr(self):
        transform = RandomBBoxTransform(
            shift_factor=0.16,
            shift_prob=0.3,
            scale_factor=0.5,
            scale_prob=1.0,
            rotate_factor=40.0,
            rotate_prob=0.6)

        self.assertEqual(
            repr(transform),
            'RandomBBoxTransform(shift_prob=0.3, shift_factor=0.16, '
            'scale_prob=1.0, scale_factor=0.5, rotate_prob=0.6, '
            'rotate_factor=40.0)')


class TestAlbumentation(TestCase):

    def setUp(self):
        """Setup the valiables which are used in each test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = 'tests/data/coco'
        results = dict(img_path=osp.join(data_prefix, '000000000785.jpg'))
        load = LoadImageFromFile()
        self.results = load(copy.deepcopy(results))

    def test_transform(self):
        # test when ``keymap`` is None
        transform = Albumentation(transforms=[
            dict(type='RandomBrightnessContrast', p=0.2),
            dict(type='ToFloat')
        ])
        results_update = transform(copy.deepcopy(self.results))
        self.assertEqual(results_update['img'].dtype, np.float32)

    def test_repr(self):
        # test when ``keymap`` is not None
        transforms = [
            dict(type='RandomBrightnessContrast', p=0.2),
            dict(type='ToFloat')
        ]
        transform = Albumentation(
            transforms=transforms, keymap={'img': 'image'})
        self.assertEqual(
            repr(transform), f'Albumentation(transforms={transforms})')


class TestPhotometricDistortion(TestCase):

    def setUp(self):
        """Setup the valiables which are used in each test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = 'tests/data/coco'
        results = dict(img_path=osp.join(data_prefix, '000000000785.jpg'))
        load = LoadImageFromFile()
        self.results = load(copy.deepcopy(results))

    def test_transform(self):
        transform = PhotometricDistortion()
        results_update = transform(copy.deepcopy(self.results))
        self.assertEqual(results_update['img'].dtype, np.uint8)

    def test_repr(self):
        transform = PhotometricDistortion()
        self.assertEqual(
            repr(transform), ('PhotometricDistortion'
                              '(brightness_delta=32, '
                              'contrast_range=(0.5, 1.5), '
                              'saturation_range=(0.5, 1.5), '
                              'hue_delta=18)'))


class TestGenerateTarget(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(480, 640),
            num_instances=1,
            with_bbox_cs=True,
            with_img_mask=True)

    def test_generate_single_target(self):
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)

        # generate heatmap
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(encoder=encoder)
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertTrue(
            np.allclose(results['keypoint_weights'], np.ones((1, 17))))

        # generate heatmap and use meta keypoint weights
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(
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

    def test_generate_multilevel_target(self):
        encoder_0 = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)
        encoder_1 = dict(encoder_0, heatmap_size=(24, 32))

        # generate multilevel heatmap
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(
                encoder=[encoder_0, encoder_1],
                multilevel=True,
                use_dataset_keypoint_weights=True)
        ])
        results = pipeline(deepcopy(self.data_info))

        self.assertTrue(is_list_of(results['heatmaps'], np.ndarray))
        self.assertTrue(is_list_of(results['keypoint_weights'], np.ndarray))
        self.assertEqual(results['heatmaps'][0].shape, (17, 64, 48))
        self.assertEqual(results['heatmaps'][1].shape, (17, 32, 24))
        self.assertEqual(results['keypoint_weights'][0].shape, (1, 17))

    def test_generate_combined_target(self):
        encoder_0 = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)
        encoder_1 = dict(type='RegressionLabel', input_size=(192, 256))
        # generate multilevel heatmap
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(
                encoder=[encoder_0, encoder_1],
                multilevel=False,
                use_dataset_keypoint_weights=True)
        ])

        results = pipeline(deepcopy(self.data_info))

        self.assertEqual(results['heatmaps'].shape, (17, 64, 48))
        self.assertEqual(results['keypoint_labels'].shape, (1, 17, 2))
        self.assertIsInstance(results['keypoint_weights'], list)
        self.assertEqual(results['keypoint_weights'][0].shape, (1, 17))

    def test_errors(self):

        # single encoder with `multilevel=True`
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)

        with self.assertRaisesRegex(AssertionError,
                                    'Need multiple encoder configs'):
            _ = GenerateTarget(encoder=encoder, multilevel=True)

        # diverse keys in multilevel encoding
        encoder_0 = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)

        encoder_1 = dict(type='RegressionLabel', input_size=(192, 256))
        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(encoder=[encoder_0, encoder_1], multilevel=True)
        ])

        with self.assertRaisesRegex(ValueError, 'have the same keys'):
            _ = pipeline(deepcopy(self.data_info))

        # overlapping keys in combined encoding
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)

        pipeline = Compose([
            TopdownAffine(input_size=(192, 256)),
            GenerateTarget(encoder=[encoder, encoder], multilevel=False)
        ])

        with self.assertRaisesRegex(ValueError, 'Overlapping item'):
            _ = pipeline(deepcopy(self.data_info))

        # deprecated argument `target_type` is given
        encoder = dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2.0)

        with self.assertWarnsRegex(DeprecationWarning,
                                   '`target_type` is deprecated'):
            _ = GenerateTarget(encoder=encoder, target_type='heatmap')
