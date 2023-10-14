# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose

from mmpose.datasets.transforms import (BottomupGetHeatmapMask,
                                        BottomupRandomAffine,
                                        BottomupRandomChoiceResize,
                                        BottomupRandomCrop, BottomupResize,
                                        RandomFlip)
from mmpose.testing import get_coco_sample


class TestBottomupRandomAffine(TestCase):

    def setUp(self):
        # prepare dummy bottom-up data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(240, 320), num_instances=4, with_bbox_cs=True)

    def test_transform(self):

        # without UDP
        transform = BottomupRandomAffine(input_size=(512, 512), use_udp=False)
        results = transform(deepcopy(self.data_info))

        self.assertEqual(results['img'].shape, (512, 512, 3))
        self.assertEqual(results['input_size'], (512, 512))
        self.assertIn('warp_mat', results)

        # with UDP
        transform = BottomupRandomAffine(input_size=(512, 512), use_udp=True)
        results = transform(deepcopy(self.data_info))

        self.assertEqual(results['img'].shape, (512, 512, 3))
        self.assertEqual(results['input_size'], (512, 512))
        self.assertIn('warp_mat', results)


class TestBottomupGetHeatmapMask(TestCase):

    def setUp(self):
        # prepare dummy bottom-up data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(240, 320), num_instances=4, with_bbox_cs=True)

    def test_transform(self):

        # single-scale heatmap mask
        pipeline = Compose([
            BottomupRandomAffine(input_size=(512, 512)),
            RandomFlip(prob=1.0, direction='horizontal'),
            BottomupGetHeatmapMask()
        ])

        results = deepcopy(self.data_info)
        results['heatmaps'] = np.random.rand(17, 64, 64).astype(np.float32)
        results = pipeline(results)

        self.assertEqual(results['heatmap_mask'].shape, (64, 64))
        self.assertTrue(results['heatmap_mask'].dtype, np.uint8)

        # multi-scale heatmap mask
        pipeline = Compose([
            BottomupRandomAffine(input_size=(512, 512)),
            RandomFlip(prob=1.0, direction='horizontal'),
            BottomupGetHeatmapMask()
        ])

        results = deepcopy(self.data_info)
        heatmap_sizes = [(64, 64), (32, 32)]
        results['heatmaps'] = [
            np.random.rand(17, h, w).astype(np.float32)
            for w, h in heatmap_sizes
        ]
        results = pipeline(results)

        self.assertIsInstance(results['heatmap_mask'], list)
        for i, sizes in enumerate(heatmap_sizes):
            mask = results['heatmap_mask'][i]
            self.assertEqual(mask.shape, sizes)
            self.assertTrue(mask.dtype, np.uint8)

        # no heatmap
        pipeline = Compose([
            BottomupRandomAffine(input_size=(512, 512)),
            RandomFlip(prob=1.0, direction='horizontal'),
            BottomupGetHeatmapMask()
        ])

        results = deepcopy(self.data_info)
        results = pipeline(results)

        self.assertEqual(results['heatmap_mask'].shape, (512, 512))
        self.assertTrue(results['heatmap_mask'].dtype, np.uint8)


class TestBottomupResize(TestCase):

    def setUp(self):
        # prepare dummy bottom-up data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(240, 480),
            img_fill=255,
            num_instances=4,
            with_bbox_cs=True)

    def test_transform(self):

        # single-scale, fit
        transform = BottomupResize(input_size=(256, 256), resize_mode='fit')
        results = transform(deepcopy(self.data_info))
        # the middle section of the image is the resized content, while the
        # top and bottom are padded with zeros
        self.assertEqual(results['img'].shape, (256, 256, 3))
        self.assertTrue(
            np.allclose(results['input_scale'], np.array([480., 480.])))
        self.assertTrue(
            np.allclose(results['input_center'], np.array([240., 120.])))
        self.assertTrue(np.all(results['img'][64:192] > 0))
        self.assertTrue(np.all(results['img'][:64] == 0))
        self.assertTrue(np.all(results['img'][192:] == 0))

        # single-scale, expand
        transform = BottomupResize(input_size=(256, 256), resize_mode='expand')
        results = transform(deepcopy(self.data_info))
        # the actual input size is expanded to (512, 256) according to the
        # original image shape
        self.assertEqual(results['img'].shape, (256, 512, 3))
        self.assertTrue(np.all(results['img'] > 0))

        # single-scale, expand, size_factor=100
        transform = BottomupResize(
            input_size=(256, 256), resize_mode='expand', size_factor=100)
        results = transform(deepcopy(self.data_info))
        # input size is ceiled from (512, 256) to (600, 300)
        self.assertEqual(results['img'].shape, (300, 600, 3))

        # multi-scale
        transform = BottomupResize(
            input_size=(256, 256), aug_scales=[1.5], resize_mode='fit')
        results = transform(deepcopy(self.data_info))
        self.assertIsInstance(results['img'], list)
        self.assertIsInstance(results['input_center'], np.ndarray)
        self.assertIsInstance(results['input_scale'], np.ndarray)
        self.assertEqual(results['img'][0].shape, (256, 256, 3))
        self.assertEqual(results['img'][1].shape, (384, 384, 3))


class TestBottomupRandomCrop(TestCase):

    def setUp(self):
        # test invalid crop_type
        with self.assertRaisesRegex(ValueError, 'Invalid crop_type'):
            BottomupRandomCrop(crop_size=(10, 10), crop_type='unknown')

        crop_type_list = ['absolute', 'absolute_range']
        for crop_type in crop_type_list:
            # test h > 0 and w > 0
            for crop_size in [(0, 0), (0, 1), (1, 0)]:
                with self.assertRaises(AssertionError):
                    BottomupRandomCrop(
                        crop_size=crop_size, crop_type=crop_type)
            # test type(h) = int and type(w) = int
            for crop_size in [(1.0, 1), (1, 1.0), (1.0, 1.0)]:
                with self.assertRaises(AssertionError):
                    BottomupRandomCrop(
                        crop_size=crop_size, crop_type=crop_type)

        # test crop_size[0] <= crop_size[1]
        with self.assertRaises(AssertionError):
            BottomupRandomCrop(crop_size=(10, 5), crop_type='absolute_range')

        # test h in (0, 1] and w in (0, 1]
        crop_type_list = ['relative_range', 'relative']
        for crop_type in crop_type_list:
            for crop_size in [(0, 1), (1, 0), (1.1, 0.5), (0.5, 1.1)]:
                with self.assertRaises(AssertionError):
                    BottomupRandomCrop(
                        crop_size=crop_size, crop_type=crop_type)

        self.data_info = get_coco_sample(img_shape=(24, 32))

    def test_transform(self):
        # test relative and absolute crop
        src_results = self.data_info
        target_shape = (12, 16)
        for crop_type, crop_size in zip(['relative', 'absolute'], [(0.5, 0.5),
                                                                   (16, 12)]):
            transform = BottomupRandomCrop(
                crop_size=crop_size, crop_type=crop_type)
            results = transform(deepcopy(src_results))
            self.assertEqual(results['img'].shape[:2], target_shape)

        # test absolute_range crop
        transform = BottomupRandomCrop(
            crop_size=(10, 20), crop_type='absolute_range')
        results = transform(deepcopy(src_results))
        h, w = results['img'].shape[:2]
        self.assertTrue(10 <= w <= 20)
        self.assertTrue(10 <= h <= 20)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])
        # test relative_range crop
        transform = BottomupRandomCrop(
            crop_size=(0.5, 0.5), crop_type='relative_range')
        results = transform(deepcopy(src_results))
        h, w = results['img'].shape[:2]
        self.assertTrue(16 <= w <= 32)
        self.assertTrue(12 <= h <= 24)
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

        # test with keypoints, bbox, segmentation
        src_results = get_coco_sample(img_shape=(10, 10), num_instances=2)
        segmentation = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        keypoints = np.ones_like(src_results['keypoints']) * 5
        src_results['segmentation'] = segmentation
        src_results['keypoints'] = keypoints
        transform = BottomupRandomCrop(
            crop_size=(7, 5),
            allow_negative_crop=False,
            recompute_bbox=False,
            bbox_clip_border=True)
        results = transform(deepcopy(src_results))
        h, w = results['img'].shape[:2]
        self.assertEqual(h, 5)
        self.assertEqual(w, 7)
        self.assertEqual(results['bbox'].shape[0], 2)
        self.assertTrue(results['keypoints_visible'].all())
        self.assertTupleEqual(results['segmentation'].shape[:2], (5, 7))
        self.assertEqual(results['img_shape'], results['img'].shape[:2])

        # test bbox_clip_border = False
        transform = BottomupRandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False)
        results = transform(deepcopy(src_results))
        self.assertTrue((results['bbox'] == src_results['bbox']).all())

        # test the crop does not contain any gt-bbox
        # allow_negative_crop = False
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        bbox = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'bbox': bbox}
        transform = BottomupRandomCrop(
            crop_size=(5, 3), allow_negative_crop=False)
        results = transform(deepcopy(src_results))
        self.assertIsNone(results)

        # allow_negative_crop = True
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        bbox = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'bbox': bbox}
        transform = BottomupRandomCrop(
            crop_size=(5, 3), allow_negative_crop=True)
        results = transform(deepcopy(src_results))
        self.assertTrue(isinstance(results, dict))


class TestBottomupRandomChoiceResize(TestCase):

    def setUp(self):
        self.data_info = get_coco_sample(img_shape=(300, 400))

    def test_transform(self):
        results = dict()
        # test with one scale
        transform = BottomupRandomChoiceResize(scales=[(1333, 800)])
        results = deepcopy(self.data_info)
        results = transform(results)
        self.assertEqual(results['img'].shape, (800, 1333, 3))

        # test with multi scales
        _scale_choice = [(1333, 800), (1333, 600)]
        transform = BottomupRandomChoiceResize(scales=_scale_choice)
        results = deepcopy(self.data_info)
        results = transform(results)
        self.assertIn((results['img'].shape[1], results['img'].shape[0]),
                      _scale_choice)

        # test keep_ratio
        transform = BottomupRandomChoiceResize(
            scales=[(900, 600)], resize_type='Resize', keep_ratio=True)
        results = deepcopy(self.data_info)
        _input_ratio = results['img'].shape[0] / results['img'].shape[1]
        results = transform(results)
        _output_ratio = results['img'].shape[0] / results['img'].shape[1]
        self.assertLess(abs(_input_ratio - _output_ratio), 1.5 * 1e-3)

        # test clip_object_border
        bbox = [[200, 150, 600, 450]]
        transform = BottomupRandomChoiceResize(
            scales=[(200, 150)], resize_type='Resize', clip_object_border=True)
        results = deepcopy(self.data_info)
        results['bbox'] = np.array(bbox)
        results = transform(results)
        self.assertEqual(results['img'].shape, (150, 200, 3))
        self.assertTrue((results['bbox'] == np.array([[100, 75, 200,
                                                       150]])).all())

        transform = BottomupRandomChoiceResize(
            scales=[(200, 150)],
            resize_type='Resize',
            clip_object_border=False)
        results = self.data_info
        results['bbox'] = np.array(bbox)
        results = transform(results)
        assert results['img'].shape == (150, 200, 3)
        assert np.equal(results['bbox'], np.array([[100, 75, 300, 225]])).all()
