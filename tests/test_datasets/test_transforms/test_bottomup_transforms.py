# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose

from mmpose.datasets.transforms import (BottomupGetHeatmapMask,
                                        BottomupRandomAffine, BottomupResize,
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
