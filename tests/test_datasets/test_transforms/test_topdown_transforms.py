# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

from mmpose.datasets.transforms import TopdownAffine
from mmpose.testing import get_coco_sample


class TestTopdownAffine(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(num_instances=1, with_bbox_cs=True)

    def test_transform(self):
        # without udp
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))
        self.assertIn('transformed_keypoints', results)

        # with udp
        transform = TopdownAffine(input_size=(192, 256), use_udp=True)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))
        self.assertIn('transformed_keypoints', results)

    def test_repr(self):
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        self.assertEqual(
            repr(transform),
            'TopdownAffine(input_size=(192, 256), use_udp=False)')
