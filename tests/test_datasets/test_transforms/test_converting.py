# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmpose.datasets.transforms import KeypointConverter
from mmpose.testing import get_coco_sample


class TestKeypointConverter(TestCase):

    def setUp(self):
        # prepare dummy bottom-up data sample with COCO metainfo
        self.data_info = get_coco_sample(
            img_shape=(240, 320), num_instances=4, with_bbox_cs=True)

    def test_transform(self):
        # 1-to-1 mapping
        mapping = [(3, 0), (6, 1), (16, 2), (5, 3)]
        transform = KeypointConverter(num_keypoints=5, mapping=mapping)
        results = transform(self.data_info.copy())

        # check shape
        self.assertEqual(results['keypoints'].shape[0],
                         self.data_info['keypoints'].shape[0])
        self.assertEqual(results['keypoints'].shape[1], 5)
        self.assertEqual(results['keypoints'].shape[2], 2)
        self.assertEqual(results['keypoints_visible'].shape[0],
                         self.data_info['keypoints_visible'].shape[0])
        self.assertEqual(results['keypoints_visible'].shape[1], 5)

        # check value
        for source_index, target_index in mapping:
            self.assertTrue((results['keypoints'][:, target_index] ==
                             self.data_info['keypoints'][:,
                                                         source_index]).all())
            self.assertTrue(
                (results['keypoints_visible'][:, target_index] ==
                 self.data_info['keypoints_visible'][:, source_index]).all())

        # 2-to-1 mapping
        mapping = [((3, 5), 0), (6, 1), (16, 2), (5, 3)]
        transform = KeypointConverter(num_keypoints=5, mapping=mapping)
        results = transform(self.data_info.copy())

        # check shape
        self.assertEqual(results['keypoints'].shape[0],
                         self.data_info['keypoints'].shape[0])
        self.assertEqual(results['keypoints'].shape[1], 5)
        self.assertEqual(results['keypoints'].shape[2], 2)
        self.assertEqual(results['keypoints_visible'].shape[0],
                         self.data_info['keypoints_visible'].shape[0])
        self.assertEqual(results['keypoints_visible'].shape[1], 5)

        # check value
        for source_index, target_index in mapping:
            if isinstance(source_index, tuple):
                source_index, source_index2 = source_index
                self.assertTrue(
                    (results['keypoints'][:, target_index] == 0.5 *
                     (self.data_info['keypoints'][:, source_index] +
                      self.data_info['keypoints'][:, source_index2])).all())
                self.assertTrue(
                    (results['keypoints_visible'][:, target_index] ==
                     self.data_info['keypoints_visible'][:, source_index] *
                     self.data_info['keypoints_visible'][:,
                                                         source_index2]).all())
            else:
                self.assertTrue(
                    (results['keypoints'][:, target_index] ==
                     self.data_info['keypoints'][:, source_index]).all())
                self.assertTrue(
                    (results['keypoints_visible'][:, target_index] ==
                     self.data_info['keypoints_visible'][:,
                                                         source_index]).all())
