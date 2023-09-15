# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import numpy as np

from mmpose.codecs import YOLOXPoseAnnotationProcessor


class TestYOLOXPoseAnnotationProcessor(TestCase):

    def test_encode(self):
        processor = YOLOXPoseAnnotationProcessor(expand_bbox=True)

        keypoints = np.array([[[0, 1], [2, 6], [4, 5]], [[5, 6], [7, 8],
                                                         [8, 9]]])
        keypoints_visible = np.array([[1, 1, 0], [1, 0, 1]])
        bbox = np.array([[0, 1, 3, 4], [1, 2, 5, 6]])
        category_id = [1, 2]

        encoded = processor.encode(keypoints, keypoints_visible, bbox,
                                   category_id)

        self.assertTrue('bbox' in encoded)
        self.assertTrue('bbox_labels' in encoded)
        self.assertTrue(
            np.array_equal(encoded['bbox'],
                           np.array([[0., 1., 3., 6.], [1., 2., 8., 9.]])))
        self.assertTrue(
            np.array_equal(encoded['bbox_labels'], np.array([0, 1])))

    def test_decode(self):
        # make sure the `decode` method has been defined
        processor = YOLOXPoseAnnotationProcessor()
        _ = processor.decode(dict())
