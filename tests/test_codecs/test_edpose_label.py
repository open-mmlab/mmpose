# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import numpy as np

from mmpose.codecs import EDPoseLabel


class TestEDPoseLabel(TestCase):

    def setUp(self):
        self.encoder = EDPoseLabel(num_select=2, num_keypoints=2)
        self.img_shape = (640, 480)
        self.keypoints = np.array([[[100, 50], [200, 50]],
                                   [[300, 400], [100, 200]]])
        self.area = np.array([5000, 8000])

    def test_encode(self):
        # Test encoding
        encoded_data = self.encoder.encode(
            img_shape=self.img_shape, keypoints=self.keypoints, area=self.area)

        self.assertEqual(encoded_data['keypoints'].shape, self.keypoints.shape)
        self.assertEqual(encoded_data['area'].shape, self.area.shape)

        # Check if the keypoints were normalized correctly
        expected_keypoints = self.keypoints / np.array(
            self.img_shape, dtype=np.float32)
        np.testing.assert_array_almost_equal(encoded_data['keypoints'],
                                             expected_keypoints)

        # Check if the area was normalized correctly
        expected_area = self.area / float(
            self.img_shape[0] * self.img_shape[1])
        np.testing.assert_array_almost_equal(encoded_data['area'],
                                             expected_area)

    def test_decode(self):
        # Dummy predictions for logits, boxes, and keypoints
        pred_logits = np.array([0.7, 0.6]).reshape(2, 1)
        pred_boxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.8, 0.8]])
        pred_keypoints = np.array([[0.2, 0.3, 1, 0.3, 0.4, 1],
                                   [0.6, 0.7, 1, 0.7, 0.8, 1]])
        input_shapes = np.array(self.img_shape)

        # Test decoding
        boxes, keypoints, scores = self.encoder.decode(
            input_shapes=input_shapes,
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            pred_keypoints=pred_keypoints)

        self.assertEqual(boxes.shape, pred_boxes.shape)
        self.assertEqual(keypoints.shape, (self.encoder.num_select,
                                           self.encoder.num_keypoints, 2))
        self.assertEqual(scores.shape,
                         (self.encoder.num_select, self.encoder.num_keypoints))
