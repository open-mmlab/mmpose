# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.structures import keypoint_clip_border


class TestKeypointClipBorder(TestCase):

    def test_keypoint_clip_border(self):
        keypoints = np.array([[[10, 20], [30, 40], [-5, 25], [50, 60]]])
        keypoints_visible = np.array([[1.0, 0.8, 0.5, 1.0]])
        shape = (50, 50)  # Example frame shape

        clipped_keypoints, clipped_keypoints_visible = keypoint_clip_border(
            keypoints, keypoints_visible, shape)

        # Check if keypoints outside the frame have visibility set to 0.0
        self.assertEqual(clipped_keypoints_visible[0, 2], 0.0)
        self.assertEqual(clipped_keypoints_visible[0, 3], 0.0)

        # Check if keypoints inside the frame have unchanged visibility values
        self.assertEqual(clipped_keypoints_visible[0, 0], 1.0)
        self.assertEqual(clipped_keypoints_visible[0, 1], 0.8)

        # Check if keypoints array shapes remain unchanged
        self.assertEqual(keypoints.shape, clipped_keypoints.shape)
        self.assertEqual(keypoints_visible.shape,
                         clipped_keypoints_visible.shape)

        keypoints = np.array([[[10, 20], [30, 40], [-5, 25], [50, 60]]])
        keypoints_visible = np.array([[1.0, 0.8, 0.5, 1.0]])
        keypoints_visible_weight = np.array([[1.0, 0.0, 1.0, 1.0]])
        keypoints_visible = np.stack(
            (keypoints_visible, keypoints_visible_weight), axis=-1)
        shape = (50, 50)  # Example frame shape

        clipped_keypoints, clipped_keypoints_visible = keypoint_clip_border(
            keypoints, keypoints_visible, shape)

        # Check if keypoints array shapes remain unchanged
        self.assertEqual(keypoints.shape, clipped_keypoints.shape)
        self.assertEqual(keypoints_visible.shape,
                         clipped_keypoints_visible.shape)

        # Check if keypoints outside the frame have visibility set to 0.0
        self.assertEqual(clipped_keypoints_visible[0, 2, 0], 0.0)
        self.assertEqual(clipped_keypoints_visible[0, 3, 0], 0.0)

        # Check if keypoints inside the frame have unchanged visibility values
        self.assertEqual(clipped_keypoints_visible[0, 0, 0], 1.0)
        self.assertEqual(clipped_keypoints_visible[0, 1, 0], 0.8)

        # Check if the visibility weights remain unchanged
        self.assertSequenceEqual(clipped_keypoints_visible[..., 1].tolist(),
                                 keypoints_visible[..., 1].tolist())
