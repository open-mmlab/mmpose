# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.visualization import FastVisualizer


class TestFastVisualizer(TestCase):

    def setUp(self):
        self.metainfo = {
            'keypoint_id2name': {
                0: 'nose',
                1: 'left_eye',
                2: 'right_eye'
            },
            'keypoint_name2id': {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2
            },
            'keypoint_colors': np.array([[255, 0, 0], [0, 255, 0], [0, 0,
                                                                    255]]),
            'skeleton_links': [(0, 1), (1, 2)],
            'skeleton_link_colors': np.array([[255, 255, 0], [255, 0, 255]])
        }
        self.visualizer = FastVisualizer(self.metainfo)

    def test_init(self):
        self.assertEqual(self.visualizer.radius, 6)
        self.assertEqual(self.visualizer.line_width, 3)
        self.assertEqual(self.visualizer.kpt_thr, 0.3)
        self.assertEqual(self.visualizer.keypoint_id2name,
                         self.metainfo['keypoint_id2name'])
        self.assertEqual(self.visualizer.keypoint_name2id,
                         self.metainfo['keypoint_name2id'])
        np.testing.assert_array_equal(self.visualizer.keypoint_colors,
                                      self.metainfo['keypoint_colors'])
        self.assertEqual(self.visualizer.skeleton_links,
                         self.metainfo['skeleton_links'])
        np.testing.assert_array_equal(self.visualizer.skeleton_link_colors,
                                      self.metainfo['skeleton_link_colors'])

    def test_draw_pose(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        instances = type('Instances', (object, ), {})()
        instances.keypoints = np.array([[[100, 100], [200, 200], [300, 300]]],
                                       dtype=np.float32)
        instances.keypoint_scores = np.array([[0.5, 0.5, 0.5]],
                                             dtype=np.float32)

        self.visualizer.draw_pose(img, instances)

        # Check if keypoints are drawn
        self.assertNotEqual(img[100, 100].tolist(), [0, 0, 0])
        self.assertNotEqual(img[200, 200].tolist(), [0, 0, 0])
        self.assertNotEqual(img[300, 300].tolist(), [0, 0, 0])

        # Check if skeleton links are drawn
        self.assertNotEqual(img[150, 150].tolist(), [0, 0, 0])
        self.assertNotEqual(img[250, 250].tolist(), [0, 0, 0])

    def test_draw_pose_with_none_instances(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        instances = None

        self.visualizer.draw_pose(img, instances)

        # Check if the image is still empty (black)
        self.assertEqual(np.count_nonzero(img), 0)
