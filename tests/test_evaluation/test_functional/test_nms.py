# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.evaluation.functional.nms import nearby_joints_nms


class TestNearbyJointsNMS(TestCase):

    def test_nearby_joints_nms(self):

        kpts_db = []
        keep_pose_inds = nearby_joints_nms(
            kpts_db, 0.05, score_per_joint=True, max_dets=1)
        self.assertEqual(len(keep_pose_inds), 0)

        kpts_db = []
        for _ in range(5):
            kpts_db.append(
                dict(keypoints=np.random.rand(3, 2), score=np.random.rand(3)))
        keep_pose_inds = nearby_joints_nms(
            kpts_db, 0.05, score_per_joint=True, max_dets=1)
        self.assertEqual(len(keep_pose_inds), 1)
        self.assertLess(keep_pose_inds[0], 5)

        kpts_db = []
        for _ in range(5):
            kpts_db.append(
                dict(keypoints=np.random.rand(3, 2), score=np.random.rand()))
        keep_pose_inds = nearby_joints_nms(
            kpts_db, 0.05, num_nearby_joints_thr=2)
        self.assertLessEqual(len(keep_pose_inds), 5)
        self.assertGreater(len(keep_pose_inds), 0)

        with self.assertRaises(AssertionError):
            _ = nearby_joints_nms(kpts_db, 0, num_nearby_joints_thr=2)

        with self.assertRaises(AssertionError):
            _ = nearby_joints_nms(kpts_db, 0.05, num_nearby_joints_thr=3)
