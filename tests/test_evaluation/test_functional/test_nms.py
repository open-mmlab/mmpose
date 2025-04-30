# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmpose.evaluation.functional.nms import nearby_joints_nms, nms_torch


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


class TestNMSTorch(TestCase):

    def test_nms_torch(self):
        bboxes = torch.tensor([[0, 0, 3, 3], [1, 0, 3, 3], [4, 4, 6, 6]],
                              dtype=torch.float32)

        scores = torch.tensor([0.9, 0.8, 0.7])

        expected_result = torch.tensor([0, 2])
        result = nms_torch(bboxes, scores, threshold=0.5)
        self.assertTrue(torch.equal(result, expected_result))

        expected_result = [torch.tensor([0, 1]), torch.tensor([2])]
        result = nms_torch(bboxes, scores, threshold=0.5, return_group=True)
        for res_out, res_expected in zip(result, expected_result):
            self.assertTrue(torch.equal(res_out, res_expected))
