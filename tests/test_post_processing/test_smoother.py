# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union
from unittest import TestCase

import numpy as np
from mmcv import is_list_of

from mmpose.core.post_processing.smoother import Smoother


class TestSmoother(TestCase):

    def build_smoother(self):
        smoother = Smoother(
            'configs/_base_/filters/gaussian.py', keypoint_dim=2)
        return smoother

    def build_pose_results(self,
                           num_target: Union[int, List[int]],
                           num_frame: int = -1,
                           has_track_id: bool = True):
        keypoint_shape = (17, 2)
        results = []

        if isinstance(num_target, list):
            num_frame = len(num_target)
        else:
            assert num_frame >= 0
            num_target = [num_target] * num_frame

        for n in num_target:
            results_t = []
            for idx in range(n):
                result = dict(keypoints=np.random.rand(*keypoint_shape))
                if has_track_id:
                    result['track_id'] = str(idx)
                results_t.append(result)
            results.append(results_t)
        return results

    def test_corner_cases(self):
        # Test empty input
        smoother = self.build_smoother()
        results = []
        with self.assertWarnsRegex(UserWarning,
                                   'Smoother received empty result.'):
            _ = smoother.smooth(results)

        # Test inconsistent tracked poses
        smoother = self.build_smoother()
        results = self.build_pose_results(num_target=[1, 2], has_track_id=True)
        with self.assertRaisesRegex(ValueError, 'Inconsistent track ids'):
            _ = smoother.smooth(results)

        # Test inconsistent untracked poses
        smoother = self.build_smoother()
        results = self.build_pose_results(
            num_target=[1, 2], has_track_id=False)
        with self.assertRaisesRegex(ValueError, 'Inconsistent target number'):
            _ = smoother.smooth(results)

    def test_smooth_online_with_trackid(self):
        smoother = self.build_smoother()
        num_target = [2] * 10 + [3] * 10
        results = self.build_pose_results(
            num_target=num_target, has_track_id=True)
        for results_t in results:
            smoothed_results_t = smoother.smooth(results_t)

            # Sort by track_id
            results_t.sort(key=lambda x: x['track_id'])
            smoothed_results_t.sort(key=lambda x: x['track_id'])

            # Check the output is non-nested list
            self.assertTrue(is_list_of(smoothed_results_t, dict))
            # Check the target number in the frame is correct
            self.assertEqual(len(smoothed_results_t), len(results_t))

            for result, smoothed_result in zip(results_t, smoothed_results_t):
                # Check the target_id is correct
                self.assertEqual(result['track_id'],
                                 smoothed_result['track_id'])
                # Check the pose shape is correct
                self.assertEqual(result['keypoints'].shape,
                                 smoothed_result['keypoints'].shape)

    def test_smooth_online_wo_trackid(self):
        smoother = self.build_smoother()
        num_target = [2] * 10 + [3] * 10
        results = self.build_pose_results(
            num_target=num_target, has_track_id=False)
        for results_t in results:
            smoothed_results_t = smoother.smooth(results_t)

            # Check the output is non-nested list
            self.assertTrue(is_list_of(smoothed_results_t, dict))
            # Check the target number in the frame is correct
            self.assertEqual(len(smoothed_results_t), len(results_t))

            for result, smoothed_result in zip(results_t, smoothed_results_t):
                # Check the pose shape is correct
                self.assertEqual(result['keypoints'].shape,
                                 smoothed_result['keypoints'].shape)

    def test_smooth_offline_with_trackid(self):
        smoother = self.build_smoother()
        results = self.build_pose_results(
            num_target=2, num_frame=20, has_track_id=True)
        smoothed_results = smoother.smooth(results)
        for results_t, smoothed_results_t in zip(results, smoothed_results):
            # Sort by track_id
            results_t.sort(key=lambda x: x['track_id'])
            smoothed_results_t.sort(key=lambda x: x['track_id'])

            # Check the output is non-nested list
            self.assertTrue(is_list_of(smoothed_results_t, dict))
            # Check the target number in the frame is correct
            self.assertEqual(len(smoothed_results_t), len(results_t))

            for result, smoothed_result in zip(results_t, smoothed_results_t):
                # Check the target_id is correct
                self.assertEqual(result['track_id'],
                                 smoothed_result['track_id'])
                # Check the pose shape is correct
                self.assertEqual(result['keypoints'].shape,
                                 smoothed_result['keypoints'].shape)

    def test_smooth_offline_wo_trackid(self):
        smoother = self.build_smoother()
        results = self.build_pose_results(
            num_target=2, num_frame=20, has_track_id=False)
        smoothed_results = smoother.smooth(results)

        for results_t, smoothed_results_t in zip(results, smoothed_results):
            # Check the output is non-nested list
            self.assertTrue(is_list_of(smoothed_results_t, dict))
            # Check the target number in the frame is correct
            self.assertEqual(len(smoothed_results_t), len(results_t))

            for result, smoothed_result in zip(results_t, smoothed_results_t):
                # Check the pose shape is correct
                self.assertEqual(result['keypoints'].shape,
                                 smoothed_result['keypoints'].shape)
