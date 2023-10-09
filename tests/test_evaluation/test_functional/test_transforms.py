# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np

from mmpose.evaluation.functional import (transform_ann, transform_pred,
                                          transform_sigmas)


class TestKeypointEval(TestCase):

    def test_transform_sigmas(self):

        mapping = [(3, 0), (6, 1), (16, 2), (5, 3)]
        num_keypoints = 5
        sigmas = np.random.rand(17)
        new_sigmas = transform_sigmas(sigmas, num_keypoints, mapping)
        self.assertEqual(len(new_sigmas), 5)
        for i, j in mapping:
            self.assertEqual(sigmas[i], new_sigmas[j])

    def test_transform_ann(self):
        mapping = [(3, 0), (6, 1), (16, 2), (5, 3)]
        num_keypoints = 5

        kpt_info = dict(
            num_keypoints=17,
            keypoints=np.random.randint(3, size=(17 * 3, )).tolist())
        kpt_info_copy = deepcopy(kpt_info)

        _ = transform_ann(kpt_info, num_keypoints, mapping)

        self.assertEqual(kpt_info['num_keypoints'], 5)
        self.assertEqual(len(kpt_info['keypoints']), 15)
        for i, j in mapping:
            self.assertListEqual(kpt_info_copy['keypoints'][i * 3:i * 3 + 3],
                                 kpt_info['keypoints'][j * 3:j * 3 + 3])

    def test_transform_pred(self):
        mapping = [(3, 0), (6, 1), (16, 2), (5, 3)]
        num_keypoints = 5

        kpt_info = dict(
            num_keypoints=17,
            keypoints=np.random.randint(3, size=(
                1,
                17,
                3,
            )),
            keypoint_scores=np.ones((1, 17)))

        _ = transform_pred(kpt_info, num_keypoints, mapping)

        self.assertEqual(kpt_info['num_keypoints'], 5)
        self.assertEqual(len(kpt_info['keypoints']), 1)
