# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.evaluation.functional import (keypoint_auc, keypoint_epe,
                                          keypoint_mpjpe, keypoint_nme,
                                          keypoint_pck_accuracy,
                                          multilabel_classification_accuracy,
                                          pose_pck_accuracy)


class TestKeypointEval(TestCase):

    def test_keypoint_pck_accuracy(self):

        output = np.zeros((2, 5, 2))
        target = np.zeros((2, 5, 2))
        mask = np.array([[True, True, False, True, True],
                         [True, True, False, True, True]])

        # first channel
        output[0, 0] = [10, 0]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [20, 20]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [-1, 0]
        # fourth channel
        output[0, 3] = [30, 30]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [0, 10]
        target[0, 4] = [0, 10]

        thr = np.full((2, 2), 10, dtype=np.float32)

        acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5,
                                                  thr)

        assert_array_almost_equal(acc, np.array([1, 0.5, -1, 1, 1]), decimal=4)
        self.assertAlmostEqual(avg_acc, 0.875, delta=1e-4)
        self.assertAlmostEqual(cnt, 4, delta=1e-4)

        acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5,
                                                  np.zeros((2, 2)))
        assert_array_almost_equal(
            acc, np.array([-1, -1, -1, -1, -1]), decimal=4)
        self.assertAlmostEqual(avg_acc, 0, delta=1e-4)
        self.assertAlmostEqual(cnt, 0, delta=1e-4)

        acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5,
                                                  np.array([[0, 0], [10, 10]]))
        assert_array_almost_equal(acc, np.array([1, 1, -1, 1, 1]), decimal=4)
        self.assertAlmostEqual(avg_acc, 1, delta=1e-4)
        self.assertAlmostEqual(cnt, 4, delta=1e-4)

    def test_keypoint_auc(self):
        output = np.zeros((1, 5, 2))
        target = np.zeros((1, 5, 2))
        mask = np.array([[True, True, False, True, True]])
        # first channel
        output[0, 0] = [10, 4]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [10, 18]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [0, -1]
        # fourth channel
        output[0, 3] = [40, 40]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [20, 10]
        target[0, 4] = [0, 10]

        auc = keypoint_auc(output, target, mask, 20, 4)
        self.assertAlmostEqual(auc, 0.375, delta=1e-4)

    def test_keypoint_epe(self):
        output = np.zeros((1, 5, 2))
        target = np.zeros((1, 5, 2))
        mask = np.array([[True, True, False, True, True]])
        # first channel
        output[0, 0] = [10, 4]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [10, 18]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [-1, -1]
        # fourth channel
        output[0, 3] = [40, 40]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [20, 10]
        target[0, 4] = [0, 10]

        epe = keypoint_epe(output, target, mask)
        self.assertAlmostEqual(epe, 11.5355339, delta=1e-4)

    def test_keypoint_nme(self):
        output = np.zeros((1, 5, 2))
        target = np.zeros((1, 5, 2))
        mask = np.array([[True, True, False, True, True]])
        # first channel
        output[0, 0] = [10, 4]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [10, 18]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [-1, -1]
        # fourth channel
        output[0, 3] = [40, 40]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [20, 10]
        target[0, 4] = [0, 10]

        normalize_factor = np.ones((output.shape[0], output.shape[2]))

        nme = keypoint_nme(output, target, mask, normalize_factor)
        self.assertAlmostEqual(nme, 11.5355339, delta=1e-4)

    def test_pose_pck_accuracy(self):
        output = np.zeros((1, 5, 64, 64), dtype=np.float32)
        target = np.zeros((1, 5, 64, 64), dtype=np.float32)
        mask = np.array([[True, True, False, False, False]])
        # first channel
        output[0, 0, 20, 20] = 1
        target[0, 0, 10, 10] = 1
        # second channel
        output[0, 1, 30, 30] = 1
        target[0, 1, 30, 30] = 1

        acc, avg_acc, cnt = pose_pck_accuracy(output, target, mask)

        assert_array_almost_equal(acc, np.array([0, 1, -1, -1, -1]), decimal=4)
        self.assertAlmostEqual(avg_acc, 0.5, delta=1e-4)
        self.assertAlmostEqual(cnt, 2, delta=1e-4)

    def test_multilabel_classification_accuracy(self):
        output = np.array([[0.7, 0.8, 0.4], [0.8, 0.1, 0.1]])
        target = np.array([[1, 0, 0], [1, 0, 1]])
        mask = np.array([[True, True, True], [True, True, True]])
        thr = 0.5
        acc = multilabel_classification_accuracy(output, target, mask, thr)
        self.assertEqual(acc, 0)

        output = np.array([[0.7, 0.2, 0.4], [0.8, 0.1, 0.9]])
        thr = 0.5
        acc = multilabel_classification_accuracy(output, target, mask, thr)
        self.assertEqual(acc, 1)

        thr = 0.3
        acc = multilabel_classification_accuracy(output, target, mask, thr)
        self.assertEqual(acc, 0.5)

        mask = np.array([[True, True, False], [True, True, True]])
        acc = multilabel_classification_accuracy(output, target, mask, thr)
        self.assertEqual(acc, 1)

    def test_keypoint_mpjpe(self):
        output = np.zeros((2, 5, 3))
        target = np.zeros((2, 5, 3))
        mask = np.array([[True, True, False, True, True],
                         [True, True, False, True, True]])

        # first channel
        output[0, 0] = [1, 0, 0]
        target[0, 0] = [1, 0, 0]
        output[1, 0] = [1, 0, 0]
        target[1, 0] = [1, 1, 0]
        # second channel
        output[0, 1] = [2, 2, 0]
        target[0, 1] = [1, 1, 1]
        output[1, 1] = [2, 2, 1]
        target[1, 1] = [1, 0, 1]
        # third channel
        output[0, 2] = [0, 0, -1]
        target[0, 2] = [-1, 0, 0]
        output[1, 2] = [-1, 0, 0]
        target[1, 2] = [-1, 0, 0]
        # fourth channel
        output[0, 3] = [3, 3, 1]
        target[0, 3] = [3, 3, 1]
        output[1, 3] = [0, 0, 3]
        target[1, 3] = [0, 0, 3]
        # fifth channel
        output[0, 4] = [0, 1, 1]
        target[0, 4] = [0, 1, 0]
        output[1, 4] = [0, 0, 1]
        target[1, 4] = [1, 1, 0]

        mpjpe = keypoint_mpjpe(output, target, mask)
        self.assertAlmostEqual(mpjpe, 0.9625211990796929, delta=1e-4)

        p_mpjpe = keypoint_mpjpe(output, target, mask, 'procrustes')
        self.assertAlmostEqual(p_mpjpe, 1.0047897634604497, delta=1e-4)

        s_mpjpe = keypoint_mpjpe(output, target, mask, 'scale')
        self.assertAlmostEqual(s_mpjpe, 1.0277129678465953, delta=1e-4)

        with self.assertRaises(ValueError):
            _ = keypoint_mpjpe(output, target, mask, 'alignment')
