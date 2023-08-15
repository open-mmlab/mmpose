# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.registry import KEYPOINT_CODECS


class TestHand3DHeatmap(TestCase):

    def build_hand_3d_heatmap(self, **kwargs):
        cfg = dict(type='Hand3DHeatmap')
        cfg.update(kwargs)
        return KEYPOINT_CODECS.build(cfg)

    def setUp(self) -> None:
        # The bbox is usually padded so the keypoint will not be near the
        # boundary
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 3))
        keypoints[..., :2] = keypoints[..., :2] * [256, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.ones((
            1,
            17,
        ), dtype=np.float32)
        heatmaps = np.random.rand(17, 64, 64, 64).astype(np.float32)
        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            heatmaps=heatmaps)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        # test default settings
        codec = self.build_hand_3d_heatmap()

        encoded = codec.encode(keypoints, keypoints_visible)

        self.assertEqual(encoded['heatmaps'].shape, (17 * 64, 64, 64))
        self.assertEqual(encoded['keypoint_weights'].shape, (
            1,
            17,
        ))

        # test with different joint weights
        codec = self.build_hand_3d_heatmap(use_different_joint_weights=True)

        encoded = codec.encode(
            keypoints,
            keypoints_visible,
            dataset_keypoint_weights=np.ones(17, ))

        self.assertEqual(encoded['heatmaps'].shape, (17 * 64, 64, 64))
        self.assertEqual(encoded['keypoint_weights'].shape, (
            1,
            17,
        ))

        # test joint_indices
        codec = self.build_hand_3d_heatmap(joint_indices=[0, 8, 16])
        encoded = codec.encode(keypoints, keypoints_visible)
        self.assertEqual(encoded['heatmaps'].shape, (3 * 64, 64, 64))
        self.assertEqual(encoded['keypoint_weights'].shape, (
            1,
            3,
        ))

    def test_decode(self):
        heatmaps = self.data['heatmaps']

        # test default settings
        codec = self.build_hand_3d_heatmap()

        keypoints, scores, _, _ = codec.decode(heatmaps, np.ones((1, )),
                                               np.ones((1, 2)))

        self.assertEqual(keypoints.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        codec = self.build_hand_3d_heatmap()

        encoded = codec.encode(keypoints, keypoints_visible)
        _keypoints, _, _, _ = codec.decode(
            encoded['heatmaps'].reshape(17, 64, 64, 64), np.ones((1, )),
            np.ones((1, 2)))

        self.assertTrue(
            np.allclose(keypoints[..., :2], _keypoints[..., :2], atol=5.))
