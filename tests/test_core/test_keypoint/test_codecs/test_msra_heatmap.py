# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.core.keypoint.codecs import MSRAHeatmap  # noqa: F401
from mmpose.registry import KEYPOINT_CODECS


class TestMSRAHeatmap(TestCase):

    def setUp(self) -> None:
        self.configs = [
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.0)
        ]

        self.data = dict(
            keypoints=np.random.rand(1, 17, 2).astype(np.float32) * [192, 256],
            keypoints_visible=np.ones((1, 17), dtype=np.float32),
            heatmaps=np.random.rand(17, 64, 48).astype(np.float32))

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            heatmaps, keypoint_weights = codec.encode(keypoints,
                                                      keypoints_visible)

            self.assertEqual(heatmaps.shape, (17, 64, 48))
            self.assertEqual(keypoint_weights.shape, (17, ))

    def test_decode(self):
        heatmaps = self.data['heatmaps']

        for cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            keypoints, scores = codec.decode(heatmaps)

            self.assertEqual(keypoints.shape, (1, 17, 2))
            self.assertEqual(scores.shape, (1, 17))

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            heatmaps, _ = codec.encode(keypoints, keypoints_visible)
            _keypoints, _ = codec.decode(heatmaps)

            self.assertTrue(np.allclose(keypoints, _keypoints))
