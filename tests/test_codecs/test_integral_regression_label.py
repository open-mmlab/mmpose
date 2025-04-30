# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import IntegralRegressionLabel  # noqa: F401
from mmpose.registry import KEYPOINT_CODECS


class TestRegressionLabel(TestCase):

    # name and configs of all test cases
    def setUp(self) -> None:
        self.configs = [
            (
                'ipr',
                dict(
                    type='IntegralRegressionLabel',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    sigma=2),
            ),
        ]

        # The bbox is usually padded so the keypoint will not be near the
        # boundary
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        heatmaps = np.random.rand(17, 64, 48).astype(np.float32)
        encoded_wo_sigma = np.random.rand(1, 17, 2)
        keypoints_visible = np.ones((1, 17), dtype=np.float32)
        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            heatmaps=heatmaps,
            encoded_wo_sigma=encoded_wo_sigma)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)
            heatmaps = encoded['heatmaps']
            keypoint_labels = encoded['keypoint_labels']
            keypoint_weights = encoded['keypoint_weights']

            self.assertEqual(heatmaps.shape, (17, 64, 48),
                             f'Failed case: "{name}"')
            self.assertEqual(keypoint_labels.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(keypoint_weights.shape, (1, 17),
                             f'Failed case: "{name}"')

    def test_decode(self):
        encoded_wo_sigma = self.data['encoded_wo_sigma']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            keypoints, scores = codec.decode(encoded_wo_sigma)

            self.assertEqual(keypoints.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores.shape, (1, 17), f'Failed case: "{name}"')

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)
            keypoint_labels = encoded['keypoint_labels']

            _keypoints, _ = codec.decode(keypoint_labels)

            self.assertTrue(
                np.allclose(keypoints, _keypoints, atol=5.),
                f'Failed case: "{name}"')
