# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import UDPHeatmap
from mmpose.registry import KEYPOINT_CODECS


class TestUDPHeatmap(TestCase):

    def setUp(self) -> None:
        # name and configs of all test cases
        self.configs = [
            (
                'udp gaussian',
                dict(
                    type='UDPHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    heatmap_type='gaussian',
                ),
            ),
            (
                'udp combined',
                dict(
                    type='UDPHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    heatmap_type='combined'),
            ),
        ]

        # The bbox is usually padded so the keypoint will not be near the
        # boundary
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.ones((1, 17), dtype=np.float32)
        self.data = dict(
            keypoints=keypoints, keypoints_visible=keypoints_visible)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)

            if codec.heatmap_type == 'combined':
                channel_per_kpt = 3
            else:
                channel_per_kpt = 1

            self.assertEqual(encoded['heatmaps'].shape,
                             (channel_per_kpt * 17, 64, 48),
                             f'Failed case: "{name}"')
            self.assertEqual(encoded['keypoint_weights'].shape,
                             (1, 17)), f'Failed case: "{name}"'

    def test_decode(self):

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)
            if codec.heatmap_type == 'combined':
                channel_per_kpt = 3
            else:
                channel_per_kpt = 1

            heatmaps = np.random.rand(channel_per_kpt * 17, 64,
                                      48).astype(np.float32)

            keypoints, scores = codec.decode(heatmaps)

            self.assertEqual(keypoints.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores.shape, (1, 17), f'Failed case: "{name}"')

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)
            _keypoints, _ = codec.decode(encoded['heatmaps'])

            self.assertTrue(
                np.allclose(keypoints, _keypoints, atol=10.),
                f'Failed case: "{name}",{abs(keypoints - _keypoints) < 5.} ')

    def test_errors(self):
        # invalid heatmap type
        with self.assertRaisesRegex(ValueError, 'invalid `heatmap_type`'):
            _ = UDPHeatmap(
                input_size=(192, 256),
                heatmap_size=(48, 64),
                heatmap_type='invalid')

        # multiple instance
        codec = UDPHeatmap(input_size=(192, 256), heatmap_size=(48, 64))
        keypoints = np.random.rand(2, 17, 2)
        keypoints_visible = np.random.rand(2, 17)

        with self.assertRaisesRegex(AssertionError,
                                    'only support single-instance'):
            codec.encode(keypoints, keypoints_visible)
