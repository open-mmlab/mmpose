# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import MegviiHeatmap
from mmpose.registry import KEYPOINT_CODECS


class TestMegviiHeatmap(TestCase):

    def setUp(self) -> None:
        # name and configs of all test cases
        self.configs = [
            (
                'megvii',
                dict(
                    type='MegviiHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    kernel_size=11),
            ),
        ]

        # The bbox is usually padded so the keypoint will not be near the
        # boundary
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.ones((1, 17), dtype=np.float32)
        heatmaps = np.random.rand(17, 64, 48).astype(np.float32)
        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            heatmaps=heatmaps)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            heatmaps, keypoint_weights = codec.encode(keypoints,
                                                      keypoints_visible)

            self.assertEqual(heatmaps.shape, (17, 64, 48),
                             f'Failed case: "{name}"')
            self.assertEqual(keypoint_weights.shape,
                             (1, 17)), f'Failed case: "{name}"'

    def test_decode(self):
        heatmaps = self.data['heatmaps']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            keypoints, scores = codec.decode(heatmaps)

            self.assertEqual(keypoints.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores.shape, (1, 17), f'Failed case: "{name}"')

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            heatmaps, _ = codec.encode(keypoints, keypoints_visible)
            _keypoints, _ = codec.decode(heatmaps)

            self.assertTrue(
                np.allclose(keypoints, _keypoints, atol=5.),
                f'Failed case: "{name}"')

    def test_errors(self):
        # multiple instance
        codec = MegviiHeatmap(
            input_size=(192, 256), heatmap_size=(48, 64), kernel_size=11)
        keypoints = np.random.rand(2, 17, 2)
        keypoints_visible = np.random.rand(2, 17)

        with self.assertRaisesRegex(AssertionError,
                                    'only support single-instance'):
            codec.encode(keypoints, keypoints_visible)
