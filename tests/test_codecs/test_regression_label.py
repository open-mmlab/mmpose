# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import RegressionLabel  # noqa: F401
from mmpose.registry import KEYPOINT_CODECS


class TestRegressionLabel(TestCase):

    # name and configs of all test cases
    def setUp(self) -> None:
        self.configs = [
            (
                'regression',
                dict(
                    type='RegressionLabel',
                    input_size=(192, 256),
                ),
            ),
        ]

        # The bbox is usually padded so the keypoint will not be near the
        # boundary
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        encoded_with_sigma = np.random.rand(1, 17, 4)
        encoded_wo_sigma = np.random.rand(1, 17, 2)
        keypoints_visible = np.ones((1, 17), dtype=np.float32)
        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            encoded_with_sigma=encoded_with_sigma,
            encoded_wo_sigma=encoded_wo_sigma)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)

            self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(encoded['keypoint_weights'].shape, (1, 17),
                             f'Failed case: "{name}"')

    def test_decode(self):
        encoded_with_sigma = self.data['encoded_with_sigma']
        encoded_wo_sigma = self.data['encoded_wo_sigma']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            keypoints1, scores1 = codec.decode(encoded_with_sigma)
            keypoints2, scores2 = codec.decode(encoded_wo_sigma)

            self.assertEqual(keypoints1.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores1.shape, (1, 17), f'Failed case: "{name}"')
            self.assertEqual(keypoints2.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores2.shape, (1, 17), f'Failed case: "{name}"')

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)

            _keypoints, _ = codec.decode(encoded['keypoint_labels'])

            self.assertTrue(
                np.allclose(keypoints, _keypoints, atol=5.),
                f'Failed case: "{name}"')
