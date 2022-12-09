# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import SimCCLabel  # noqa: F401
from mmpose.registry import KEYPOINT_CODECS


class TestSimCCLabel(TestCase):

    # name and configs of all test cases
    def setUp(self) -> None:
        self.configs = [
            (
                'simcc gaussian',
                dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    smoothing_type='gaussian',
                    sigma=6.0,
                    simcc_split_ratio=2.0),
            ),
            (
                'simcc smoothing',
                dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    smoothing_type='standard',
                    sigma=5.0,
                    simcc_split_ratio=3.0,
                    label_smooth_weight=0.1),
            ),
            (
                'simcc one-hot',
                dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    smoothing_type='standard',
                    sigma=5.0,
                    simcc_split_ratio=3.0),
            ),
            (
                'simcc dark',
                dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    smoothing_type='gaussian',
                    sigma=6.0,
                    simcc_split_ratio=2.0,
                    use_dark=True),
            ),
            (
                'simcc separated sigmas',
                dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    smoothing_type='gaussian',
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2.0),
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

            self.assertEqual(encoded['keypoint_x_labels'].shape,
                             (1, 17, int(192 * codec.simcc_split_ratio)),
                             f'Failed case: "{name}"')
            self.assertEqual(encoded['keypoint_y_labels'].shape,
                             (1, 17, int(256 * codec.simcc_split_ratio)),
                             f'Failed case: "{name}"')
            self.assertEqual(encoded['keypoint_weights'].shape, (1, 17),
                             f'Failed case: "{name}"')

    def test_decode(self):
        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            simcc_x = np.random.rand(1, 17, int(192 * codec.simcc_split_ratio))
            simcc_y = np.random.rand(1, 17, int(256 * codec.simcc_split_ratio))

            keypoints, scores = codec.decode(simcc_x, simcc_y)

            self.assertEqual(keypoints.shape, (1, 17, 2),
                             f'Failed case: "{name}"')
            self.assertEqual(scores.shape, (1, 17), f'Failed case: "{name}"')

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']

        for name, cfg in self.configs:
            codec = KEYPOINT_CODECS.build(cfg)

            encoded = codec.encode(keypoints, keypoints_visible)
            keypoint_x_labels = encoded['keypoint_x_labels']
            keypoint_y_labels = encoded['keypoint_y_labels']

            _keypoints, _ = codec.decode(keypoint_x_labels, keypoint_y_labels)

            self.assertTrue(
                np.allclose(keypoints, _keypoints, atol=5.),
                f'Failed case: "{name}"')

    def test_errors(self):
        cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='uniform',
            sigma=1.0,
            simcc_split_ratio=2.0)

        with self.assertRaisesRegex(ValueError,
                                    'got invalid `smoothing_type`'):
            _ = KEYPOINT_CODECS.build(cfg)

        # invalid label_smooth_weight in smoothing
        cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='standard',
            sigma=1.0,
            simcc_split_ratio=2.0,
            label_smooth_weight=1.1)

        with self.assertRaisesRegex(ValueError,
                                    '`label_smooth_weight` should be'):
            _ = KEYPOINT_CODECS.build(cfg)

        # invalid label_smooth_weight for gaussian
        cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=1.0,
            simcc_split_ratio=2.0,
            label_smooth_weight=0.1)

        with self.assertRaisesRegex(ValueError,
                                    'is only used for `standard` mode.'):
            _ = KEYPOINT_CODECS.build(cfg)
