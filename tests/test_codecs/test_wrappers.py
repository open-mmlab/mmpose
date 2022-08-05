# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import MultiLevelHeatmapEncoder
from mmpose.registry import KEYPOINT_CODECS


class TestMultiLevelHeatmapEncoder(TestCase):

    def setUp(self) -> None:

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

        encoder_cfgs = [
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.0),
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=3.0)
        ]

        wrapper_cfg = dict(
            type='MultiLevelHeatmapEncoder', encoders=encoder_cfgs)
        encoders = [KEYPOINT_CODECS.build(cfg) for cfg in encoder_cfgs]
        wrapper = KEYPOINT_CODECS.build(wrapper_cfg)

        # encoding by wrapper
        heatmaps, keypoint_weights = wrapper.encode(keypoints,
                                                    keypoints_visible)

        # encoding by encoders
        all_heatmaps, all_keypoint_weights = zip(*[
            encoder.encode(keypoints, keypoints_visible)
            for encoder in encoders
        ])

        gt_heatmaps = np.concatenate(all_heatmaps)
        gt_keypoint_weights = np.stack(all_keypoint_weights, axis=1)

        self.assertTrue(np.allclose(heatmaps, gt_heatmaps))
        self.assertTrue(np.allclose(keypoint_weights, gt_keypoint_weights))

    def tset_errors(self):
        # encoder number is less than 2
        with self.assertRaisesRegex(AssertionError, 'more than 1'):
            _ = MultiLevelHeatmapEncoder(encoders=[])

        # invalid parameter type
        with self.assertRaisesRegex(AssertionError, 'more than 1'):
            _ = MultiLevelHeatmapEncoder(
                encoders=dict(
                    type='MSRAHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                ))

        # unmatched input size
        encoder_cfgs = [
            dict(
                type='MSRAHeatmap',
                input_size=(48, 64),
                heatmap_size=(48, 64),
                sigma=2.0),
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=3.0)
        ]

        with self.assertRaisesRegex(AssertionError, 'unmatched input_size'):
            _ = MultiLevelHeatmapEncoder(encoders=encoder_cfgs)

        # unmatched heatmap size
        encoder_cfgs = [
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.0),
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(72, 96),
                sigma=3.0)
        ]

        with self.assertRaisesRegex(AssertionError, 'unmatched heatmap_size'):
            _ = MultiLevelHeatmapEncoder(encoders=encoder_cfgs)

        # decode() envoded
        encoder_cfgs = [
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2.0),
            dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=3.0)
        ]
        wrapper = MultiLevelHeatmapEncoder(encoders=encoder_cfgs)

        with self.assertRaisesRegex(NotImplementedError,
                                    'does not support decoding'):
            wrapper.decode(None)
