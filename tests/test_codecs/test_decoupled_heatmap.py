# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import DecoupledHeatmap
from mmpose.registry import KEYPOINT_CODECS
from mmpose.testing import get_coco_sample


class TestDecoupledHeatmap(TestCase):

    def setUp(self) -> None:
        pass

    def _make_multi_instance_data(self, data):
        bbox = data['bbox'].reshape(-1, 2, 2)
        keypoints = data['keypoints']
        keypoints_visible = data['keypoints_visible']

        keypoints_visible[..., 0] = 0

        offset = keypoints.max(axis=1, keepdims=True)
        bbox_outside = bbox - offset
        keypoints_outside = keypoints - offset
        keypoints_outside_visible = np.zeros(keypoints_visible.shape)

        bbox_overlap = bbox.mean(
            axis=1, keepdims=True) + 0.8 * (
                bbox - bbox.mean(axis=1, keepdims=True))
        keypoint_overlap = keypoints.mean(
            axis=1, keepdims=True) + 0.8 * (
                keypoints - keypoints.mean(axis=1, keepdims=True))
        keypoint_overlap_visible = keypoints_visible

        data['bbox'] = np.concatenate((bbox, bbox_outside, bbox_overlap),
                                      axis=0)
        data['keypoints'] = np.concatenate(
            (keypoints, keypoints_outside, keypoint_overlap), axis=0)
        data['keypoints_visible'] = np.concatenate(
            (keypoints_visible, keypoints_outside_visible,
             keypoint_overlap_visible),
            axis=0)

        return data

    def test_build(self):
        cfg = dict(
            type='DecoupledHeatmap',
            input_size=(512, 512),
            heatmap_size=(128, 128),
        )
        codec = KEYPOINT_CODECS.build(cfg)
        self.assertIsInstance(codec, DecoupledHeatmap)

    def test_encode(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)
        data['bbox'] = np.tile(data['bbox'], 2).reshape(-1, 4, 2)
        data['bbox'][:, 1:3, 0] = data['bbox'][:, 0:2, 0]
        data = self._make_multi_instance_data(data)

        codec = DecoupledHeatmap(
            input_size=(512, 512),
            heatmap_size=(128, 128),
        )

        print(data['bbox'].shape)
        encoded = codec.encode(
            data['keypoints'], data['keypoints_visible'], bbox=data['bbox'])

        heatmaps = encoded['heatmaps']
        instance_heatmaps = encoded['instance_heatmaps']
        keypoint_weights = encoded['keypoint_weights']
        instance_coords = encoded['instance_coords']

        self.assertEqual(heatmaps.shape, (18, 128, 128))
        self.assertEqual(keypoint_weights.shape, (2, 17))
        self.assertEqual(instance_heatmaps.shape, (34, 128, 128))
        self.assertEqual(instance_coords.shape, (2, 2))

        # without bbox
        encoded = codec.encode(
            data['keypoints'], data['keypoints_visible'], bbox=None)

        heatmaps = encoded['heatmaps']
        instance_heatmaps = encoded['instance_heatmaps']
        keypoint_weights = encoded['keypoint_weights']
        instance_coords = encoded['instance_coords']

        self.assertEqual(heatmaps.shape, (18, 128, 128))
        self.assertEqual(keypoint_weights.shape, (2, 17))
        self.assertEqual(instance_heatmaps.shape, (34, 128, 128))
        self.assertEqual(instance_coords.shape, (2, 2))

        # root_type
        with self.assertRaises(ValueError):
            codec = DecoupledHeatmap(
                input_size=(512, 512),
                heatmap_size=(128, 128),
                root_type='box_center',
            )
            encoded = codec.encode(
                data['keypoints'],
                data['keypoints_visible'],
                bbox=data['bbox'])

        codec = DecoupledHeatmap(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            root_type='bbox_center',
        )

        encoded = codec.encode(
            data['keypoints'], data['keypoints_visible'], bbox=data['bbox'])

        heatmaps = encoded['heatmaps']
        instance_heatmaps = encoded['instance_heatmaps']
        keypoint_weights = encoded['keypoint_weights']
        instance_coords = encoded['instance_coords']

        self.assertEqual(heatmaps.shape, (18, 128, 128))
        self.assertEqual(keypoint_weights.shape, (2, 17))
        self.assertEqual(instance_heatmaps.shape, (34, 128, 128))
        self.assertEqual(instance_coords.shape, (2, 2))

    def test_decode(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=2)
        data['bbox'] = np.tile(data['bbox'], 2).reshape(-1, 4, 2)
        data['bbox'][:, 1:3, 0] = data['bbox'][:, 0:2, 0]

        codec = DecoupledHeatmap(
            input_size=(512, 512),
            heatmap_size=(128, 128),
        )

        encoded = codec.encode(
            data['keypoints'], data['keypoints_visible'], bbox=data['bbox'])
        instance_heatmaps = encoded['instance_heatmaps'].reshape(
            encoded['instance_coords'].shape[0], -1,
            *encoded['instance_heatmaps'].shape[-2:])
        instance_scores = np.ones(encoded['instance_coords'].shape[0])
        decoded = codec.decode(instance_heatmaps, instance_scores[:, None])
        keypoints, keypoint_scores = decoded

        self.assertEqual(keypoints.shape, (2, 17, 2))
        self.assertEqual(keypoint_scores.shape, (2, 17))

    def test_cicular_verification(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)
        data['bbox'] = np.tile(data['bbox'], 2).reshape(-1, 4, 2)
        data['bbox'][:, 1:3, 0] = data['bbox'][:, 0:2, 0]

        codec = DecoupledHeatmap(
            input_size=(512, 512),
            heatmap_size=(128, 128),
        )

        encoded = codec.encode(
            data['keypoints'], data['keypoints_visible'], bbox=data['bbox'])
        instance_heatmaps = encoded['instance_heatmaps'].reshape(
            encoded['instance_coords'].shape[0], -1,
            *encoded['instance_heatmaps'].shape[-2:])
        instance_scores = np.ones(encoded['instance_coords'].shape[0])
        decoded = codec.decode(instance_heatmaps, instance_scores[:, None])
        keypoints, _ = decoded
        keypoints += 1.5

        self.assertTrue(np.allclose(keypoints, data['keypoints'], atol=5.))
